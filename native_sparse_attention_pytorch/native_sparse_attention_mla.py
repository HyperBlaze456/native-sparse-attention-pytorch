from __future__ import annotations

from copy import deepcopy
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn, arange, stack, cat, tensor, Tensor
from torch.nn import Module, ModuleList

from local_attention import LocalAttention

from rotary_embedding_torch import RotaryEmbedding

# einstein notation
import einx
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# flex attention (optional)
flex_attention = None
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass


# -------------------------------------------------------------
# Helper functions and mask creators with detailed comments
# -------------------------------------------------------------

def create_sliding_mask(seq_len, window_size):
    """
    Creates a causal sliding window mask.
    Query index must be >= key index and within the window_size difference.
    """

    def sliding_mask(_, __, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        sliding_mask = (q_idx - kv_idx) <= window_size
        return causal_mask & sliding_mask

    block_mask = create_block_mask(sliding_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
    return block_mask


def create_compress_mask(seq_len, kv_seq_len, compress_block_size, mem_kv_len=0):
    """
    Creates a mask for the compression branch.
    This mask shows which keys (after accounting for extra memory tokens) should be
    attended to for compression. (Note: In this implementation the mask is mainly used for computing importance scores.)
    """

    def compress_mask(_, __, q_idx, kv_idx):
        is_mem_kv = kv_idx < mem_kv_len
        kv_without_mem = kv_idx - mem_kv_len
        compress_kv_idx = (kv_without_mem * compress_block_size) + (compress_block_size - 1)
        causal_mask = q_idx > compress_kv_idx
        return causal_mask | is_mem_kv

    block_mask = create_block_mask(compress_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=kv_seq_len + mem_kv_len,
                                   _compile=True)
    return block_mask


def create_fine_mask(seq_len, fine_block_size):
    """
    Returns a function that creates a fine attention mask given selected block indices.
    The mask enforces causal ordering and allows tokens within the same fine block (or those explicitly selected)
    to attend.
    """

    def inner(selected_block_indices: Tensor, num_grouped_queries=1):
        device = selected_block_indices.device
        batch, kv_heads = selected_block_indices.shape[:2]

        one_hot_selected_block_indices = torch.zeros(
            (*selected_block_indices.shape[:-1], seq_len // fine_block_size),
            device=device, dtype=torch.bool
        )
        one_hot_selected_block_indices.scatter_(-1, selected_block_indices, True)

        def fine_mask(b_idx, h_idx, q_idx, kv_idx):
            compressed_q_idx = q_idx // fine_block_size
            compressed_kv_idx = kv_idx // fine_block_size
            # Determine if this block is selected for attention
            is_selected = one_hot_selected_block_indices[b_idx, h_idx, q_idx, compressed_kv_idx]
            causal_mask = q_idx >= kv_idx
            block_diagonal = compressed_q_idx == compressed_kv_idx
            return causal_mask & (block_diagonal | is_selected)

        block_mask = create_block_mask(
            fine_mask, B=batch, H=kv_heads * num_grouped_queries,
            Q_LEN=seq_len, KV_LEN=seq_len, _compile=True
        )
        return block_mask

    return inner


# Basic helper functions

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def round_down_mult(n, mult):
    return n // mult * mult


def round_up_mult(n, mult):
    return ceil(n / mult) * mult


def divisible_by(num, den):
    return (num % den) == 0


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def pack_one_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out):
        return unpack(out, ps, pattern)[0]

    return packed, inverse


def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def interpolate_1d(x, length, mode='bilinear'):
    x, inverse_pack = pack_one_with_inverse(x, '* n')
    x = rearrange(x, 'b n -> b 1 n 1')
    x = F.interpolate(x, (length, 1), mode=mode)
    x = rearrange(x, 'b 1 n 1 -> b n')
    return inverse_pack(x)


def straight_through(t, target):
    return t + (target - t).detach()


# -------------------------------------------------------------
# Main attend function (scaled dot-product attention)
# -------------------------------------------------------------

def attend(q, k, v, mask=None, return_sim=False, scale=None):
    """
    SDPA
    Computes scaled dot-product attention with an optional mask.

    q, k, v: Input tensors with shape adjustments done prior to calling this function.
    mask: Boolean mask indicating valid attention positions.
    scale: Scaling factor (default: inverse square root of head dimension).

    Returns:
      attn_out: The weighted sum output.
      (optionally) sim: The similarity scores before softmax.
    """
    scale = default(scale, q.shape[-1] ** -0.5)

    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads

    # Reshape queries to group query heads if necessary (for GQA)
    q = rearrange(q, 'b (h qh) ... -> b h qh ...', qh=num_grouped_queries)

    # Compute similarity (dot-product)
    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale
    mask_value = max_neg_value(sim)
    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value)

    attn = sim.softmax(dim=-1)
    attn_out = einsum(attn, v, 'b h qh i j, b h j d -> b h qh i d')

    # Merge grouped query heads back into one dimension
    attn_out = rearrange(attn_out, 'b h qh ... -> b (h qh) ...')

    if not return_sim:
        return attn_out
    sim = rearrange(sim, 'b h qh ... -> b (h qh) ...')
    return attn_out, sim


# -------------------------------------------------------------
# SparseAttention Module with detailed docstrings and KV cache handling
# -------------------------------------------------------------

class SparseAttention(Module):
    """
    SparseAttention implements a native sparse attention mechanism that splits
    the attention process into three branches:
      1. Compression Branch (coarse global summary of KV cache)
      2. Selection Branch (fine-grained selection from KV cache)
      3. Sliding Window Branch (local context)

    Additionally, this module supports explicit KV cache management.

    Parameters:
      dim: Input feature dimension. Same as d_model
      dim_head: Dimension per attention head. Same as dh or d_head in other code
      heads: Total number of query heads.
      sliding_window_size: Window size for local attention.
      compress_block_size: Block size for compressing older tokens.
      selection_block_size: Block size for fine selection of tokens.
      num_selected_blocks: Number of blocks to select for fine attention.
      kv_heads: Number of key/value heads (for GQA); defaults to heads if not provided.
      num_compressed_mem_kv: Number of extra compressed memory tokens to add.
      norm: Whether to apply RMSNorm to the input.
      use_diff_topk: Whether to use a different top-k strategy.
      use_triton_kernel: Whether to use a custom Triton kernel.
      interpolated_importance_score: If True, interpolate importance scores when block sizes differ.
      query_heads_share_selected_kv: For GQA, whether query heads share the same selected KV blocks.
      compress_mlp: Optional MLP for compressing tokens; if None, a default is created.
      compress_mlp_expand_factor: Expansion factor for the compress MLP.
      strategy_combine_mlp: Optional MLP for combining branch outputs; if None, a default is created.
    """

    def __init__(
            self,
            dim,
            dim_head,
            heads,
            sliding_window_size,
            compress_block_size,
            selection_block_size,
            num_selected_blocks,
            kv_heads: int | None = None,
            num_compressed_mem_kv=1,
            norm=True,
            use_diff_topk=False,
            use_triton_kernel=False,
            interpolated_importance_score=False,
            query_heads_share_selected_kv=True,  # For GQA: share selected KV blocks among grouped queries.
            compress_mlp: Module | None = None,
            compress_mlp_expand_factor=1.,
            strategy_combine_mlp: Module | None = None
    ):
        super().__init__()
        # -------------------------------------------------------------
        # Determine if using GQA (Grouped-Query Attention) or MHA (Multi-Head Attention)
        # Here, if kv_heads is provided and is less than heads, then we are in GQA mode.
        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads)
        self.heads = heads
        self.kv_heads = kv_heads

        # Separate GQA and MHA: if kv_heads < heads, we are in GQA mode, else we are using MHA.
        if kv_heads < heads:
            self.mode = 'GQA'
            self.num_grouped_queries = heads // kv_heads
        else:
            self.mode = 'MHA'
            self.num_grouped_queries = 1

        # Scaling for attention scores
        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()
        # MLA REQUIREMENTS

        self.q_proj_dim = dim // 2
        self.kv_proj_dim = 2 * dim // 3

        self.qk_NoPE_dim = dim_head // 2
        self.qk_RoPE_dim = dim_head // 2

        # Q projection, LoRA based.
        self.W_dq = nn.Parameter(0.01*torch.randn((dim, self.q_proj_dim)))
        self.W_uq = nn.Parameter(0.01*torch.randn((self.q_proj_dim, dim)))
        self.q_layernorm = torch.nn.RMSNorm(self.q_proj_dim)

        # KV projection, intermediate latent LoRA like.
        self.W_dkv = nn.Parameter(0.01*torch.randn(dim, self.kv_proj_dim))
        self.W_ukv = nn.Parameter(0.01*torch.randn((self.kv_proj_dim, 2*dim)))
        self.kv_layernorm = torch.nn.RMSNorm(self.kv_proj_dim)

        # Rotary positional embeddings for relative position handling
        self.rotary_emb = RotaryEmbedding(dim_head)

        # QKV projection: project input to queries, keys, and values.
        qkv_latent_split = (dim_inner, self.kv_proj_dim)
        self.to_qkv_latent = nn.Linear(dim, sum(qkv_latent_split), bias=False) # 그냥 행렬간 내적.
        self.q_kv_split = qkv_latent_split # q, (k, v)
        self.joint_kv_split = (self.qk_NoPE_dim, self.kv_proj_dim / 2) # K, V로 쪼개기

        # -------------------------------------------------------------
        # Sliding Window Branch: handles local context
        self.sliding_window = LocalAttention(
            dim=dim_head,
            window_size=sliding_window_size,
            causal=True,
            exact_windowsize=True,
            autopad=True,
            use_rotary_pos_emb=False
        )
        self.sliding_window_size = sliding_window_size

        # -------------------------------------------------------------
        # Compression Branch: compresses older parts of KV cache
        self.compress_block_size = compress_block_size
        # Rearrange tokens into blocks: (b, h, seq_len, d) -> (b, h, num_blocks, compress_block_size, d)
        self.split_compress_window = Rearrange('b h (w n) d -> b h w n d', n=compress_block_size)

        # Learnable memory tokens to provide fixed, compressed context for very old information.
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head))

        # Intra-block positional embeddings
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))

        # Default compression MLP if not provided: flattens each block and reduces it to a vector.
        if not exists(compress_mlp):
            compress_dim = compress_block_size * dim_head
            compress_mlp_dim_hidden = int(compress_mlp_expand_factor * compress_dim)
            compress_mlp = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(compress_dim, compress_mlp_dim_hidden),
                nn.ReLU(),
                nn.Linear(compress_mlp_dim_hidden, dim_head),
            )
        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        # -------------------------------------------------------------
        # Selection Branch: fine selection from the full KV cache
        self.use_diff_topk = use_diff_topk
        self.interpolated_importance_score = interpolated_importance_score
        self.query_heads_share_selected_kv = query_heads_share_selected_kv
        self.selection_block_size = selection_block_size
        assert num_selected_blocks > 0, "`num_selected_blocks` should be greater than 0."
        self.num_selected_blocks = num_selected_blocks
        self.use_triton_kernel = use_triton_kernel

        # -------------------------------------------------------------
        # Strategy Combine: learn to weight the outputs from the three branches.
        if not exists(strategy_combine_mlp):
            strategy_combine_mlp = nn.Linear(dim, 3 * heads)
            # Initialize biases so that the sliding window branch is initially favored.
            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(tensor([-2., -2., 2.] * heads))
        self.to_strategy_combine = nn.Sequential( # help weighted sum for three different branches
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h=heads)
        )

        # -------------------------------------------------------------
        # Head splitting/merging functions
        self.split_heads = Rearrange('b n (h d) -> b h n d', d=dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # Final linear layer to combine heads back into output dimension.
        self.combine_heads = nn.Linear(dim_inner, dim, bias=False)

    def forward(
            self,
            inp,
            disable_triton_kernel=False,
            sliding_window_flex_mask=None,
            fine_selection_flex_mask=None,
            kv_cache: dict | None = None  # Optional KV cache: expected keys "k" and "v"
    ):
        """
        Forward pass for SparseAttention.

        Parameters:
          inp: Input tensor of shape (batch, seq_len, dim)
          disable_triton_kernel: Option to disable Triton kernel (for selection branch)
          sliding_window_flex_mask: Precomputed mask for sliding window branch (if any)
          fine_selection_flex_mask: Precomputed mask for fine selection branch (if any)
          kv_cache: Optional dict containing cached keys and values.
                  If provided, should have keys "k" and "v" with shapes matching the output of to_qkv.
                  This allows explicit KV cache management to reduce memory footprint.

        Returns:
          out: Output tensor of shape (batch, seq_len, dim)
          (Also updates kv_cache if provided)
        """
        batch, seq_len, device = inp.shape[0], inp.shape[1], inp.device # BSD 지정

        # Optionally incorporate external KV cache.
        # If kv_cache is provided, concatenate the cached keys/values with the current ones.
        # This explicitly shows how the KV cache is being handled.
        cache_k, cache_v = None, None
        if kv_cache is not None:
            cache_k = kv_cache.get("k", None)
            cache_v = kv_cache.get("v", None)

        # Adjust sequence lengths based on compress and selection block sizes.
        # 이 블록 단위로 압축을 진행합니다.
        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # Normalize input.
        inp = self.norm(inp)

        # QKV projection.
        # w/ MLA here.
        qkv = self.to_qkv_latent(inp)  # shape: (b, seq_len, dim_inner + 2*dim_kv_inner // 3. Requires divisble embedding size of 3)
        q, kv = qkv.split(self.q_kv_split, dim=-1) # joint kv split

        # --- MLA MLA MLA ---
        # we should compress the keys and value into latent representation.
        q, k, v = map(self.split_heads,
                      (q, kv))
        # If a KV cache exists, prepend cached keys/values to the current ones.
        if cache_k is not None and cache_v is not None:
            # Assume cache_k, cache_v have shapes compatible with (b, heads, cache_seq_len, d)
            k = cat([cache_k, k], dim=2)
            v = cat([cache_v, v], dim=2)
            seq_len = k.shape[2]  # update sequence length to include cached tokens

        # --- Compression Branch ---
        # This branch compresses the (older) part of the KV cache into fewer tokens.
        # Only use tokens up to compress_divisible_seq_len for block splitting.
        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r=num_compress_blocks)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r=num_compress_blocks)
        # Apply intra-block positions to the keys/values before splitting.
        k_blocks = self.split_compress_window(k[..., :compress_divisible_seq_len, :] + k_pos)
        v_blocks = self.split_compress_window(v[..., :compress_divisible_seq_len, :] + v_pos)
        # Compress each block using the compression MLP.
        ck = self.k_compress(k_blocks)  # Equation (7) equivalent: compressed keys.
        cv = self.v_compress(v_blocks)  # Compressed values.

        # Prepare extra memory tokens (learnable, fixed) and concatenate to the compressed tokens.
        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b=batch)
        num_mem_compress_kv = mem_ck.shape[-2]
        ck = cat((mem_ck, ck), dim=-2)
        cv = cat((mem_cv, cv), dim=-2)

        # Create a causal mask for compressed branch using token indices.
        cq_seq = arange(seq_len, device=device)
        ck_seq = ((arange(num_compress_blocks, device=device) + 1) * self.compress_block_size) - 1
        ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value=-1)
        cmask = einx.less('j, i -> i j', ck_seq, cq_seq)
        # Compute attention over the compressed keys/values.
        compressed_attn_out, csim = attend(q, ck, cv, mask=cmask, return_sim=True)

        # --- Rotary Positional Embedding for Fine and Sliding Branches ---
        rotated_q, rotated_k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # --- Selection (Fine) Branch ---
        # This branch uses the compressed attention scores to compute importance scores
        # and then selects the top-N blocks from the full KV cache.
        importance_scores = csim[..., num_mem_compress_kv:]
        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0

        # Separate code paths for GQA and MHA:
        if self.mode == 'GQA':
            # For GQA, average the importance scores across grouped query heads.
            importance_scores = reduce(importance_scores, 'b (h grouped) ... -> b h ...', 'mean',
                                       grouped=self.num_grouped_queries)
            fine_num_grouped_queries = self.num_grouped_queries
        else:  # MHA: each head handles its own KV selection.
            fine_num_grouped_queries = 1

        # Adjust importance scores if compress_block_size != selection_block_size.
        if has_selected_kv_for_fine_attn:
            if self.compress_block_size != self.selection_block_size:
                compress_seq_len = num_compress_blocks * self.compress_block_size
                if self.interpolated_importance_score:
                    importance_scores = interpolate_1d(importance_scores, compress_seq_len)
                else:
                    importance_scores = repeat(importance_scores, '... j -> ... (j block_size)',
                                               block_size=self.compress_block_size)
                padding = fine_divisible_seq_len - compress_seq_len
                importance_scores = F.pad(importance_scores, (0, padding))
                # Create a block causal mask to zero-out irrelevant positions.
                block_causal_mask = torch.ones((num_fine_blocks,) * 2, device=device, dtype=torch.bool).tril(-1)
                block_causal_mask = repeat(block_causal_mask, 'i j -> (i n1) (j n2)', n1=self.selection_block_size,
                                           n2=self.selection_block_size)
                block_causal_mask = block_causal_mask[:importance_scores.shape[-2]]
                importance_scores = importance_scores.masked_fill(~block_causal_mask, max_neg_value(csim))
                importance_scores = reduce(importance_scores, '... (j block_size) -> ... j', 'mean',
                                           block_size=self.selection_block_size)
            # Pad and softmax the importance scores.
            importance_scores = F.pad(importance_scores, (1, 0), value=-1e3)
            importance_scores = importance_scores.softmax(dim=-1)
            importance_scores = importance_scores[..., 1:]

        # Rotate queries and keys for fine attention.
        fq = rotated_q
        fk = rotated_k
        fv = v  # use original values for fine attention

        if has_selected_kv_for_fine_attn:
            # Get the top-N block indices for fine attention.
            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim=-1)

            # Optionally use a different top-k strategy (not detailed here)
            if self.use_diff_topk:
                gates = straight_through(selected_importance_values, 1.)
                gates = gates.cumprod(dim=-1)[..., -1]
                gates = repeat(gates, 'b h ... -> b (h qh) ...', qh=fine_num_grouped_queries)

            # Fine attention can be executed with either a Triton kernel, flex attention,
            # or a custom implementation below.
            if self.use_triton_kernel and not disable_triton_kernel:
                from native_sparse_attention_pytorch.triton_native_sparse_attention import native_sparse_attend
                fmask = selected_importance_values > 1e-10
                fine_attn_out = native_sparse_attend(
                    fq, fk, fv,
                    self.selection_block_size,
                    selected_block_indices,
                    fmask
                )
            elif fine_selection_flex_mask is not None:
                fine_block_mask = fine_selection_flex_mask(selected_block_indices,
                                                           num_grouped_queries=fine_num_grouped_queries)
                fine_attn_out = flex_attention(fq, fk, fv, block_mask=fine_block_mask, enable_gqa=True)
            else:
                # If sequence length is less than expected, pad appropriately.
                fmask = selected_importance_values > 1e-10
                if seq_len < fine_divisible_seq_len:
                    remainder = fine_divisible_seq_len - seq_len
                    fk = pad_at_dim(fk, (0, remainder), value=0., dim=-2)
                    fv = pad_at_dim(fv, (0, remainder), value=0., dim=-2)
                    fq = pad_at_dim(fq, (0, remainder), value=0., dim=-2)
                    fmask = pad_at_dim(fmask, (0, remainder), value=False, dim=-2)
                    selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value=0, dim=-2)
                    if self.use_diff_topk:
                        gates = pad_at_dim(gates, (0, remainder), value=1.)
                # Create causal masks and rearrange fine blocks.
                fine_window_seq = arange(fine_divisible_seq_len, device=device) // self.selection_block_size
                fine_window_seq = repeat(fine_window_seq, 'n -> b h n 1', b=batch, h=selected_block_indices.shape[1])
                selected_block_indices = cat((selected_block_indices, fine_window_seq), dim=-1)
                fmask = repeat(fmask, 'b h i w -> b h i w j', j=self.selection_block_size)
                causal_mask = torch.ones((self.selection_block_size,) * 2, device=device, dtype=torch.bool).tril()
                causal_mask = repeat(causal_mask, 'i j -> b h (w i) 1 j', w=num_fine_blocks, b=batch, h=fmask.shape[1])
                fmask = cat((fmask, causal_mask), dim=-2)
                fmask = rearrange(fmask, 'b h i w j -> b h i (w j)')
                # Rearrange keys/values into fine blocks.
                fk = rearrange(fk, 'b h (w n) d -> b h w n d', w=num_fine_blocks)
                fv = rearrange(fv, 'b h (w n) d -> b h w n d', w=num_fine_blocks)
                if self.mode == 'GQA':
                    fk = repeat(fk, 'b h w j d -> b h i w j d', i=selected_block_indices.shape[2])
                    fv = repeat(fv, 'b h w j d -> b h i w j d', i=selected_block_indices.shape[2])
                else:  # MHA: each head handles its own selection
                    fk = repeat(fk, 'b h w j d -> b (h qh) i w j d', i=selected_block_indices.shape[2],
                                qh=self.num_grouped_queries)
                    fv = repeat(fv, 'b h w j d -> b (h qh) i w j d', i=selected_block_indices.shape[2],
                                qh=self.num_grouped_queries)
                selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j=fk.shape[-2],
                                                d=fk.shape[-1])
                fk = fk.gather(3, selected_block_indices)
                fv = fv.gather(3, selected_block_indices)
                fk, fv = (rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fk, fv))
                fmask = rearrange(fmask, 'b h ... -> b h 1 ...')
                fq = rearrange(fq, 'b (h qh) ... -> b h qh ...', qh=fine_num_grouped_queries)
                fsim = einsum(fq, fk, 'b h qh i d, b h i j d -> b h qh i j') * self.scale
                mask_value = max_neg_value(fsim)
                fsim = fsim.masked_fill(~fmask, mask_value)
                fattn = fsim.softmax(dim=-1)
                fine_attn_out = einsum(fattn, fv, 'b h qh i j, b h i j d -> b h qh i d')
                fine_attn_out = rearrange(fine_attn_out, 'b h qh ... -> b (h qh) ...')
                fine_attn_out = fine_attn_out[..., :seq_len, :]
            if self.use_diff_topk:
                gates = gates[..., :seq_len]
                fine_attn_out = einx.multiply('b h n, b h n d -> b h n d', gates, fine_attn_out)
        else:
            # If there is no selection (only one block), perform standard causal attention.
            seq_len = fk.shape[-2]
            fmask = causal_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
            fine_attn_out = attend(fq, fk, fv, mask=fmask)

        # --- Sliding Window Branch ---
        # Handles the recent tokens using a fixed local window.
        sq = rotated_q
        sk = rotated_k
        sv = v
        if exists(sliding_window_flex_mask):
            sliding_window_attn_out = flex_attention(sq, sk, sv, block_mask=sliding_window_flex_mask, enable_gqa=True)
        else:
            # For GQA mode, repeat keys/values appropriately.
            sk, sv = (
            repeat(t, 'b h ... -> b (h num_grouped_queries) ...', num_grouped_queries=self.num_grouped_queries) for t in
            (sk, sv))
            sliding_window_attn_out = self.sliding_window(sq, sk, sv)

        # --- Combine Branches ---
        # The three branches are combined via a learned gating mechanism.
        # The gating weights have shape (b, heads, seq_len, 3) corresponding to the three branches.
        strategy_weighted_combine = self.to_strategy_combine(inp)
        # Stack the branch outputs: order is [compressed, fine, sliding]
        combined = stack([compressed_attn_out, fine_attn_out, sliding_window_attn_out])
        out = einsum(strategy_weighted_combine, combined, 'b h n s, s b h n d -> b h n d')
        out = self.merge_heads(out)
        out = self.combine_heads(out)

        # --- KV Cache Update ---
        # If a KV cache is provided, update it with the new keys and values.
        if kv_cache is not None:
            # For autoregressive generation, you might only append the new keys/values.
            # Here we assume kv_cache is a dict with keys "k" and "v".
            updated_k = k if cache_k is None else cat([cache_k, k], dim=2)
            updated_v = v if cache_v is None else cat([cache_v, v], dim=2)
            kv_cache["k"] = updated_k
            kv_cache["v"] = updated_v

        return out

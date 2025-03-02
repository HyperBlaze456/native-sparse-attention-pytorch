from __future__ import annotations

from copy import deepcopy
from math import ceil

import torch
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from torch import nn, arange, stack, cat, tensor, Tensor
from torch.nn import Module

from local_attention import LocalAttention

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
# Helper functions
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
    attended to for compression.
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
# Scaled Dot-Product Attention Function
# -------------------------------------------------------------

def attend(q, k, v, mask=None, return_sim=False, scale=None):
    """
    Computes scaled dot-product attention with an optional mask.
    """
    scale = default(scale, q.shape[-1] ** -0.5)
    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads
    q = rearrange(q, 'b (h qh) ... -> b h qh ...', qh=num_grouped_queries)
    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale
    mask_value = max_neg_value(sim)
    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value)
    attn = sim.softmax(dim=-1)
    attn_out = einsum(attn, v, 'b h qh i j, b h j d -> b h qh i d')
    attn_out = rearrange(attn_out, 'b h qh ... -> b (h qh) ...')
    if not return_sim:
        return attn_out
    sim = rearrange(sim, 'b h qh ... -> b (h qh) ...')
    return attn_out, sim


# -------------------------------------------------------------
# MLA_SparseAttention Module with Joint Latent KV Cache
# -------------------------------------------------------------

class MLA_SparseAttention(Module):
    """
    MLA_SparseAttention integrates Multi-Latent Attention (MLA) into the native sparse attention (NSA)
    mechanism. In this design, the KV cache reduction is achieved by jointly compressing the KV inputs:
      - The input is projected into a lower-dimensional latent space for both keys and values.
      - This latent KV representation is then up-projected with a single matrix (W_ukv) and split into keys and values.
      - No separate cache keys are maintained; the input KV cache (if provided) is assumed to be the latent,
        jointly compressed KV cache.

    The remainder of the NSA processing (compression, selection, sliding window) then uses the computed Q, K, and V.

    Parameters are similar to the previous NSA module, with added MLA parameters.
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
            query_heads_share_selected_kv=True,
            compress_mlp: Module | None = None,
            compress_mlp_expand_factor=1.,
            strategy_combine_mlp: Module | None = None,
    ):
        super().__init__()
        # Determine attention mode (GQA or MHA)
        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads)
        self.heads = heads
        self.kv_heads = kv_heads
        if kv_heads < heads:
            self.mode = 'GQA'
            self.num_grouped_queries = heads // kv_heads
        else:
            self.mode = 'MHA'
            self.num_grouped_queries = 1

        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # -------------------------------------------------------------
        # MLA Setup: Joint latent projection for Q and KV.
        self.q_proj_dim = dim // 2  # latent dimension for queries
        self.kv_proj_dim = (2 * dim) // 3  # latent dimension for joint KV. Half becomes K, half becomes V.

        # For decoupling positional embedding
        self.qk_NoPE_dim = dim_head // 2 # Unused, currently
        self.qk_RoPE_dim = dim_head - self.qk_NoPE_dim

        # Q projections.
        self.W_dq = nn.Parameter(0.01 * torch.randn((dim, self.q_proj_dim)))
        self.W_uq = nn.Parameter(0.01 * torch.randn((self.q_proj_dim, dim)))
        self.q_layernorm = torch.nn.RMSNorm(self.q_proj_dim)

        # KV projections (jointly).
        self.W_dkv = nn.Parameter(0.01 * torch.randn((dim, self.kv_proj_dim)))
        self.W_ukv = nn.Parameter(0.01 * torch.randn((self.kv_proj_dim, 2 * dim)))
        self.kv_layernorm = torch.nn.RMSNorm(self.kv_proj_dim)

        # RoPE
        self.rotary_emb = RotaryEmbedding(dim_head)

        # -------------------------------------------------------------
        # Latent QKV projection.
        # We project the input to a concatenation of (latent Q, latent KV).
        qkv_latent_split = (dim_inner, self.kv_proj_dim)
        self.to_qkv_latent = nn.Linear(dim, sum(qkv_latent_split), bias=False)
        self.q_kv_split = qkv_latent_split

        # -------------------------------------------------------------
        # NSA Branches: Sliding Window, Compression, and Selection.
        self.sliding_window = LocalAttention(
            dim=dim_head,
            window_size=sliding_window_size,
            causal=True,
            exact_windowsize=True,
            autopad=True,
            use_rotary_pos_emb=False
        )
        self.sliding_window_size = sliding_window_size

        self.compress_block_size = compress_block_size
        self.split_compress_window = Rearrange('b h (w n) d -> b h w n d', n=compress_block_size)
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head))
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))
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

        self.use_diff_topk = use_diff_topk
        self.interpolated_importance_score = interpolated_importance_score
        self.query_heads_share_selected_kv = query_heads_share_selected_kv
        self.selection_block_size = selection_block_size
        assert num_selected_blocks > 0, "`num_selected_blocks` should be > 0."
        self.num_selected_blocks = num_selected_blocks
        self.use_triton_kernel = use_triton_kernel

        if not exists(strategy_combine_mlp):
            strategy_combine_mlp = nn.Linear(dim, 3 * heads)
            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(tensor([-2., -2., 2.] * heads))
        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h=heads)
        )

        self.split_heads = Rearrange('b n (h d) -> b h n d', d=dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = nn.Linear(dim_inner, dim, bias=False)

    def forward(
            self,
            inp,
            disable_triton_kernel=False,
            sliding_window_flex_mask=None,
            fine_selection_flex_mask=None,
            kv_cache: Tensor | None = None  # Input latent KV cache (jointly compressed), or None.
    ):
        """
        Forward pass for MLA_SparseAttention with joint latent KV cache reduction.

        Steps:
          1. Normalize the input.
          2. Compute latent Q and latent KV jointly using to_qkv_latent.
             - For queries: down-project (W_dq), layernorm, then up-project (W_uq).
             - For KV: If a latent KV cache is provided, concatenate it with the new latent projection;
                   otherwise, compute the new latent representation.
          3. Up-project the latent KV to obtain full-dimensional KV, then split into K and V.
          4. Process Q, K, and V into multiple heads.
          5. Proceed with the NSA branches (Compression, Selection, Sliding Window).
          6. Combine branch outputs using a learned gating mechanism.
          7. Update and return the latent KV cache.

        Returns:
          out: Output tensor of shape (batch, seq_len, dim)
          updated_kv_cache: Updated latent KV cache tensor. Reuse it!
        """
        B, S, device = inp.shape[0], inp.shape[1], inp.device

        # Adjust sequence lengths for compression and selection.
        compress_divisible_seq_len = round_down_mult(S, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size
        fine_divisible_seq_len = round_up_mult(S, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        inp = self.norm(inp)

        # --- MLA: Joint Latent Projections ---
        # Project input to latent Q and latent KV.
        qkv = self.to_qkv_latent(inp)  # Shape: (B, S, dim_inner + kv_proj_dim)
        q_latent, latent_kv_new = qkv.split(self.q_kv_split, dim=-1)
        # Process Q: down-project then up-project.
        compressed_q = q_latent @ self.W_dq  # (B, S, q_proj_dim)
        compressed_q = self.q_layernorm(compressed_q)
        Q_full = compressed_q @ self.W_uq  # (B, S, dim_inner)
        # If a latent KV cache is provided, treat it as already jointly compressed.
        if kv_cache is None:
            latent_kv = latent_kv_new
        else:
            latent_kv = torch.cat([kv_cache, latent_kv_new], dim=1)  # Concatenate along sequence dim.

        # Up-project latent KV to full KV space.
        KV_full = latent_kv @ self.W_ukv  # (B, S_total, 2*dim), where S_total = latent KV cache length.
        # Split into keys and values.
        K_full, V_full = torch.split(KV_full, self.norm.normalized_shape[0], dim=-1)

        # --- Process into Multiple Heads ---
        Q_heads = self.split_heads(Q_full)  # (B, heads, S, d)
        k_heads = self.split_heads(K_full)  # (B, heads, S_total, d)
        v_heads = self.split_heads(V_full)  # (B, heads, S_total, d)

        # --- NSA Branches ---
        # Compression Branch: compress older parts of the latent KV.
        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r=num_compress_blocks)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r=num_compress_blocks)
        k_blocks = self.split_compress_window(k_heads[..., :compress_divisible_seq_len, :] + k_pos)
        v_blocks = self.split_compress_window(v_heads[..., :compress_divisible_seq_len, :] + v_pos)
        ck = self.k_compress(k_blocks)
        cv = self.v_compress(v_blocks)
        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b=B)
        num_mem_compress_kv = mem_ck.shape[-2]
        ck = cat((mem_ck, ck), dim=-2)
        cv = cat((mem_cv, cv), dim=-2)
        cq_seq = arange(S, device=device)
        ck_seq = ((arange(num_compress_blocks, device=device) + 1) * self.compress_block_size) - 1
        ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value=-1)
        cmask = einx.less('j, i -> i j', ck_seq, cq_seq)
        compressed_attn_out, csim = attend(Q_heads, ck, cv, mask=cmask, return_sim=True)

        # Rotary embeddings for the fine and sliding branches.
        rotated_q, rotated_k = self.rotary_emb.rotate_queries_with_cached_keys(Q_heads, k_heads)

        # Selection Branch: use compressed branch similarity scores to select important blocks.
        importance_scores = csim[..., num_mem_compress_kv:]
        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0
        if self.mode == 'GQA':
            importance_scores = reduce(importance_scores, 'b (h grouped) ... -> b h ...', 'mean',
                                       grouped=self.num_grouped_queries)
            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1
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
                block_causal_mask = torch.ones((num_fine_blocks,) * 2, device=device, dtype=torch.bool).tril(-1)
                block_causal_mask = repeat(block_causal_mask, 'i j -> (i n1) (j n2)', n1=self.selection_block_size,
                                           n2=self.selection_block_size)
                block_causal_mask = block_causal_mask[:importance_scores.shape[-2]]
                importance_scores = importance_scores.masked_fill(~block_causal_mask, max_neg_value(csim))
                importance_scores = reduce(importance_scores, '... (j block_size) -> ... j', 'mean',
                                           block_size=self.selection_block_size)
            importance_scores = F.pad(importance_scores, (1, 0), value=-1e3)
            importance_scores = importance_scores.softmax(dim=-1)
            importance_scores = importance_scores[..., 1:]
        fq = rotated_q
        fk = rotated_k
        fv = v_heads  # Fine branch uses full-resolution values.
        if has_selected_kv_for_fine_attn:
            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim=-1)
            if self.use_diff_topk:
                gates = straight_through(selected_importance_values, 1.)
                gates = gates.cumprod(dim=-1)[..., -1]
                gates = repeat(gates, 'b h ... -> b (h qh) ...', qh=fine_num_grouped_queries)
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
                fmask = selected_importance_values > 1e-10
                if S < fine_divisible_seq_len:
                    remainder = fine_divisible_seq_len - S
                    fk = pad_at_dim(fk, (0, remainder), value=0., dim=-2)
                    fv = pad_at_dim(fv, (0, remainder), value=0., dim=-2)
                    fq = pad_at_dim(fq, (0, remainder), value=0., dim=-2)
                    fmask = pad_at_dim(fmask, (0, remainder), value=False, dim=-2)
                    selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value=0, dim=-2)
                    if self.use_diff_topk:
                        gates = pad_at_dim(gates, (0, remainder), value=1.)
                fine_window_seq = arange(fine_divisible_seq_len, device=device) // self.selection_block_size
                fine_window_seq = repeat(fine_window_seq, 'n -> b h n 1', b=B, h=selected_block_indices.shape[1])
                selected_block_indices = cat((selected_block_indices, fine_window_seq), dim=-1)
                fmask = repeat(fmask, 'b h i w -> b h i w j', j=self.selection_block_size)
                causal_mask = torch.ones((self.selection_block_size,) * 2, device=device, dtype=torch.bool).tril()
                causal_mask = repeat(causal_mask, 'i j -> b h (w i) 1 j', w=num_fine_blocks, b=B, h=fmask.shape[1])
                fmask = cat((fmask, causal_mask), dim=-2)
                fmask = rearrange(fmask, 'b h i w j -> b h i (w j)')
                fk = rearrange(fk, 'b h (w n) d -> b h w n d', w=num_fine_blocks)
                fv = rearrange(fv, 'b h (w n) d -> b h w n d', w=num_fine_blocks)
                if self.mode == 'GQA':
                    fk = repeat(fk, 'b h w j d -> b h i w j d', i=selected_block_indices.shape[2])
                    fv = repeat(fv, 'b h w j d -> b h i w j d', i=selected_block_indices.shape[2])
                else:
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
                fine_attn_out = fine_attn_out[..., :S, :]
            if self.use_diff_topk:
                gates = gates[..., :S]
                fine_attn_out = einx.multiply('b h n, b h n d -> b h n d', gates, fine_attn_out)
        else:
            S_full = fk.size(2)
            fmask = torch.ones((S, S_full), device=device, dtype=torch.bool).tril()
            fine_attn_out = attend(fq, fk, fv, mask=fmask)

        # --- Sliding Window Branch ---
        sq = rotated_q
        sk = rotated_k
        sv = v_heads
        if exists(sliding_window_flex_mask):
            sliding_window_attn_out = flex_attention(sq, sk, sv, block_mask=sliding_window_flex_mask, enable_gqa=True)
        else:
            sk, sv = (
            repeat(t, 'b h ... -> b (h num_grouped_queries) ...', num_grouped_queries=self.num_grouped_queries) for t in
            (sk, sv))
            sliding_window_attn_out = self.sliding_window(sq, sk, sv)

        # --- Combine Branches ---
        strategy_weighted_combine = self.to_strategy_combine(inp)
        combined = stack([compressed_attn_out, fine_attn_out, sliding_window_attn_out])
        out = einsum(strategy_weighted_combine, combined, 'b h n s, s b h n d -> b h n d')
        out = self.merge_heads(out)
        out = self.combine_heads(out)

        # --- KV Cache Update ---
        # In this design, the latent KV cache is stored jointly.
        # We update the latent KV cache to be the concatenation of the old latent and the new latent projection.
        if kv_cache is None:
            updated_kv_cache = latent_kv
        else:
            updated_kv_cache = torch.cat([kv_cache, latent_kv_new], dim=1)

        return out, updated_kv_cache

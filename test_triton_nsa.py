from math import ceil
import torch
from native_sparse_attention_pytorch.triton_native_sparse_attention import native_sparse_attend, round_up_multiple, pad_to_multiple

import einx
from einops import rearrange, einsum, repeat

assert torch.cuda.is_available()

def exists(v):
    return v is not None

def abs_diff(x, y):
    return (x - y).abs().amax()

def divisible_by(num, den):
    return (num % den) == 0

def regular_attend(
    q, k, v,
    indices,
    mask,
    block_size,
    return_lse = False
):
    q_heads, seq_len, kv_heads, device = q.shape[1], q.shape[-2], k.shape[1], q.device
    assert divisible_by(q_heads, kv_heads)

    q, k, v = tuple(pad_to_multiple(t, block_size, dim = -2) for t in (q, k, v))

    g = q_heads // kv_heads # `g` stands for `g`roups of query heads per kv head

    w = ceil(seq_len / block_size)

    q, k, v = tuple(rearrange(t, 'b h (w n) d -> b h w n d', n = block_size) for t in (q, k, v))

    scale = q.shape[-1] ** -0.5
    q = q * scale

    q = rearrange(q, 'b (h g) ... -> b h g ...', g = g)

    # block causal diagonal

    sim = einsum(q, k, 'b h g w i d, b h w j d -> b h g w i j')
    causal_mask = torch.ones((block_size, block_size), device = device, dtype = torch.bool).triu(1)
    sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    # rest of the indices

    num_sel_kv_blocks = indices.shape[-1]
    has_sel_kv_blocks = num_sel_kv_blocks > 0

    if has_sel_kv_blocks:
        indices, mask = tuple(pad_to_multiple(t, block_size, dim = -2) for t in (indices, mask))

        bk, bv = k, v
        sel_bk = einx.get_at('b h [w] n d, b h i sel -> b h i (sel n) d', bk, indices)
        sel_bv = einx.get_at('b h [w] n d, b h i sel -> b h i (sel n) d', bv, indices)

        q = rearrange(q, 'b h g w n d -> b h g (w n) d')
        bsim = einsum(q, sel_bk, 'b h g i d, b h i j d -> b h g i j')

        bsim = rearrange(bsim, 'b h g (w i) (sel j) -> b h g w i sel j', sel = num_sel_kv_blocks, i = fine_block_size)

        mask = rearrange(mask, 'b h (w i) sel -> b h 1 w i sel', i = fine_block_size)
        bsim = torch.where(mask[..., None], bsim, -torch.finfo(bsim.dtype).max)

        sim = rearrange(sim, 'b h g w i j -> b h g w i 1 j')

        sim = torch.cat((sim, bsim), dim = -2)
        sim = rearrange(sim, 'b h g w i causal_and_sel j -> b h g w i (causal_and_sel j)')

        sel_bv = rearrange(sel_bv, 'b h (w i) j d -> b h w i j d', i = fine_block_size)

        v = repeat(v, 'b h w j d -> b h w i j d', i = fine_block_size)
        v = torch.cat((v, sel_bv), dim = -2)
        v = rearrange(v, 'b h w i j d -> b h w i j d')

    # attend

    attn = sim.softmax(dim = -1)

    if has_sel_kv_blocks:
        out = einsum(attn, v, 'b h g w i j, b h w i j d -> b h g w i d')
    else:
        out = einsum(attn, v, 'b h g w i j, b h w j d -> b h g w i d')

    out = rearrange(out, 'b h g w n d -> b (h g) (w n) d')

    out = out[..., :seq_len, :]

    if not return_lse:
        return out

    lse = sim.logsumexp(dim = -1)
    lse = rearrange(lse, 'b g h w n -> b (g h) (w n)')
    lse = lse[..., :seq_len]

    return out, lse

# mock inputs

batch = 2
seq_len = 511
q_heads = 4
kv_heads = 2
fine_block_size = 16
num_sel = 6

q = torch.randn(batch, q_heads, seq_len, 64).cuda()
k = torch.randn(batch, kv_heads, seq_len, 64).cuda()
v = torch.randn(batch, kv_heads, seq_len, 64).cuda()

indices = torch.randint(0, 2, (batch, kv_heads, seq_len, num_sel)).cuda()
mask = torch.randint(0, 2, (batch, kv_heads, seq_len, num_sel)).bool().cuda()

# both regular and nsa pathways `r` and `n`

rq, rk, rv = tuple(t.clone().requires_grad_() for t in (q, k, v))
nq, nk, nv = tuple(t.clone().requires_grad_() for t in (q, k, v))

# regular forwards and backwards

out, rlse = regular_attend(rq, rk, rv, indices, mask, block_size = fine_block_size, return_lse = True)
out.sum().backward()

# triton nsa forwards and backwards

nsa_out, nlse = native_sparse_attend(nq, nk, nv, fine_block_size, indices, mask, return_lse = True)
nsa_out.sum().backward()

# asserts

assert torch.allclose(out, nsa_out, atol = 1e-2)
assert torch.allclose(rlse, nlse, atol = 1e-2)

assert torch.allclose(nv.grad, rv.grad, atol = 1e-2)
assert torch.allclose(nk.grad, rk.grad, atol = 1e-2)
assert torch.allclose(nq.grad, rq.grad, atol = 1e-2)

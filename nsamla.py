import torch
from native_sparse_attention_pytorch.native_sparse_attention_mla import MLA_SparseAttention

def test_autoregressive():
    # Model configuration.
    dim = 72           # d_model
    dim_head = 12      # Dimension per head (must be even for RoPE)
    heads = 6          # Total number of heads
    sliding_window_size = 4
    compress_block_size = 4
    selection_block_size = 4
    num_selected_blocks = 2

    # Instantiate the MLA_SparseAttention module.
    model = MLA_SparseAttention(
        dim=dim,
        dim_head=dim_head,
        heads=heads,
        sliding_window_size=sliding_window_size,
        compress_block_size=compress_block_size,
        selection_block_size=selection_block_size,
        num_selected_blocks=num_selected_blocks,
        use_triton_kernel=False
    )
    model.eval()

    # Autoregressive simulation: each step produces a new token.
    kv_cache = None
    num_steps = 10
    for step in range(num_steps):
        # New token as input: shape (batch, 1, dim)
        inp = torch.randn(2, 1, dim)
        out, kv_cache = model(inp, kv_cache=kv_cache)
        print(f"Step {step}: Output shape = {out.shape}, KV cache shape = {kv_cache.shape}")

if __name__ == "__main__":
    test_autoregressive()

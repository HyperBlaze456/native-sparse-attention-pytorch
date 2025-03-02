import torch
from native_sparse_attention_pytorch.native_sparse_attention_mla import MLA_SparseAttention  # Assuming the module is in mla_sparse_attention.py


def mla_nsa():
    # For testing, we choose small dimensions.
    dim = 66  # Input dimension (d_model)
    dim_head = 11  # Dimension per head
    heads = 6  # Total number of heads
    sliding_window_size = 8
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

    # Set the model to evaluation mode.
    model.eval()

    # No initial KV cache (latent joint cache).
    kv_cache = None

    # Simulate autoregressive decoding: for several steps, new tokens are added.
    # Here, each step feeds an input tensor of shape (batch, seq_len, dim).
    # The KV cache (latent) is updated with each call.
    for step in range(5):
        # For testing, use a random input with sequence length 10.
        inp = torch.randn(2, 10, dim)  # Batch=2, Seq_len=10
        out, kv_cache = model(inp, kv_cache=kv_cache)
        print(f"Step {step}: Output shape = {out.shape}")
        # The latent KV cache should have shape (batch, total_latent_seq_len, kv_proj_dim)
        # where kv_proj_dim = (2 * dim) // 3 (integer division)
        print(f"Step {step}: Latent KV cache shape = {kv_cache.shape}")


if __name__ == "__main__":
    mla_nsa()

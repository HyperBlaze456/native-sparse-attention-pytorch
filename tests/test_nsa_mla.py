import pytest
import torch
from native_sparse_attention_pytorch.native_sparse_attention_mla import SparseAttention


@pytest.mark.parametrize('use_diff_topk', (False, True))
@pytest.mark.parametrize('causal', (False, True))
@pytest.mark.parametrize('seq_len', (1, 4, 31, 32, 120))
@pytest.mark.parametrize('selection_block_size', (8, 4, 2))
@pytest.mark.parametrize('num_selected_blocks', (1, 2))
@pytest.mark.parametrize('interpolated_importance_score', (False, True))
def test_sparse_attn(
        use_diff_topk,
        causal,
        seq_len,
        selection_block_size,
        num_selected_blocks,
        interpolated_importance_score
):
    attn = SparseAttention(
        dim=512,
        dim_head=64,
        heads=8,
        causal=causal,
        sliding_window_size=2,
        compress_block_size=4,
        selection_block_size=selection_block_size,
        num_selected_blocks=num_selected_blocks,
        use_diff_topk=use_diff_topk,
        interpolated_importance_score=interpolated_importance_score
    )

    tokens = torch.randn(2, seq_len, 512)
    attended = attn(tokens)
    assert tokens.shape == attended.shape

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / np.sum(ex, axis=-1, keepdims=True)


def test_causal_mask_applies_in_multihead_attention():
    from src.transformer.attention_layers.multi_head_attention import MultiHeadAttention

    rng = np.random.default_rng(0)
    batch_size = 2
    seq_len = 6
    d_model = 16
    num_heads = 4

    queries = rng.standard_normal((batch_size, seq_len, d_model)).astype(np.float64)
    keys = rng.standard_normal((batch_size, seq_len, d_model)).astype(np.float64)
    values = rng.standard_normal((batch_size, seq_len, d_model)).astype(np.float64)

    mha = MultiHeadAttention(d_model, num_heads)

    # 2D causal mask (seq_len, seq_len) should broadcast to (batch, seq_len, seq_len)
    mask = mha.create_causal_mask(seq_len)

    _, attn_weights_per_head = mha.forward(queries, keys, values, mask)

    # Verify masked (future) positions receive ~0 attention
    for head_weights in attn_weights_per_head:
        # head_weights: (batch, seq_len, seq_len)
        upper_triangle = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        assert np.allclose(
            head_weights[:, upper_triangle], 0.0, atol=1e-12
        ), "Causal mask failed to zero out future positions"


def test_padding_mask_applies_in_multihead_attention_with_broadcast():
    from src.transformer.attention_layers.multi_head_attention import MultiHeadAttention
    from src.transformer.masking.masks import create_padding_mask

    rng = np.random.default_rng(1)
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4

    # Build sequences with padding token 0 at the end of each row
    # First example pads last 2, second pads last 3
    sequences = np.array([
        [5, 7, 9, 0, 0],
        [1, 2, 0, 0, 0],
    ], dtype=int)

    queries = rng.standard_normal((batch_size, seq_len, d_model)).astype(np.float64)
    keys = rng.standard_normal((batch_size, seq_len, d_model)).astype(np.float64)
    values = rng.standard_normal((batch_size, seq_len, d_model)).astype(np.float64)

    mha = MultiHeadAttention(d_model, num_heads)

    # Encoder-style padding mask (batch, 1, 1, seq_len) -> squeeze to (batch, 1, seq_len)
    pad4d = create_padding_mask(sequences)  # (batch, 1, 1, seq_len)
    pad = np.squeeze(np.squeeze(pad4d, axis=1), axis=1)  # (batch, seq_len)
    mask = pad[:, None, :]  # (batch, 1, seq_len), broadcast to (batch, q_len, k_len)

    _, attn_weights_per_head = mha.forward(queries, keys, values, mask)

    # For each batch, last positions (pads) should receive ~0 attention across all queries
    for b in range(batch_size):
        pad_positions = np.where(sequences[b] == 0)[0]
        if pad_positions.size == 0:
            continue
        for head_weights in attn_weights_per_head:
            # Sum attention over all queries for masked key positions should be 0
            masked_column_sum = head_weights[b, :, pad_positions].sum()
            assert np.allclose(masked_column_sum, 0.0, atol=1e-12), (
                f"Padding mask failed for batch {b}"
            )


def test_cross_attention_padding_mask_broadcasting():
    from src.transformer.attention_layers.multi_head_attention import MultiHeadAttention

    rng = np.random.default_rng(2)
    batch_size = 2
    q_len = 3
    k_len = 5
    d_model = 16
    num_heads = 4

    queries = rng.standard_normal((batch_size, q_len, d_model)).astype(np.float64)
    keys = rng.standard_normal((batch_size, k_len, d_model)).astype(np.float64)
    values = rng.standard_normal((batch_size, k_len, d_model)).astype(np.float64)

    # Build a (batch, 1, k_len) padding mask to be broadcast across q_len
    # Mask last 2 keys for b=0 and last 3 keys for b=1
    pad = np.zeros((batch_size, 1, k_len))
    pad[0, 0, -2:] = -1e9
    pad[1, 0, -3:] = -1e9

    mha = MultiHeadAttention(d_model, num_heads)
    _, attn_weights_per_head = mha.forward(queries, keys, values, pad)

    for b in range(batch_size):
        masked_cols = np.where(pad[b, 0] < 0)[0]
        for head_weights in attn_weights_per_head:
            assert np.allclose(head_weights[b, :, masked_cols], 0.0, atol=1e-12)


def test_complete_transformer_forward_shapes():
    from src.transformer.complete.transformer import CompleteTransformer

    rng = np.random.default_rng(3)
    batch_size = 2
    src_len = 5
    tgt_len = 4
    vocab_size = 100

    src = rng.integers(0, vocab_size, size=(batch_size, src_len))
    tgt = rng.integers(0, vocab_size, size=(batch_size, tgt_len))

    model = CompleteTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
    )

    logits = model.forward(src, tgt)
    assert logits.shape == (batch_size, tgt_len, vocab_size)



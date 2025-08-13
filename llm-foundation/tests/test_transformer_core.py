import numpy as np
from src.transformer.core.transformer import Transformer


def test_transformer_forward_shapes_and_masking():
    vocab_size = 20
    d_model = 16
    num_heads = 4
    num_layers = 2
    d_ff = 32

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=32,
    )

    batch_size = 3
    src_len = 5
    tgt_len = 6

    # Build toy src/tgt with padding (PAD=0)
    rng = np.random.default_rng(0)
    src = rng.integers(low=1, high=vocab_size, size=(batch_size, src_len))
    tgt = rng.integers(low=1, high=vocab_size, size=(batch_size, tgt_len))
    # Add PADs to last positions for some samples
    src[0, -2:] = 0
    tgt[1, -3:] = 0

    src_pad_mask = (src == 0)

    logits = model.forward(src, tgt, src_key_padding_mask=src_pad_mask)

    assert logits.shape == (batch_size, tgt_len, vocab_size)
    assert np.isfinite(logits).all()


def test_transformer_generate_basic():
    vocab_size = 15
    model = Transformer(vocab_size=vocab_size, d_model=8, num_heads=2, num_layers=1, d_ff=16, max_seq_length=16)

    src = np.array([[1, 2, 3, 0]])  # PAD=0

    out = model.generate(src, max_length=5, start_token=1, end_token=2)
    assert isinstance(out, list)
    assert len(out) >= 1


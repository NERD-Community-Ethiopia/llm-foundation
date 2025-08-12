"""
Complete Transformer Example
"""
import numpy as np
from ..complete.transformer import CompleteTransformer
from ..masking.masks import create_padding_mask, create_causal_mask

def basic_transformer_example():
    """Demonstrate basic complete transformer"""
    print("🚀 Basic Complete Transformer Example")
    
    # Create sample data
    batch_size = 2
    src_seq_length = 5
    tgt_seq_length = 4
    vocab_size = 1000
    d_model = 64
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    
    # Create random sequences
    src_sequences = np.random.randint(0, vocab_size, (batch_size, src_seq_length))
    tgt_sequences = np.random.randint(0, vocab_size, (batch_size, tgt_seq_length))
    
    # Create complete transformer
    transformer = CompleteTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_model * 4
    )
    
    print(f"Source sequences shape: {src_sequences.shape}")
    print(f"Target sequences shape: {tgt_sequences.shape}")
    print(f"Model config: d_model={d_model}, heads={num_heads}, encoder_layers={num_encoder_layers}")
    
    # Forward pass
    print("\nRunning forward pass...")
    output = transformer.forward(src_sequences, tgt_sequences)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (batch_size, tgt_seq_length, vocab_size)")
    
    return transformer, output

def masking_example():
    """Demonstrate masking functionality"""
    print("\n🎭 Masking Example")
    
    # Create sample sequences with padding
    batch_size = 3
    max_seq_length = 6
    vocab_size = 100
    
    # Create sequences with different lengths (padded with 0)
    sequences = np.array([
        [1, 2, 3, 4, 0, 0],  # Length 4, padded with 2 zeros
        [1, 2, 3, 0, 0, 0],  # Length 3, padded with 3 zeros
        [1, 2, 3, 4, 5, 6]   # Length 6, no padding
    ])
    
    print(f"Input sequences:\n{sequences}")
    
    # Create padding mask
    padding_mask = create_padding_mask(sequences)
    print(f"\nPadding mask shape: {padding_mask.shape}")
    print(f"Padding mask (first sequence):\n{padding_mask[0, 0, 0]}")
    
    # Create causal mask
    causal_mask = create_causal_mask(max_seq_length)
    print(f"\nCausal mask shape: {causal_mask.shape}")
    print(f"Causal mask:\n{causal_mask[0, 0]}")
    
    return sequences, padding_mask, causal_mask

if __name__ == "__main__":
    print("🚀 Running Complete Transformer Examples\n")
    
    # Basic transformer example
    transformer, output = basic_transformer_example()
    
    # Masking example
    sequences, padding_mask, causal_mask = masking_example()
    
    print("\n✅ All complete transformer examples completed!")
    print("🎯 The complete transformer is working with:")
    print("   - Multiple encoder and decoder blocks")
    print("   - Proper masking for padding and causality")
    print("   - Full encoder-decoder architecture")

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

def text_transformer_example():
    """Test transformer with realistic text-like sequences"""
    print("\n📝 Text Transformer Example")
    
    # Create a simple vocabulary mapping
    vocab = {
        'hello': 1, 'world': 2, 'how': 3, 'are': 4, 'you': 5,
        'i': 6, 'am': 7, 'fine': 8, 'thank': 9, 'goodbye': 10,
        'the': 11, 'cat': 12, 'sat': 13, 'on': 14, 'mat': 15,
        '<PAD>': 0, '<START>': 16, '<END>': 17
    }
    
    # Create sample text sequences
    src_texts = [
        ['hello', 'world', 'how', 'are', 'you'],
        ['the', 'cat', 'sat', 'on', 'mat']
    ]
    
    tgt_texts = [
        ['i', 'am', 'fine', 'thank', '<PAD>'],  # Pad to length 5
        ['the', 'cat', 'sat', 'on', 'mat']      # Already length 5
    ]
    
    # Convert to token IDs
    src_sequences = np.array([[vocab[word] for word in seq] for seq in src_texts])
    tgt_sequences = np.array([[vocab[word] for word in seq] for seq in tgt_texts])
    
    print("Source texts:")
    for i, text in enumerate(src_texts):
        print(f"  {i+1}: {' '.join(text)}")
    
    print("\nTarget texts:")
    for i, text in enumerate(tgt_texts):
        print(f"  {i+1}: {' '.join(text)}")
    
    # Create transformer
    vocab_size = len(vocab)
    d_model = 128
    num_heads = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    
    transformer = CompleteTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_model * 4
    )
    
    print(f"\nModel config: vocab_size={vocab_size}, d_model={d_model}, heads={num_heads}")
    
    # Forward pass
    print("\nRunning forward pass...")
    output = transformer.forward(src_sequences, tgt_sequences)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch_size={len(src_texts)}, tgt_seq_length={len(tgt_texts[0])}, vocab_size={vocab_size})")
    
    # Show some output probabilities
    print("\nOutput logits (first sequence, first token):")
    first_logits = output[0, 0, :]
    top_5_indices = np.argsort(first_logits)[-5:][::-1]
    for idx in top_5_indices:
        word = [k for k, v in vocab.items() if v == idx][0]
        print(f"  {word}: {first_logits[idx]:.3f}")
    
    return transformer, output, vocab

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
    
    # Text transformer example
    text_transformer, text_output, vocab = text_transformer_example()
    
    # Masking example
    sequences, padding_mask, causal_mask = masking_example()
    
    print("\n✅ All complete transformer examples completed!")
    print("🎯 The complete transformer is working with:")
    print("   - Multiple encoder and decoder blocks")
    print("   - Proper masking for padding and causality")
    print("   - Full encoder-decoder architecture")
    print("   - Real text-like sequences")
    print(f"   - Vocabulary size: {len(vocab)} tokens")
    
    # Test Ethiopian language example
    print(f"\n🇪🇹 Testing Ethiopian Language Support...")
    try:
        ethiopian_vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3,
            'ሰላም': 4, 'እንደምን': 5, 'ደህና': 6, 'አማሪኛ': 7,
            'akkam': 8, 'nagaa': 9, 'gaarii': 10, 'galata': 11
        }
        
        ethiopian_transformer = CompleteTransformer(
            vocab_size=len(ethiopian_vocab),
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=256
        )
        
        # Test Amharic to Oromiffa
        amharic_seq = np.array([[4, 5, 0, 0, 0]])  # ሰላም እንደምን
        oromiffa_seq = np.array([[8, 9, 0, 0, 0]])  # akkam nagaa
        
        ethiopian_output = ethiopian_transformer.forward(amharic_seq, oromiffa_seq)
        print(f"✅ Ethiopian language test successful!")
        print(f"   Amharic: ሰላም እንደምን")
        print(f"   Oromiffa: akkam nagaa")
        print(f"   Output shape: {ethiopian_output.shape}")
        
    except Exception as e:
        print(f"❌ Ethiopian language test error: {e}")

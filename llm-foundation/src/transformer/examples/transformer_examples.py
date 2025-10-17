"""
Transformer Examples and Demonstrations
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from ..core.transformer import Transformer

def simple_translation_example():
    """Simple translation example"""
    print("�� Simple Translation Example")
    print("=" * 50)
    
    # Create vocabulary
    vocab = {
        '<PAD>': 0, '<START>': 1, '<END>': 2,
        'hello': 3, 'world': 4, 'bonjour': 5, 'monde': 6,
        'the': 7, 'cat': 8, 'sat': 9, 'on': 10, 'mat': 11,
        'le': 12, 'chat': 13, 's\'est': 14, 'assis': 15, 'sur': 16, 'tapis': 17
    }
    
    vocab_size = len(vocab)
    
    # Create Transformer
    transformer = Transformer(vocab_size=vocab_size, d_model=128, num_heads=4, 
                            num_layers=2, d_ff=512)
    
    # Training data
    src_sentences = [
        [vocab['hello'], vocab['world']],
        [vocab['the'], vocab['cat'], vocab['sat'], vocab['on'], vocab['the'], vocab['mat']]
    ]
    
    tgt_sentences = [
        [vocab['<START>'], vocab['bonjour'], vocab['monde'], vocab['<END>']],
        [vocab['<START>'], vocab['le'], vocab['chat'], vocab['s\'est'], vocab['assis'], 
         vocab['sur'], vocab['le'], vocab['tapis'], vocab['<END>']]
    ]
    
    # Pad sequences
    max_src_len = max(len(s) for s in src_sentences)
    max_tgt_len = max(len(s) for s in tgt_sentences)
    
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_sentences, tgt_sentences):
        src_padded.append(src + [vocab['<PAD>']] * (max_src_len - len(src)))
        tgt_padded.append(tgt + [vocab['<PAD>']] * (max_tgt_len - len(tgt)))
    
    src_batch = np.array(src_padded)
    tgt_batch = np.array(tgt_padded)
    
    print("Source batch shape:", src_batch.shape)
    print("Target batch shape:", tgt_batch.shape)
    
    # Forward pass
    logits = transformer.forward(src_batch, tgt_batch)
    print("Logits shape:", logits.shape)
    
    # Generate translation
    print("\nGenerating translation...")
    generated = transformer.generate(src_batch[0:1], max_length=10, 
                                   start_token=vocab['<START>'], end_token=vocab['<END>'])
    
    # Convert back to words
    id_to_word = {v: k for k, v in vocab.items()}
    generated_words = [id_to_word[token_id] for token_id in generated]
    
    print("Generated:", ' '.join(generated_words))
    
    return transformer, vocab

def visualize_attention_weights():
    """Visualize attention weights"""
    print("\n🎨 Visualizing Attention Weights")
    print("=" * 50)
    
    # Create a simple example
    transformer, vocab = simple_translation_example()
    
    # Get attention weights (simplified)
    src = np.array([[vocab['hello'], vocab['world']]])
    tgt = np.array([[vocab['<START>'], vocab['bonjour']]])
    
    # This would require modifying the attention layers to return weights
    print("Attention visualization would show how each word attends to others")
    
    return transformer

def demonstrate_positional_encoding():
    """Demonstrate positional encoding"""
    print("\n📍 Demonstrating Positional Encoding")
    print("=" * 50)
    
    from ..positional_encoding.positional_encoding import PositionalEncoding
    
    # Create positional encoding
    pe = PositionalEncoding(d_model=64, max_seq_length=20)
    
    # Visualize
    pe.visualize(seq_length=20)
    
    # Show values
    print("Positional encoding shape:", pe.pe.shape)
    print("First few values:")
    print(pe.pe[:5, :5])

if __name__ == "__main__":
    # Run examples
    transformer, vocab = simple_translation_example()
    visualize_attention_weights()
    demonstrate_positional_encoding()
    
    print("\n✅ Transformer implementation complete!")

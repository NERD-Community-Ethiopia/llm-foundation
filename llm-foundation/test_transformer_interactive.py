#!/usr/bin/env python3
"""
Interactive Transformer Testing Script
Test your Complete Transformer with custom text!
"""

import numpy as np
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transformer.complete.transformer import CompleteTransformer
from transformer.masking.masks import create_padding_mask, create_causal_mask

def create_simple_vocab(texts):
    """Create vocabulary from input texts"""
    vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2}
    token_id = 3
    
    for text in texts:
        words = text.lower().split()
        for word in words:
            if word not in vocab:
                vocab[word] = token_id
                token_id += 1
    
    return vocab

def text_to_sequence(text, vocab, max_length=10):
    """Convert text to token sequence"""
    words = text.lower().split()
    sequence = [vocab.get(word, vocab['<PAD>']) for word in words[:max_length]]
    
    # Pad to max_length
    while len(sequence) < max_length:
        sequence.append(vocab['<PAD>'])
    
    return sequence

def sequence_to_text(sequence, vocab):
    """Convert token sequence back to text"""
    id_to_word = {v: k for k, v in vocab.items()}
    words = []
    for token_id in sequence:
        if token_id == vocab['<PAD>']:
            break
        words.append(id_to_word.get(token_id, f'<UNK:{token_id}>'))
    return ' '.join(words)

def test_transformer_interactive():
    """Interactive testing of the Complete Transformer"""
    print("🚀 Interactive Transformer Testing")
    print("=" * 50)
    
    # Get input texts from user
    print("\nEnter source text (e.g., 'hello world how are you'):")
    src_text = input("Source: ").strip()
    
    print("\nEnter target text (e.g., 'i am fine thank you'):")
    tgt_text = input("Target: ").strip()
    
    if not src_text or not tgt_text:
        print("❌ Please provide both source and target text!")
        return
    
    # Create vocabulary
    all_texts = [src_text, tgt_text]
    vocab = create_simple_vocab(all_texts)
    
    print(f"\n📚 Vocabulary created with {len(vocab)} tokens:")
    for word, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"  {token_id:2d}: {word}")
    
    # Convert texts to sequences
    src_sequence = text_to_sequence(src_text, vocab)
    tgt_sequence = text_to_sequence(tgt_text, vocab)
    
    print(f"\n🔄 Converting text to sequences:")
    print(f"Source: '{src_text}' → {src_sequence}")
    print(f"Target: '{tgt_text}' → {tgt_sequence}")
    
    # Create transformer
    vocab_size = len(vocab)
    d_model = 64  # Smaller for faster testing
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    
    print(f"\n🏗️  Creating Complete Transformer...")
    print(f"   Config: vocab_size={vocab_size}, d_model={d_model}, heads={num_heads}")
    
    transformer = CompleteTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_model * 4
    )
    
    # Prepare batch (add batch dimension)
    src_batch = np.array([src_sequence])
    tgt_batch = np.array([tgt_sequence])
    
    print(f"\n📊 Input shapes:")
    print(f"   Source batch: {src_batch.shape}")
    print(f"   Target batch: {tgt_batch.shape}")
    
    # Run forward pass
    print(f"\n🚀 Running forward pass...")
    try:
        output = transformer.forward(src_batch, tgt_batch)
        print(f"✅ Forward pass successful!")
        print(f"📈 Output shape: {output.shape}")
        
        # Show output logits for first target token
        print(f"\n🎯 Output logits for first target token '{tgt_text.split()[0]}':")
        first_logits = output[0, 0, :]
        top_5_indices = np.argsort(first_logits)[-5:][::-1]
        
        for i, idx in enumerate(top_5_indices):
            word = [k for k, v in vocab.items() if v == idx][0]
            confidence = first_logits[idx]
            print(f"   {i+1}. {word}: {confidence:.3f}")
        
        # Test generation
        print(f"\n🎲 Testing generation...")
        generated = transformer.generate(src_batch, max_length=len(tgt_text.split()) + 2)
        print(f"   Generated sequence: {generated[0]}")
        print(f"   Generated text: '{sequence_to_text(generated[0], vocab)}'")
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("🎯 Complete Transformer Interactive Testing")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Test with custom text")
        print("2. Run quick demo")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            test_transformer_interactive()
        elif choice == '2':
            print("\n🚀 Running quick demo...")
            # Quick demo with predefined text
            src_text = "hello world how are you"
            tgt_text = "i am fine thank you"
            print(f"Source: '{src_text}'")
            print(f"Target: '{tgt_text}'")
            
            # Create vocabulary and test
            all_texts = [src_text, tgt_text]
            vocab = create_simple_vocab(all_texts)
            src_sequence = text_to_sequence(src_text, vocab)
            tgt_sequence = text_to_sequence(tgt_text, vocab)
            
            transformer = CompleteTransformer(
                vocab_size=len(vocab),
                d_model=64,
                num_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                d_ff=256
            )
            
            src_batch = np.array([src_sequence])
            tgt_batch = np.array([tgt_sequence])
            
            try:
                output = transformer.forward(src_batch, tgt_batch)
                print(f"✅ Demo successful! Output shape: {output.shape}")
            except Exception as e:
                print(f"❌ Demo error: {e}")
                
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()

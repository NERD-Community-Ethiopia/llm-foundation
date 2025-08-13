#!/usr/bin/env python3
"""
Amharic to Oromiffa Translation Example
Using the Complete Transformer for Ethiopian language translation
"""

import numpy as np
from ..complete.transformer import CompleteTransformer
from ..masking.masks import create_padding_mask, create_causal_mask

def create_ethiopian_vocab():
    """Create vocabulary with Amharic and Oromiffa words"""
    vocab = {
        # Special tokens
        '<PAD>': 0,      # Padding token
        '<START>': 1,     # Start of sequence
        '<END>': 2,       # End of sequence
        '<UNK>': 3,       # Unknown token
        
        # Amharic words (source language)
        'ሰላም': 4,        # selam - hello/peace
        'እንደምን': 5,      # endemen - how are you
        'እሺ': 6,          # eshi - okay
        'ደህና': 7,        # dehna - good
        'አማሪኛ': 8,      # amarinya - Amharic
        'ኦሮምኛ': 9,      # oromigna - Oromiffa
        'ይህ': 10,         # yih - this
        'ነው': 11,         # new - is
        'ቋንቋ': 12,       # kwanqwa - language
        'እኔ': 13,         # ene - I/me
        'የምለው': 14,      # yemilew - I speak
        'አዎ': 15,         # awo - yes
        'አይ': 16,         # ay - no
        'እናመሰግናለን': 17,  # enamesegnalen - thank you
        'እባክህ': 18,      # ebakih - please
        'ደስ': 19,         # des - nice
        'እንደት': 20,      # endet - how
        'አለ': 21,         # ale - there is
        'የት': 22,         # yet - where
        'እዚህ': 23,        # ezih - here
        
        # Oromiffa words (target language)
        'akkam': 24,      # hello/how are you
        'nagaa': 25,      # peace/hello
        'tolesa': 26,     # welcome
        'galata': 27,     # thank you
        'waan': 28,       # yes
        'laki': 29,       # no
        'maaloo': 30,     # please
        'gaarii': 31,     # good
        'bari': 32,       # morning
        'galgala': 33,    # evening
        'halkan': 34,     # here
        'as': 35,         # is
        'kan': 36,         # this
        'qooqooda': 37,   # language
        'afaan': 38,      # language/mouth
        'Oromoo': 39,     # Oromo
        'Amaariffaa': 40, # Amharic
        'barreessuu': 41, # to write
        'dubbisuu': 42,   # to speak
        'beekuu': 43,     # to know
        'jiru': 44,       # there is
        'eessa': 45,      # where
        'maal': 46,       # what
        'isaan': 47,      # they
        'keenya': 48,     # our
        'kanneen': 49,    # these
    }
    return vocab

def create_translation_pairs():
    """Create Amharic-Oromiffa translation pairs"""
    translation_pairs = [
        # Greetings
        ("ሰላም", "akkam"),
        ("እንደምን አለህ", "akkam jirta"),
        ("ደህና ነው", "gaarii dha"),
        ("እናመሰግናለን", "galata"),
        
        # Language identification
        ("ይህ አማሪኛ ነው", "kan Amaariffaa dha"),
        ("እኔ ኦሮምኛ የምለው", "ani afaan Oromoo dubbisuu beeka"),
        ("ኦሮምኛ ቋንቋ ነው", "afaan Oromoo qooqooda dha"),
        
        # Basic phrases
        ("እባክህ", "maaloo"),
        ("ደስ አለ", "gaarii dha"),
        ("እንደት አለ", "akkam jirta"),
        ("አዎ", "waan"),
        ("አይ", "laki"),
        
        # Questions
        ("የት ነው", "eessa dha"),
        ("እዚህ ነው", "halkan dha"),
        ("ምን ነው", "maal dha"),
        
        # Longer sentences
        ("እኔ ኦሮምኛ እያወራሁ ነው", "ani afaan Oromoo dubbisuu jiru"),
        ("ይህ ቋንቋ ደስ አለ", "kan qooqooda gaarii dha"),
        ("እናመሰግናለን እባክህ", "galata maaloo"),
    ]
    return translation_pairs

def text_to_sequence(text, vocab, max_length=10):
    """Convert text to token sequence with padding"""
    # Split by spaces (assuming space-separated tokens)
    tokens = text.split()
    sequence = []
    
    for token in tokens:
        if token in vocab:
            sequence.append(vocab[token])
        else:
            sequence.append(vocab['<UNK>'])
    
    # Pad to max_length
    while len(sequence) < max_length:
        sequence.append(vocab['<PAD>'])
    
    return sequence[:max_length]

def sequence_to_text(sequence, vocab):
    """Convert token sequence back to text"""
    id_to_word = {v: k for k, v in vocab.items()}
    words = []
    
    for token_id in sequence:
        if token_id == vocab['<PAD>']:
            break
        word = id_to_word.get(token_id, f'<UNK:{token_id}>')
        words.append(word)
    
    return ' '.join(words)

def amharic_oromiffa_translation_example():
    """Demonstrate Amharic to Oromiffa translation"""
    print("🇪🇹 Amharic to Oromiffa Translation Example")
    print("=" * 60)
    
    # Create vocabulary
    vocab = create_ethiopian_vocab()
    print(f"📚 Vocabulary created with {len(vocab)} tokens")
    print("   Includes Amharic, Oromiffa, and special tokens")
    
    # Get translation pairs
    translation_pairs = create_translation_pairs()
    print(f"\n🔄 {len(translation_pairs)} translation pairs created")
    
    # Show some examples
    print("\n📝 Sample Translation Pairs:")
    for i, (amharic, oromiffa) in enumerate(translation_pairs[:5]):
        print(f"  {i+1}. {amharic} → {oromiffa}")
    
    # Create transformer
    vocab_size = len(vocab)
    d_model = 128
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    
    print(f"\n🏗️  Creating Complete Transformer...")
    print(f"   Config: vocab_size={vocab_size}, d_model={d_model}, heads={num_heads}")
    print(f"   Encoder layers: {num_encoder_layers}, Decoder layers: {num_decoder_layers}")
    
    transformer = CompleteTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_model * 4
    )
    
    # Test with a few translation pairs
    print(f"\n🚀 Testing Translation Pipeline...")
    
    for i, (amharic, oromiffa) in enumerate(translation_pairs[:3]):
        print(f"\n--- Translation {i+1} ---")
        print(f"Amharic: {amharic}")
        print(f"Oromiffa: {oromiffa}")
        
        # Convert to sequences
        src_sequence = text_to_sequence(amharic, vocab, max_length=8)
        tgt_sequence = text_to_sequence(oromiffa, vocab, max_length=8)
        
        print(f"Source tokens: {src_sequence}")
        print(f"Target tokens: {tgt_sequence}")
        
        # Prepare batch
        src_batch = np.array([src_sequence])
        tgt_batch = np.array([tgt_sequence])
        
        try:
            # Forward pass
            output = transformer.forward(src_batch, tgt_batch)
            print(f"✅ Translation successful!")
            print(f"Output shape: {output.shape}")
            
            # Show top predictions for first target token
            first_logits = output[0, 0, :]
            top_5_indices = np.argsort(first_logits)[-5:][::-1]
            
            print(f"Top 5 predictions for '{oromiffa.split()[0]}':")
            for j, idx in enumerate(top_5_indices):
                word = [k for k, v in vocab.items() if v == idx][0]
                confidence = first_logits[idx]
                print(f"  {j+1}. {word}: {confidence:.3f}")
                
        except Exception as e:
            print(f"❌ Translation error: {e}")
    
    return transformer, vocab, translation_pairs

def main():
    """Main function"""
    print("🇪🇹 Amharic to Oromiffa Translation with Complete Transformer")
    print("=" * 70)
    
    # Run translation example
    transformer, vocab, translation_pairs = amharic_oromiffa_translation_example()
    
    print(f"\n✅ Translation example completed!")
    print("🎯 Your transformer is ready for Ethiopian language translation!")

if __name__ == "__main__":
    main()

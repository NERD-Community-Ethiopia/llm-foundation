"""
Transformer Training Example
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from ..core.transformer import Transformer
from ..training.training_pipeline import TransformerTrainer

def create_training_data(vocab: Dict[str, int], num_samples: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create training data for translation task
    
    Args:
        vocab: Vocabulary dictionary
        num_samples: Number of training samples
        
    Returns:
        List of (src, tgt) pairs
    """
    # Simple English to French translation pairs
    translation_pairs = [
        ("hello world", "bonjour monde"),
        ("the cat sat", "le chat s'est assis"),
        ("on the mat", "sur le tapis"),
        ("i love you", "je t'aime"),
        ("good morning", "bonjour"),
        ("how are you", "comment allez-vous"),
        ("thank you", "merci"),
        ("goodbye", "au revoir"),
        ("please help", "s'il vous plaît aidez"),
        ("where is the", "où est le"),
    ]
    
    # Expand with variations
    expanded_pairs = []
    for _ in range(num_samples // len(translation_pairs) + 1):
        expanded_pairs.extend(translation_pairs)
    
    training_data = []
    
    for src_text, tgt_text in expanded_pairs[:num_samples]:
        # Tokenize source
        src_tokens = src_text.split()
        src_ids = [vocab.get(token, vocab['<UNK>']) for token in src_tokens]
        
        # Tokenize target (add start and end tokens)
        tgt_tokens = tgt_text.split()
        tgt_ids = [vocab['<START>']] + [vocab.get(token, vocab['<UNK>']) for token in tgt_tokens] + [vocab['<END>']]
        
        # Pad sequences
        max_src_len = max(len(s) for s, _ in expanded_pairs[:num_samples])
        max_tgt_len = max(len(t) for _, t in expanded_pairs[:num_samples]) + 2  # +2 for start/end tokens
        
        src_padded = src_ids + [vocab['<PAD>']] * (max_src_len - len(src_ids))
        tgt_padded = tgt_ids + [vocab['<PAD>']] * (max_tgt_len - len(tgt_ids))
        
        training_data.append((np.array(src_padded), np.array(tgt_padded)))
    
    return training_data

def train_transformer_example():
    """Complete training example"""
    print("🚀 TRANSFORMER TRAINING EXAMPLE")
    print("=" * 60)
    
    # Create vocabulary
    vocab = {
        '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3,
        'hello': 4, 'world': 5, 'bonjour': 6, 'monde': 7,
        'the': 8, 'cat': 9, 'sat': 10, 'le': 11, 'chat': 12, 's\'est': 13, 'assis': 14,
        'on': 15, 'mat': 16, 'sur': 17, 'tapis': 18,
        'i': 19, 'love': 20, 'you': 21, 'je': 22, 't\'aime': 23,
        'good': 24, 'morning': 25, 'how': 26, 'are': 27, 'comment': 28, 'allez-vous': 29,
        'thank': 30, 'merci': 31, 'goodbye': 32, 'au': 33, 'revoir': 34,
        'please': 35, 'help': 36, 's\'il': 37, 'vous': 38, 'plaît': 39, 'aidez': 40,
        'where': 41, 'is': 42, 'où': 43, 'est': 44
    }
    
    vocab_size = len(vocab)
    print(f"📚 Vocabulary size: {vocab_size}")
    
    # Create Transformer
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=64,  # Smaller for faster training
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_length=50  # Increased to handle longer sequences
    )
    
    print(f"🧠 Transformer created with {transformer.num_layers} layers")
    
    # Create training data
    print("\n📊 Creating training data...")
    train_data = create_training_data(vocab, num_samples=50)
    val_data = create_training_data(vocab, num_samples=10)
    
    print(f"✅ Training samples: {len(train_data)}")
    print(f"✅ Validation samples: {len(val_data)}")
    
    # Sample training data
    print("\n📝 Sample training data:")
    id_to_word = {v: k for k, v in vocab.items()}
    for i, (src, tgt) in enumerate(train_data[:3]):
        src_words = [id_to_word[id] for id in src if id != vocab['<PAD>']]
        tgt_words = [id_to_word[id] for id in tgt if id not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]]
        print(f"  {i+1}. {' '.join(src_words)} → {' '.join(tgt_words)}")
    
    # Create trainer
    trainer = TransformerTrainer(
        transformer=transformer,
        learning_rate=0.001,
        optimizer='adam',
        clip_norm=1.0
    )
    
    print(f"\n🎯 Trainer initialized with Adam optimizer")
    print(f"📈 Learning rate: {trainer.learning_rate}")
    
    # Train the model
    print("\n🔥 Starting training...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=5,  # Small number for demonstration
        batch_size=8,
        save_attention=True
    )
    
    # Plot training history
    print("\n📈 Plotting training history...")
    trainer.plot_training_history()
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_src = np.array([vocab['hello'], vocab['world'], vocab['<PAD>'], vocab['<PAD>']])
    test_src_batch = test_src[np.newaxis, :]
    
    generated = transformer.generate(
        test_src_batch,
        max_length=10,
        start_token=vocab['<START>'],
        end_token=vocab['<END>']
    )
    
    generated_words = [id_to_word[token_id] for token_id in generated]
    print(f"Input: hello world")
    print(f"Generated: {' '.join(generated_words)}")
    
    # Test another example
    test_src2 = np.array([vocab['the'], vocab['cat'], vocab['sat'], vocab['<PAD>']])
    test_src_batch2 = test_src2[np.newaxis, :]
    
    generated2 = transformer.generate(
        test_src_batch2,
        max_length=10,
        start_token=vocab['<START>'],
        end_token=vocab['<END>']
    )
    
    generated_words2 = [id_to_word[token_id] for token_id in generated2]
    print(f"Input: the cat sat")
    print(f"Generated: {' '.join(generated_words2)}")
    
    print("\n✅ Training example completed!")
    
    return transformer, trainer, vocab

def demonstrate_attention_learning():
    """Demonstrate how attention weights change during training"""
    print("\n🎭 ATTENTION LEARNING DEMONSTRATION")
    print("=" * 50)
    
    # Create a simple example
    vocab = {
        '<PAD>': 0, '<START>': 1, '<END>': 2,
        'hello': 3, 'world': 4, 'bonjour': 5, 'monde': 6
    }
    
    transformer = Transformer(
        vocab_size=len(vocab),
        d_model=32,
        num_heads=2,
        num_layers=1,
        d_ff=128
    )
    
    # Create simple training data
    train_data = [
        (np.array([vocab['hello'], vocab['world']]), 
         np.array([vocab['<START>'], vocab['bonjour'], vocab['monde'], vocab['<END>']]))
    ] * 20  # Repeat 20 times
    
    trainer = TransformerTrainer(transformer, learning_rate=0.01)
    
    print("Training for attention visualization...")
    history = trainer.train(train_data, epochs=3, batch_size=4)
    
    print("✅ Attention learning demonstration completed!")
    
    return transformer, trainer

if __name__ == "__main__":
    # Run complete training example
    transformer, trainer, vocab = train_transformer_example()
    
    # Run attention demonstration
    transformer2, trainer2 = demonstrate_attention_learning()
    
    print("\n🎉 All training examples completed successfully!") 
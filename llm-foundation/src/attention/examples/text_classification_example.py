"""
Text Classification with Attention
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class AttentionTextClassifier:
    """
    Simple text classifier using attention mechanisms
    """
    
    def __init__(self, vocab_size: int, num_classes: int, hidden_dim: int = 64):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.embedding_dim = 32
        self.embeddings = np.random.randn(vocab_size, self.embedding_dim) * 0.01
        
        # Attention weights
        self.W_q = np.random.randn(self.embedding_dim, hidden_dim) * 0.01
        self.W_k = np.random.randn(self.embedding_dim, hidden_dim) * 0.01
        self.W_v = np.random.randn(self.embedding_dim, hidden_dim) * 0.01
        
        # Classification weights
        self.W_classifier = np.random.randn(hidden_dim, num_classes) * 0.01
        self.b_classifier = np.zeros((num_classes,))
        
    def forward(self, text_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for text classification with attention
        
        Args:
            text_indices: List of word indices
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # Get embeddings
        embeddings = self.embeddings[text_indices]  # (seq_len, embedding_dim)
        
        # Apply attention
        Q = np.dot(embeddings, self.W_q)  # (seq_len, hidden_dim)
        K = np.dot(embeddings, self.W_k)  # (seq_len, hidden_dim)
        V = np.dot(embeddings, self.W_v)  # (seq_len, hidden_dim)
        
        # Compute attention scores
        scores = np.dot(Q, K.T) / np.sqrt(self.hidden_dim)  # (seq_len, seq_len)
        attention_weights = self.softmax(scores)  # (seq_len, seq_len)
        
        # Apply attention to values
        attended = np.dot(attention_weights, V)  # (seq_len, hidden_dim)
        
        # Global average pooling
        pooled = np.mean(attended, axis=0)  # (hidden_dim,)
        
        # Classification
        logits = np.dot(pooled, self.W_classifier) + self.b_classifier
        predictions = self.softmax(logits)
        
        return predictions, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def test_text_classification():
    """Test attention-based text classification"""
    print("📚 Testing Attention-Based Text Classification")
    print("=" * 60)
    
    # Sample data
    texts = [
        "I love this movie it was amazing",
        "This film was terrible and boring", 
        "Great acting and wonderful story",
        "Waste of time horrible film",
        "Excellent performance by the actors",
        "Boring plot and bad acting"
    ]
    
    labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    
    # Create vocabulary
    all_words = set()
    for text in texts:
        all_words.update(text.lower().split())
    
    vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: 2 (positive/negative)")
    
    # Create classifier
    classifier = AttentionTextClassifier(vocab_size=vocab_size, num_classes=2)
    
    # Test on each text
    for i, (text, label) in enumerate(zip(texts, labels)):
        print(f"\n🔢 Text {i+1}: '{text}'")
        print(f"   True label: {'Positive' if label == 1 else 'Negative'}")
        
        # Convert text to indices
        words = text.lower().split()
        indices = [vocab.get(word, 0) for word in words]
        
        # Get predictions and attention
        predictions, attention_weights = classifier.forward(indices)
        
        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions)
        
        print(f"   Predicted: {'Positive' if predicted_label == 1 else 'Negative'} (confidence: {confidence:.3f})")
        print(f"   Attention weights shape: {attention_weights.shape}")
        
        # Visualize attention
        plt.figure(figsize=(8, 6))
        im = plt.imshow(attention_weights, cmap='Blues', aspect='auto')
        plt.colorbar(im)
        
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.yticks(range(len(words)), words)
        plt.title(f'Attention Weights: "{text}"')
        plt.xlabel('Words')
        plt.ylabel('Words')
        
        plt.tight_layout()
        plt.savefig(f'attention_classification_text_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Visualization saved: attention_classification_text_{i+1}.png")

if __name__ == "__main__":
    test_text_classification() 
"""
Comprehensive Attention Demonstration
"""
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_attention_with_text():
    """Demonstrate attention with actual text examples"""
    
    print("🎭 ATTENTION WITH REAL TEXT EXAMPLES")
    print("=" * 50)
    
    # Example 1: Basic attention with translation
    print("\n📝 Example 1: Translation Attention")
    print("-" * 30)
    
    # English to French translation example
    english_words = ["I", "love", "attention", "mechanisms"]
    french_words = ["J'", "aime", "les", "mécanismes", "d'attention"]
    
    print(f"English: {english_words}")
    print(f"French: {french_words}")
    
    # Simulate attention weights (in real model, these would be learned)
    attention_matrix = np.array([
        [0.8, 0.1, 0.05, 0.05],  # "I" attends to "J'"
        [0.1, 0.7, 0.1, 0.1],   # "love" attends to "aime"
        [0.05, 0.1, 0.6, 0.25], # "attention" attends to "mécanismes"
        [0.05, 0.1, 0.25, 0.6]  # "mechanisms" attends to "d'attention"
    ])
    
    print("\nAttention Weights (English → French):")
    print("        J'   aime  méca  d'att")
    for i, word in enumerate(english_words):
        print(f"{word:8} {attention_matrix[i][0]:.2f}  {attention_matrix[i][1]:.2f}  {attention_matrix[i][2]:.2f}  {attention_matrix[i][3]:.2f}")
    
    # Example 2: Self-attention with sentence
    print("\n📝 Example 2: Self-Attention in Sentence")
    print("-" * 30)
    
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    
    # Simulate self-attention weights
    self_attention = np.array([
        [0.9, 0.05, 0.02, 0.01, 0.01, 0.01],  # "The" attends to itself
        [0.1, 0.6, 0.2, 0.05, 0.03, 0.02],   # "cat" attends to "sat"
        [0.05, 0.3, 0.4, 0.15, 0.05, 0.05],  # "sat" attends to "cat" and "on"
        [0.02, 0.1, 0.3, 0.4, 0.1, 0.08],    # "on" attends to "sat" and "the"
        [0.01, 0.05, 0.1, 0.2, 0.5, 0.14],   # "the" attends to "mat"
        [0.01, 0.02, 0.05, 0.1, 0.3, 0.52]   # "mat" attends to "the"
    ])
    
    print(f"Sentence: {' '.join(sentence)}")
    print("\nSelf-Attention Weights:")
    print("        The  cat   sat   on   the  mat")
    for i, word in enumerate(sentence):
        print(f"{word:8} {self_attention[i][0]:.2f}  {self_attention[i][1]:.2f}  {self_attention[i][2]:.2f}  {self_attention[i][3]:.2f}  {self_attention[i][4]:.2f}  {self_attention[i][5]:.2f}")
    
    # Example 3: Multi-head attention
    print("\n📝 Example 3: Multi-Head Attention")
    print("-" * 30)
    
    # Simulate different attention heads focusing on different aspects
    heads = {
        "Syntactic": "Subject-verb-object relationships",
        "Semantic": "Meaning and context",
        "Positional": "Word order and position",
        "Contextual": "Surrounding word context"
    }
    
    for head_name, description in heads.items():
        print(f"\n{head_name} Head: {description}")
        # Simulate different attention patterns for each head
        if head_name == "Syntactic":
            pattern = "Focuses on grammatical relationships"
        elif head_name == "Semantic":
            pattern = "Focuses on meaning connections"
        elif head_name == "Positional":
            pattern = "Focuses on word positions"
        else:
            pattern = "Focuses on context"
        print(f"  Pattern: {pattern}")
    
    return attention_matrix, self_attention, heads

def visualize_attention_patterns():
    """Create comprehensive attention visualizations"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Attention Mechanism Patterns', fontsize=16, weight='bold')
    
    # 1. Basic Attention
    ax1 = axes[0, 0]
    attention_basic = np.array([[0.8, 0.1, 0.05, 0.05],
                               [0.1, 0.7, 0.1, 0.1],
                               [0.05, 0.1, 0.6, 0.25]])
    im1 = ax1.imshow(attention_basic, cmap='Blues', aspect='auto')
    ax1.set_title('Basic Attention (Query → Keys)')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Self-Attention
    ax2 = axes[0, 1]
    attention_self = np.array([[0.9, 0.05, 0.02, 0.01, 0.01, 0.01],
                              [0.1, 0.6, 0.2, 0.05, 0.03, 0.02],
                              [0.05, 0.3, 0.4, 0.15, 0.05, 0.05],
                              [0.02, 0.1, 0.3, 0.4, 0.1, 0.08],
                              [0.01, 0.05, 0.1, 0.2, 0.5, 0.14],
                              [0.01, 0.02, 0.05, 0.1, 0.3, 0.52]])
    im2 = ax2.imshow(attention_self, cmap='Greens', aspect='auto')
    ax2.set_title('Self-Attention (Word → Word)')
    ax2.set_xlabel('Word Position')
    ax2.set_ylabel('Word Position')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Causal Attention
    ax3 = axes[1, 0]
    causal_mask = np.triu(np.ones((6, 6)), k=1) * -1e9
    causal_attention = np.array([[0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.1, 0.8, 0.0, 0.0, 0.0, 0.0],
                                [0.05, 0.2, 0.7, 0.0, 0.0, 0.0],
                                [0.02, 0.1, 0.2, 0.6, 0.0, 0.0],
                                [0.01, 0.05, 0.1, 0.2, 0.5, 0.0],
                                [0.01, 0.02, 0.05, 0.1, 0.3, 0.52]])
    im3 = ax3.imshow(causal_attention, cmap='Reds', aspect='auto')
    ax3.set_title('Causal Attention (No Future)')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Position')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Multi-Head Attention
    ax4 = axes[1, 1]
    multi_head = np.array([[0.8, 0.1, 0.05, 0.05],
                          [0.1, 0.7, 0.1, 0.1],
                          [0.05, 0.1, 0.6, 0.25],
                          [0.02, 0.1, 0.2, 0.68]])
    im4 = ax4.imshow(multi_head, cmap='Purples', aspect='auto')
    ax4.set_title('Multi-Head Attention (Concatenated)')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Position')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('attention_patterns_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Comprehensive attention visualizations created!")

if __name__ == "__main__":
    print("🎭 Running Comprehensive Attention Demonstration\n")
    
    # Run text examples
    attention_matrix, self_attention, heads = demonstrate_attention_with_text()
    
    # Create visualizations
    visualize_attention_patterns()
    
    print("\n✅ Attention demonstration completed!")
    print("📈 Check the generated visualizations:")
    print("   - attention_patterns_comprehensive.png") 
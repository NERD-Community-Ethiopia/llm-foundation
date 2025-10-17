"""
Attention Mechanism Examples
"""
import numpy as np
import matplotlib.pyplot as plt
from ..basic_attention.attention import BasicAttention
from ..self_attention.self_attention import SelfAttention
from ..multi_head_attention.multi_head_attention import MultiHeadAttention

def basic_attention_example():
    """Demonstrate basic attention mechanism"""
    print("🔍 Basic Attention Example")
    
    # Create sample data
    seq_len_q = 3
    seq_len_k = 4
    query_dim = 5
    key_dim = 5
    value_dim = 6
    
    queries = np.random.randn(seq_len_q, query_dim)
    keys = np.random.randn(seq_len_k, key_dim)
    values = np.random.randn(seq_len_k, value_dim)
    
    # Create attention mechanism
    attention = BasicAttention(query_dim, key_dim, value_dim)
    
    # Forward pass
    output, attention_weights = attention.forward(queries, keys, values)
    
    print(f"Input queries shape: {queries.shape}")
    print(f"Input keys shape: {keys.shape}")
    print(f"Input values shape: {values.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights:\n{attention_weights}")
    
    return attention, attention_weights

def self_attention_example():
    """Demonstrate self-attention mechanism"""
    print("\n🔍 Self-Attention Example")
    
    # Create sample data
    seq_len = 5
    input_dim = 8
    num_heads = 2
    
    x = np.random.randn(seq_len, input_dim)
    
    # Create self-attention mechanism
    self_attn = SelfAttention(input_dim, num_heads)
    
    # Forward pass
    output, attention_weights = self_attn.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights:\n{attention_weights}")
    
    return self_attn, attention_weights

def multi_head_attention_example():
    """Demonstrate multi-head attention mechanism"""
    print("\n🔍 Multi-Head Attention Example")
    
    # Create sample data
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    
    queries = np.random.randn(batch_size, seq_len, d_model)
    keys = np.random.randn(batch_size, seq_len, d_model)
    values = np.random.randn(batch_size, seq_len, d_model)
    
    # Create multi-head attention mechanism
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, attention_weights = mha.forward(queries, keys, values)
    
    print(f"Input queries shape: {queries.shape}")
    print(f"Input keys shape: {keys.shape}")
    print(f"Input values shape: {values.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test with causal mask
    causal_mask = mha.create_causal_mask(seq_len)
    output_masked, attention_weights_masked = mha.forward(queries, keys, values, causal_mask)
    
    print(f"With causal mask - Output shape: {output_masked.shape}")
    print(f"With causal mask - Attention weights shape: {attention_weights_masked.shape}")
    
    return mha, attention_weights

def visualize_attention_weights(attention_weights: np.ndarray, title: str = "Attention Weights"):
    """Visualize attention weights"""
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.savefig(f'attention_weights_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("🚀 Running Attention Mechanism Examples\n")
    
    # Basic attention
    basic_attn, basic_weights = basic_attention_example()
    visualize_attention_weights(basic_weights, "Basic Attention")
    
    # Self-attention
    self_attn, self_weights = self_attention_example()
    visualize_attention_weights(self_weights, "Self Attention")
    
    # Multi-head attention
    mha, mha_weights = multi_head_attention_example()
    visualize_attention_weights(mha_weights[0, 0], "Multi-Head Attention (Head 0)")
    
    print("\n✅ All attention examples completed!")

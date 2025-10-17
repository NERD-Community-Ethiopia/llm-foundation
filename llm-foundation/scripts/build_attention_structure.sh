#!/bin/bash

# Create attention mechanisms directory structure
mkdir -p src/attention/basic_attention
mkdir -p src/attention/self_attention
mkdir -p src/attention/multi_head_attention
mkdir -p src/attention/examples
mkdir -p src/attention/visualizations

# Create basic attention implementation
cat > src/attention/basic_attention/attention.py << 'EOF'
"""
Basic Attention Mechanism Implementation
"""
import numpy as np
from typing import Tuple, List

class BasicAttention:
    """
    Basic attention mechanism implementation
    """
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int):
        """
        Initialize attention mechanism
        
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key vectors  
            value_dim: Dimension of value vectors
        """
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Initialize weight matrices
        self.W_q = np.random.randn(query_dim, query_dim) * 0.01
        self.W_k = np.random.randn(key_dim, key_dim) * 0.01
        self.W_v = np.random.randn(value_dim, value_dim) * 0.01
        
    def forward(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of attention mechanism
        
        Args:
            queries: Query vectors (seq_len_q, query_dim)
            keys: Key vectors (seq_len_k, key_dim)
            values: Value vectors (seq_len_k, value_dim)
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Step 1: Transform queries, keys, and values
        Q = np.dot(queries, self.W_q)  # (seq_len_q, query_dim)
        K = np.dot(keys, self.W_k)     # (seq_len_k, key_dim)
        V = np.dot(values, self.W_v)   # (seq_len_k, value_dim)
        
        # Step 2: Compute attention scores
        # scores = Q * K^T / sqrt(d_k)
        scores = np.dot(Q, K.T) / np.sqrt(self.key_dim)  # (seq_len_q, seq_len_k)
        
        # Step 3: Apply softmax to get attention weights
        attention_weights = self.softmax(scores)  # (seq_len_q, seq_len_k)
        
        # Step 4: Apply attention weights to values
        attention_output = np.dot(attention_weights, V)  # (seq_len_q, value_dim)
        
        return attention_output, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray, 
                attention_weights: np.ndarray, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass of attention mechanism
        
        Args:
            queries: Query vectors from forward pass
            keys: Key vectors from forward pass
            values: Value vectors from forward pass
            attention_weights: Attention weights from forward pass
            grad_output: Gradient of output
            
        Returns:
            Tuple of gradients for queries, keys, and values
        """
        # Transform inputs
        Q = np.dot(queries, self.W_q)
        K = np.dot(keys, self.W_k)
        V = np.dot(values, self.W_v)
        
        # Gradient with respect to attention weights
        grad_attention_weights = np.dot(grad_output, V.T)
        
        # Gradient with respect to values
        grad_V = np.dot(attention_weights.T, grad_output)
        
        # Gradient with respect to scores
        grad_scores = self.softmax_gradient(attention_weights, grad_attention_weights)
        
        # Gradient with respect to Q and K
        grad_Q = np.dot(grad_scores, K) / np.sqrt(self.key_dim)
        grad_K = np.dot(grad_scores.T, Q) / np.sqrt(self.key_dim)
        
        # Gradient with respect to original inputs
        grad_queries = np.dot(grad_Q, self.W_q.T)
        grad_keys = np.dot(grad_K, self.W_k.T)
        grad_values = np.dot(grad_V, self.W_v.T)
        
        return grad_queries, grad_keys, grad_values
    
    def softmax_gradient(self, softmax_output: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradient of softmax function"""
        # This is a simplified version - in practice, you'd need the full Jacobian
        return grad_output * softmax_output * (1 - softmax_output)
EOF

# Create self-attention implementation
cat > src/attention/self_attention/self_attention.py << 'EOF'
"""
Self-Attention Mechanism Implementation
"""
import numpy as np
from typing import Tuple, List

class SelfAttention:
    """
    Self-attention mechanism where queries, keys, and values come from the same input
    """
    
    def __init__(self, input_dim: int, num_heads: int = 1):
        """
        Initialize self-attention
        
        Args:
            input_dim: Dimension of input vectors
            num_heads: Number of attention heads
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Initialize weight matrices for Q, K, V
        self.W_q = np.random.randn(input_dim, input_dim) * 0.01
        self.W_k = np.random.randn(input_dim, input_dim) * 0.01
        self.W_v = np.random.randn(input_dim, input_dim) * 0.01
        self.W_o = np.random.randn(input_dim, input_dim) * 0.01  # Output projection
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of self-attention
        
        Args:
            x: Input sequence (seq_len, input_dim)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        seq_len, input_dim = x.shape
        
        # Step 1: Linear transformations
        Q = np.dot(x, self.W_q)  # (seq_len, input_dim)
        K = np.dot(x, self.W_k)  # (seq_len, input_dim)
        V = np.dot(x, self.W_v)  # (seq_len, input_dim)
        
        # Step 2: Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim)
        K = K.reshape(seq_len, self.num_heads, self.head_dim)
        V = V.reshape(seq_len, self.num_heads, self.head_dim)
        
        # Step 3: Transpose for easier computation
        Q = Q.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        K = K.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        V = V.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        
        # Step 4: Compute attention for each head
        attention_outputs = []
        attention_weights = []
        
        for head in range(self.num_heads):
            # Compute attention scores
            scores = np.dot(Q[head], K[head].T) / np.sqrt(self.head_dim)
            
            # Apply softmax
            weights = self.softmax(scores)
            
            # Apply attention
            head_output = np.dot(weights, V[head])
            
            attention_outputs.append(head_output)
            attention_weights.append(weights)
        
        # Step 5: Concatenate heads
        attention_output = np.concatenate(attention_outputs, axis=1)  # (seq_len, input_dim)
        
        # Step 6: Final linear transformation
        output = np.dot(attention_output, self.W_o)
        
        # Average attention weights across heads
        avg_attention_weights = np.mean(attention_weights, axis=0)
        
        return output, avg_attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
EOF

# Create multi-head attention implementation
cat > src/attention/multi_head_attention/multi_head_attention.py << 'EOF'
"""
Multi-Head Attention Implementation
"""
import numpy as np
from typing import Tuple, List, Optional

class MultiHeadAttention:
    """
    Multi-head attention mechanism with scaled dot-product attention
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Linear transformations for Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
        # Initialize gradients
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)
        
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention
        
        Args:
            Q: Query matrix (batch_size, num_heads, seq_len_q, d_k)
            K: Key matrix (batch_size, num_heads, seq_len_k, d_k)
            V: Value matrix (batch_size, num_heads, seq_len_k, d_k)
            mask: Optional mask for padding or causal attention
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply dropout (simplified - just multiply by (1 - dropout))
        if self.dropout > 0:
            attention_weights = attention_weights * (1 - self.dropout)
        
        # Apply attention weights to values
        attention_output = np.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention
        
        Args:
            queries: Query matrix (batch_size, seq_len_q, d_model)
            keys: Key matrix (batch_size, seq_len_k, d_model)
            values: Value matrix (batch_size, seq_len_k, d_model)
            mask: Optional mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len_q, _ = queries.shape
        _, seq_len_k, _ = keys.shape
        
        # Linear transformations
        Q = np.dot(queries, self.W_q)  # (batch_size, seq_len_q, d_model)
        K = np.dot(keys, self.W_k)     # (batch_size, seq_len_k, d_model)
        V = np.dot(values, self.W_v)   # (batch_size, seq_len_k, d_model)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        
        # Final linear transformation
        output = np.dot(attention_output, self.W_o)
        
        return output, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create causal mask for decoder self-attention
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Causal mask (seq_len, seq_len)
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9  # Large negative value
        return mask
EOF

# Create attention examples
cat > src/attention/examples/attention_examples.py << 'EOF'
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
EOF

# Create attention visualizations
cat > src/attention/visualizations/attention_viz.py << 'EOF'
"""
Attention Mechanism Visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_attention_mechanism():
    """Plot the basic attention mechanism"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define positions
    query_pos = (1, 6)
    key_pos = (3, 6)
    value_pos = (5, 6)
    output_pos = (7, 6)
    
    # Draw boxes
    ax.add_patch(patches.Rectangle((0.5, 5.5), 1, 1, fill=False, edgecolor='blue', linewidth=2))
    ax.add_patch(patches.Rectangle((2.5, 5.5), 1, 1, fill=False, edgecolor='red', linewidth=2))
    ax.add_patch(patches.Rectangle((4.5, 5.5), 1, 1, fill=False, edgecolor='green', linewidth=2))
    ax.add_patch(patches.Rectangle((6.5, 5.5), 1, 1, fill=False, edgecolor='purple', linewidth=2))
    
    # Add labels
    ax.text(1, 6, 'Query', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(3, 6, 'Key', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(5, 6, 'Value', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(7, 6, 'Output', ha='center', va='center', fontsize=12, weight='bold')
    
    # Draw attention computation
    ax.arrow(1.5, 6, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(2, 6.3, 'Scores', ha='center', va='center', fontsize=10)
    
    ax.arrow(3.5, 6, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(4, 6.3, 'Weights', ha='center', va='center', fontsize=10)
    
    ax.arrow(5.5, 6, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(6, 6.3, 'Weighted Sum', ha='center', va='center', fontsize=10)
    
    # Add formula
    ax.text(4, 4, 'Attention(Q,K,V) = softmax(QK^T/√d_k)V', 
            ha='center', va='center', fontsize=14, weight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax.set_xlim(0, 8)
    ax.set_ylim(3.5, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Attention Mechanism', fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('attention_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_multi_head_attention():
    """Plot multi-head attention mechanism"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Draw input
    ax.add_patch(patches.Rectangle((1, 8), 2, 1, fill=False, edgecolor='blue', linewidth=2))
    ax.text(2, 8.5, 'Input', ha='center', va='center', fontsize=12, weight='bold')
    
    # Draw linear transformations
    for i in range(3):
        ax.add_patch(patches.Rectangle((0.5, 6-i*1.5), 1, 0.8, fill=False, edgecolor='red', linewidth=2))
        ax.text(1, 6.4-i*1.5, f'Q{i+1}', ha='center', va='center', fontsize=10)
        
        ax.add_patch(patches.Rectangle((2.5, 6-i*1.5), 1, 0.8, fill=False, edgecolor='green', linewidth=2))
        ax.text(3, 6.4-i*1.5, f'K{i+1}', ha='center', va='center', fontsize=10)
        
        ax.add_patch(patches.Rectangle((4.5, 6-i*1.5), 1, 0.8, fill=False, edgecolor='orange', linewidth=2))
        ax.text(5, 6.4-i*1.5, f'V{i+1}', ha='center', va='center', fontsize=10)
    
    # Draw attention heads
    for i in range(3):
        ax.add_patch(patches.Rectangle((6.5, 6-i*1.5), 1, 0.8, fill=False, edgecolor='purple', linewidth=2))
        ax.text(7, 6.4-i*1.5, f'Head {i+1}', ha='center', va='center', fontsize=10)
    
    # Draw concatenation
    ax.add_patch(patches.Rectangle((8.5, 5), 1, 2, fill=False, edgecolor='brown', linewidth=2))
    ax.text(9, 6, 'Concat', ha='center', va='center', fontsize=12, weight='bold')
    
    # Draw output
    ax.add_patch(patches.Rectangle((10.5, 5.5), 1, 1, fill=False, edgecolor='black', linewidth=2))
    ax.text(11, 6, 'Output', ha='center', va='center', fontsize=12, weight='bold')
    
    # Add arrows
    for i in range(3):
        # Input to Q,K,V
        ax.arrow(2, 8, -0.3, -1.1-i*1.5, head_width=0.05, head_length=0.05, fc='black', ec='black')
        ax.arrow(2, 8, 0.3, -1.1-i*1.5, head_width=0.05, head_length=0.05, fc='black', ec='black')
        ax.arrow(2, 8, 0.9, -1.1-i*1.5, head_width=0.05, head_length=0.05, fc='black', ec='black')
        
        # Q,K,V to heads
        ax.arrow(1.5, 6.4-i*1.5, 0.8, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
        ax.arrow(3.5, 6.4-i*1.5, 0.8, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
        ax.arrow(5.5, 6.4-i*1.5, 0.8, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
        
        # Heads to concat
        ax.arrow(7.5, 6.4-i*1.5, 0.8, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # Concat to output
    ax.arrow(9.5, 6, 0.8, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(4, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multi-Head Attention Mechanism', fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("�� Creating Attention Visualizations...")
    
    plot_attention_mechanism()
    plot_multi_head_attention()
    
    print("✅ Attention visualizations created!")
EOF

# Create __init__.py files
touch src/attention/__init__.py
touch src/attention/basic_attention/__init__.py
touch src/attention/self_attention/__init__.py
touch src/attention/multi_head_attention/__init__.py
touch src/attention/examples/__init__.py
touch src/attention/visualizations/__init__.py

echo "✅ Attention mechanisms structure created successfully!"
echo "📁 Created directories:"
echo "   - src/attention/basic_attention/"
echo "   - src/attention/self_attention/"
echo "   - src/attention/multi_head_attention/"
echo "   - src/attention/examples/"
echo "   - src/attention/visualizations/"
echo ""
echo "�� Ready to implement attention mechanisms!" 
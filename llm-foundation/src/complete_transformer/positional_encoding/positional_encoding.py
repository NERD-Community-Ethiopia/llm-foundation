"""
Positional Encoding Implementation
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class PositionalEncoding:
    """
    Positional encoding for Transformer
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length
        """
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create positional encoding matrix
        self.pe = np.zeros((max_seq_length, d_model))
        
        # Compute positional encoding
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                # Even indices: sin
                self.pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                # Odd indices: cos
                if i + 1 < d_model:
                    self.pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model) or (seq_length, d_model)
            
        Returns:
            Input with positional encoding added
        """
        if len(x.shape) == 3:
            # Batch input: (batch_size, seq_length, d_model)
            batch_size, seq_length, d_model = x.shape
            return x + self.pe[:seq_length, :].reshape(1, seq_length, d_model)
        else:
            # Single sequence: (seq_length, d_model)
            seq_length = x.shape[0]
            return x + self.pe[:seq_length, :]
    
    def visualize(self, seq_length: int = 50):
        """Visualize positional encoding"""
        plt.figure(figsize=(12, 8))
        plt.imshow(self.pe[:seq_length, :].T, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Positional Encoding Visualization')
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.show()

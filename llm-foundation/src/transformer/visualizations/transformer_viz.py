"""
Transformer Visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attention_weights, title="Attention Weights"):
    """Plot attention weights as heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, annot=True, cmap='viridis', fmt='.2f')
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()

def plot_positional_encoding(d_model=64, max_seq_length=50):
    """Plot positional encoding"""
    pe = np.zeros((max_seq_length, d_model))
    
    for pos in range(max_seq_length):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pe.T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.show()

def plot_transformer_architecture():
    """Plot Transformer architecture diagram"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Define positions
    positions = {
        'input_embed': (1, 8),
        'pos_encoding': (1, 7),
        'encoder': (1, 6),
        'decoder': (1, 4),
        'output': (1, 2),
        'attention': (3, 6),
        'ffn': (3, 4),
        'norm': (5, 6)
    }
    
    # Draw boxes
    for name, pos in positions.items():
        ax.add_patch(plt.Rectangle((pos[0]-0.3, pos[1]-0.3), 0.6, 0.6, 
                                 fill=False, edgecolor='black', linewidth=2))
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=10)
    
    # Draw arrows
    arrows = [
        ((1, 8), (1, 7)),
        ((1, 7), (1, 6)),
        ((1, 6), (1, 4)),
        ((1, 4), (1, 2)),
        ((1, 6), (3, 6)),
        ((3, 6), (3, 4)),
        ((3, 4), (5, 6))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.set_xlim(0, 6)
    ax.set_ylim(1, 9)
    ax.set_title('Transformer Architecture', fontsize=16)
    ax.axis('off')
    plt.show()

if __name__ == "__main__":
    plot_positional_encoding()
    plot_transformer_architecture()

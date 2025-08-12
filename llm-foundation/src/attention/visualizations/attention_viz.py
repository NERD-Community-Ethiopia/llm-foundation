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

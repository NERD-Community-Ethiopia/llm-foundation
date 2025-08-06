#!/usr/bin/env python3
"""
RNN Architecture Visualization
Shows the unfolding concept and information flow through time
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patches as mpatches

def create_rnn_architecture_diagram():
    """Create a visual representation of RNN unfolding"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Folded RNN
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('Folded RNN (Single Recurrent Unit)', fontsize=14, fontweight='bold')
    
    # Draw the RNN unit
    unit = FancyBboxPatch((4, 4), 2, 2, boxstyle="round,pad=0.1", 
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(unit)
    ax1.text(5, 5, 'RNN\nUnit', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Input arrow
    ax1.arrow(2, 5, 1.8, 0, head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2)
    ax1.text(1.5, 5.5, 'x', fontsize=14, fontweight='bold', color='red')
    ax1.text(2.5, 4.5, 'W_xh', fontsize=10, color='red')
    
    # Output arrow
    ax1.arrow(6, 5, 1.8, 0, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=2)
    ax1.text(8.5, 5.5, 'y', fontsize=14, fontweight='bold', color='green')
    ax1.text(7.5, 4.5, 'W_hy', fontsize=10, color='green')
    
    # Recurrent connection (self-loop)
    circle = plt.Circle((5, 3), 0.5, fill=False, color='blue', linewidth=2)
    ax1.add_patch(circle)
    ax1.arrow(5, 3.5, 0, 0.8, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    ax1.text(6.5, 3, 'W_hh', fontsize=10, color='blue')
    ax1.text(5, 2.5, 'h_{t-1}', fontsize=10, color='blue')
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Right: Unfolded RNN
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 10)
    ax2.set_title('Unfolded RNN (Multiple Time Steps)', fontsize=14, fontweight='bold')
    
    # Draw multiple RNN units
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    for i in range(4):
        # RNN unit
        unit = FancyBboxPatch((i*2.5 + 0.5, 4), 1.5, 2, boxstyle="round,pad=0.1", 
                             facecolor=colors[i], edgecolor='black', linewidth=2)
        ax2.add_patch(unit)
        ax2.text(i*2.5 + 1.25, 5, f'RNN\nt={i}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Input
        ax2.arrow(i*2.5 + 0.2, 5, 0.2, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=1.5)
        ax2.text(i*2.5 + 0.1, 5.3, f'x_{i}', fontsize=10, color='red')
        
        # Output
        ax2.arrow(i*2.5 + 1.8, 5, 0.2, 0, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=1.5)
        ax2.text(i*2.5 + 2.1, 5.3, f'y_{i}', fontsize=10, color='green')
        
        # Hidden state connections
        if i > 0:
            # Previous to current
            ax2.arrow(i*2.5 + 0.2, 4.5, -0.2, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=1.5)
            ax2.text(i*2.5 - 0.1, 4.3, f'h_{i-1}', fontsize=8, color='blue')
    
    # Add shared weights note
    ax2.text(6, 8, 'Shared Weights:\nW_xh, W_hh, W_hy', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/rnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_information_flow_diagram():
    """Show how information flows through time in an RNN"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create a timeline
    time_steps = ['t=0', 't=1', 't=2', 't=3', 't=4']
    inputs = ['x₀', 'x₁', 'x₂', 'x₃', 'x₄']
    hidden_states = ['h₀', 'h₁', 'h₂', 'h₃', 'h₄']
    outputs = ['y₀', 'y₁', 'y₂', 'y₃', 'y₄']
    
    # Draw boxes for each time step
    for i, (t, x, h, y) in enumerate(zip(time_steps, inputs, hidden_states, outputs)):
        x_pos = i * 2.5
        
        # Input box
        input_box = FancyBboxPatch((x_pos - 0.8, 6), 1.6, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor='lightcoral', edgecolor='red', linewidth=2)
        ax.add_patch(input_box)
        ax.text(x_pos, 6.4, x, ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x_pos, 5.8, 'Input', ha='center', va='center', fontsize=10)
        
        # Hidden state box
        hidden_box = FancyBboxPatch((x_pos - 0.8, 4), 1.6, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(hidden_box)
        ax.text(x_pos, 4.4, h, ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x_pos, 3.8, 'Memory', ha='center', va='center', fontsize=10)
        
        # Output box
        output_box = FancyBboxPatch((x_pos - 0.8, 2), 1.6, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax.add_patch(output_box)
        ax.text(x_pos, 2.4, y, ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(x_pos, 1.8, 'Output', ha='center', va='center', fontsize=10)
        
        # Time step label
        ax.text(x_pos, 0.5, t, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw information flow arrows
    for i in range(len(time_steps) - 1):
        x_pos = i * 2.5
        
        # Input to hidden
        ax.arrow(x_pos, 5.8, 0.5, -0.5, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
        ax.text(x_pos + 0.25, 5.3, 'W_xh', fontsize=10, color='red')
        
        # Hidden to output
        ax.arrow(x_pos, 3.8, 0.5, -0.5, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)
        ax.text(x_pos + 0.25, 3.3, 'W_hy', fontsize=10, color='green')
        
        # Recurrent connection (hidden to next hidden)
        if i < len(time_steps) - 1:
            ax.arrow(x_pos + 0.8, 4.4, 0.9, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
            ax.text(x_pos + 1.25, 4.8, 'W_hh', fontsize=10, color='blue')
    
    # Add formula
    ax.text(6, 7.5, 'h_t = tanh(W_xh × x_t + W_hh × h_{t-1} + b_h)', 
            fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(6, 7, 'y_t = softmax(W_hy × h_t + b_y)', 
            fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax.set_xlim(-1, 12)
    ax.set_ylim(0, 8)
    ax.set_title('RNN Information Flow Through Time', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/rnn_information_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_vanishing_gradient_diagram():
    """Show the vanishing gradient problem in RNNs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: RNN gradient flow
    ax1.set_title('RNN: Vanishing Gradients', fontsize=14, fontweight='bold')
    
    # Draw RNN units with decreasing gradient
    for i in range(5):
        x_pos = i * 2
        alpha = 1.0 - i * 0.15  # Decreasing opacity
        
        unit = FancyBboxPatch((x_pos, 2), 1.5, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='black', linewidth=2, alpha=alpha)
        ax1.add_patch(unit)
        ax1.text(x_pos + 0.75, 2.5, f't={i}', ha='center', va='center', fontsize=10)
        
        if i > 0:
            # Gradient arrow with decreasing thickness
            arrow_width = 2 - i * 0.3
            ax1.arrow(x_pos, 2.5, -1.5, 0, head_width=0.1, head_length=0.1, 
                     fc='red', ec='red', linewidth=arrow_width, alpha=alpha)
            ax1.text(x_pos - 0.75, 2.8, f'grad_{i}', fontsize=8, color='red')
    
    ax1.text(5, 4, 'Gradient gets smaller\nas it flows back', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(1, 5)
    ax1.axis('off')
    
    # Right: LSTM gradient flow
    ax2.set_title('LSTM: Better Gradient Flow', fontsize=14, fontweight='bold')
    
    # Draw LSTM units with consistent gradient
    for i in range(5):
        x_pos = i * 2
        
        unit = FancyBboxPatch((x_pos, 2), 1.5, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax2.add_patch(unit)
        ax2.text(x_pos + 0.75, 2.5, f't={i}', ha='center', va='center', fontsize=10)
        
        if i > 0:
            # Consistent gradient arrow
            ax2.arrow(x_pos, 2.5, -1.5, 0, head_width=0.1, head_length=0.1, 
                     fc='blue', ec='blue', linewidth=2)
            ax2.text(x_pos - 0.75, 2.8, f'grad_{i}', fontsize=8, color='blue')
    
    # Cell state highway
    ax2.plot([0, 8], [1.2, 1.2], 'orange', linewidth=3, label='Cell State Highway')
    ax2.text(4, 0.8, 'Cell State: Direct gradient flow', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/vanishing_gradient_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_lstm_gates_diagram():
    """Show LSTM gates and their functions"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Draw LSTM cell
    cell = FancyBboxPatch((3, 3), 6, 4, boxstyle="round,pad=0.2", 
                         facecolor='lightblue', edgecolor='black', linewidth=3)
    ax.add_patch(cell)
    ax.text(6, 5, 'LSTM Cell', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input gate
    input_gate = FancyBboxPatch((1, 6), 1.5, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(input_gate)
    ax.text(1.75, 6.4, 'Input Gate', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.75, 6, 'i_t = σ(...)', ha='center', va='center', fontsize=8)
    
    # Forget gate
    forget_gate = FancyBboxPatch((1, 4.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(forget_gate)
    ax.text(1.75, 4.9, 'Forget Gate', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.75, 4.5, 'f_t = σ(...)', ha='center', va='center', fontsize=8)
    
    # Output gate
    output_gate = FancyBboxPatch((1, 3), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(output_gate)
    ax.text(1.75, 3.4, 'Output Gate', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.75, 3, 'o_t = σ(...)', ha='center', va='center', fontsize=8)
    
    # Cell state
    cell_state = FancyBboxPatch((8, 4.5), 1.5, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightgray', edgecolor='purple', linewidth=2)
    ax.add_patch(cell_state)
    ax.text(8.75, 5, 'Cell State', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8.75, 4.6, 'c_t', ha='center', va='center', fontsize=12)
    
    # Arrows
    ax.arrow(2.5, 6.4, 0.4, -0.4, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.arrow(2.5, 4.9, 0.4, -0.4, head_width=0.1, head_length=0.1, fc='orange', ec='orange', linewidth=2)
    ax.arrow(2.5, 3.4, 0.4, -0.4, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)
    ax.arrow(8, 5, -0.4, 0, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
    
    # Formulas
    ax.text(6, 2, 'c_t = f_t × c_{t-1} + i_t × c̃_t', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    ax.text(6, 1.5, 'h_t = o_t × tanh(c_t)', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_title('LSTM Gates and Cell State', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/lstm_gates.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all RNN visualizations"""
    print("🎨 Creating RNN Architecture Visualizations...")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Generate all diagrams
    create_rnn_architecture_diagram()
    print("✅ RNN Architecture Diagram created")
    
    create_information_flow_diagram()
    print("✅ Information Flow Diagram created")
    
    create_vanishing_gradient_diagram()
    print("✅ Vanishing Gradient Comparison created")
    
    create_lstm_gates_diagram()
    print("✅ LSTM Gates Diagram created")
    
    print("\n🎉 All visualizations saved to 'plots/' directory!")
    print("📁 Generated files:")
    print("  - plots/rnn_architecture.png")
    print("  - plots/rnn_information_flow.png")
    print("  - plots/vanishing_gradient_comparison.png")
    print("  - plots/lstm_gates.png")

if __name__ == "__main__":
    main() 
"""
RNN Training Examples
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from rnn_lstm.basic_rnn.rnn import SimpleRNN
from rnn_lstm.lstm.lstm import LSTM
from rnn_lstm.sequence_models.sequence_utils import (
    create_sequences, one_hot_encode, create_vocab,
    text_to_sequences, sample_from_model
)

def simple_sequence_example():
    """
    Train RNN on a simple sequence prediction task
    """
    print("🔍 Training RNN on Simple Sequence Prediction")
    
    # Create simple sequence: [1, 2, 3, 1, 2, 3, ...]
    data = [1, 2, 3] * 100
    seq_length = 3
    vocab_size = 4  # 0, 1, 2, 3
    
    # Create sequences
    sequences, targets = create_sequences(data, seq_length)
    
    # Convert to one-hot encoding
    X = []
    y = []
    
    for seq, target in zip(sequences, targets):
        # Convert sequence to one-hot
        seq_one_hot = []
        for token in seq:
            one_hot = np.zeros((vocab_size, 1))
            one_hot[token] = 1
            seq_one_hot.append(one_hot)
        X.append(seq_one_hot)
        
        # Convert target to one-hot
        target_one_hot = np.zeros((vocab_size, 1))
        target_one_hot[target] = 1
        y.append(target_one_hot)
    
    # Create RNN
    rnn = SimpleRNN(input_size=vocab_size, hidden_size=10, output_size=vocab_size)
    
    # Training
    learning_rate = 0.1
    epochs = 100
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(X)):
            # Forward pass
            outputs, hidden_states = rnn.forward(X[i])
            
            # Compute loss (cross-entropy)
            loss = -np.sum(y[i] * np.log(outputs[-1] + 1e-8))
            total_loss += loss
            
            # Backward pass
            target_sequence = [y[i]] * len(outputs)
            rnn.backward(X[i], hidden_states, outputs, target_sequence)
            rnn.update_weights(learning_rate)
        
        avg_loss = total_loss / len(X)
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('RNN Training Loss - Simple Sequence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('plots/rnn_simple_sequence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Final Loss: {losses[-1]:.4f}")
    return rnn, losses

def text_generation_example():
    """
    Train LSTM on text generation task
    """
    print("\n🔍 Training LSTM on Text Generation")
    
    # Simple text for demonstration - MUCH SHORTER
    text = "hello world this is a simple example" * 3  # Reduced from * 10
    
    # Create vocabulary
    char_to_idx, idx_to_char = create_vocab(text)
    vocab_size = len(char_to_idx)
    
    # Convert text to sequences
    data = text_to_sequences(text, char_to_idx)
    seq_length = 20
    
    # Create sequences
    sequences, targets = create_sequences(data, seq_length)
    
    # Convert to one-hot encoding
    X = []
    y = []
    
    for seq, target in zip(sequences, targets):
        # Convert sequence to one-hot
        seq_one_hot = []
        for token in seq:
            one_hot = np.zeros((vocab_size, 1))
            one_hot[token] = 1
            seq_one_hot.append(one_hot)
        X.append(seq_one_hot)
        
        # Convert target to one-hot
        target_one_hot = np.zeros((vocab_size, 1))
        target_one_hot[target] = 1
        y.append(target_one_hot)
    
    # Create LSTM - SMALLER MODEL
    lstm = LSTM(input_size=vocab_size, hidden_size=20, output_size=vocab_size)  # Reduced from 50
    
    # Training - MUCH FASTER
    learning_rate = 0.2  # Increased from 0.1
    epochs = 10  # Reduced from 20
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(X)):
            # Forward pass
            outputs, hidden_states, cell_states = lstm.forward(X[i])
            
            # Compute loss
            loss = -np.sum(y[i] * np.log(outputs[-1] + 1e-8))
            total_loss += loss
            
            # Backward pass
            target_sequence = [np.zeros_like(outputs[0])] * (len(outputs) - 1) + [y[i]]
            lstm.backward(X[i], hidden_states, cell_states, outputs, target_sequence)
            lstm.update_weights(learning_rate)
        
        avg_loss = total_loss / len(X)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('LSTM Training Loss - Text Generation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('plots/lstm_text_generation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate some text
    seed_text = "hello world"
    generated_text = sample_from_model(lstm, seed_text, char_to_idx, idx_to_char, 
                                     length=50, temperature=0.8)
    
    print(f"\nGenerated text: {generated_text}")
    print(f"Final Loss: {losses[-1]:.4f}")
    
    return lstm, losses, generated_text

if __name__ == "__main__":
    print("🚀 Running RNN & LSTM Examples\n")
    
    # Simple sequence example
    rnn, rnn_losses = simple_sequence_example()
    
    # Text generation example - COMMENT OUT FOR SPEED
    lstm, lstm_losses, generated_text = text_generation_example()
    
    print("\n✅ RNN example completed!")  # Changed message

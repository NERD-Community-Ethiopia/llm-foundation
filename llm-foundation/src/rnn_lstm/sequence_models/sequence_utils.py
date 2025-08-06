"""
Utilities for sequence modeling
"""
import numpy as np
from typing import List, Tuple, Dict
import random

def create_sequences(data: List[int], seq_length: int) -> Tuple[List[List[int]], List[int]]:
    """
    Create sequences for training
    
    Args:
        data: List of integers (e.g., text as character indices)
        seq_length: Length of each sequence
        
    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return sequences, targets

def one_hot_encode(sequences: List[List[int]], vocab_size: int) -> List[np.ndarray]:
    """
    Convert sequences to one-hot encoded format
    
    Args:
        sequences: List of sequences
        vocab_size: Size of vocabulary
        
    Returns:
        List of one-hot encoded sequences
    """
    encoded_sequences = []
    
    for seq in sequences:
        encoded_seq = []
        for token in seq:
            one_hot = np.zeros((vocab_size, 1))
            one_hot[token] = 1
            encoded_seq.append(one_hot)
        encoded_sequences.append(encoded_seq)
    
    return encoded_sequences

def create_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create vocabulary from text
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (char_to_idx, idx_to_char) dictionaries
    """
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return char_to_idx, idx_to_char

def text_to_sequences(text: str, char_to_idx: Dict[str, int]) -> List[int]:
    """
    Convert text to sequence of indices
    
    Args:
        text: Input text
        char_to_idx: Character to index mapping
        
    Returns:
        List of indices
    """
    return [char_to_idx[ch] for ch in text]

def sequences_to_text(sequences: List[int], idx_to_char: Dict[int, str]) -> str:
    """
    Convert sequence of indices back to text
    
    Args:
        sequences: List of indices
        idx_to_char: Index to character mapping
        
    Returns:
        Text string
    """
    return ''.join([idx_to_char[idx] for idx in sequences])

def sample_from_model(model, seed_text: str, char_to_idx: Dict[str, int], 
                     idx_to_char: Dict[int, str], length: int = 100, 
                     temperature: float = 1.0) -> str:
    """
    Generate text using trained model
    
    Args:
        model: Trained RNN/LSTM model
        seed_text: Starting text
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        length: Length of text to generate
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Generated text
    """
    vocab_size = len(char_to_idx)
    generated_text = seed_text
    
    # Convert seed text to sequence
    seed_sequence = text_to_sequences(seed_text, char_to_idx)
    
    for _ in range(length):
        # Prepare input
        x = np.array(seed_sequence[-1]).reshape(-1, 1)
        x_one_hot = np.zeros((vocab_size, 1))
        x_one_hot[x] = 1
        
        # Get prediction
        output = model.predict([x_one_hot])[0]
        
        # Apply temperature
        output = np.log(output) / temperature
        output = np.exp(output)
        output = output / np.sum(output)
        
        # Sample next character
        next_char_idx = np.random.choice(vocab_size, p=output.flatten())
        next_char = idx_to_char[next_char_idx]
        
        generated_text += next_char
        seed_sequence.append(next_char_idx)
    
    return generated_text

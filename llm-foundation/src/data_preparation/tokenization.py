"""
Tokenization and Vocabulary Building
"""
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np


class Tokenizer:
    """
    Simple word-level tokenizer for translation tasks
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "<UNK>", 
                 pad_token: str = "<PAD>", start_token: str = "<START>", 
                 end_token: str = "<END>"):
        """
        Initialize tokenizer with vocabulary
        
        Args:
            vocab: Dictionary mapping tokens to indices
            unk_token: Unknown token
            pad_token: Padding token
            start_token: Start of sequence token
            end_token: End of sequence token
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        
        # Create reverse mapping
        self.idx_to_token = {idx: token for token, idx in vocab.items()}
        
        # Get special token indices
        self.unk_idx = vocab.get(unk_token, 0)
        self.pad_idx = vocab.get(pad_token, 0)
        self.start_idx = vocab.get(start_token, 1)
        self.end_idx = vocab.get(end_token, 2)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple word-level tokenization
        # Convert to lowercase and split on whitespace
        text = text.lower().strip()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token indices
        
        Args:
            text: Input text
            add_special_tokens: Whether to add start/end tokens
            
        Returns:
            List of token indices
        """
        tokens = self.tokenize(text)
        indices = []
        
        if add_special_tokens:
            indices.append(self.start_idx)
        
        for token in tokens:
            idx = self.vocab.get(token, self.unk_idx)
            indices.append(idx)
        
        if add_special_tokens:
            indices.append(self.end_idx)
        
        return indices
    
    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """
        Decode token indices back to text
        
        Args:
            indices: List of token indices
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for idx in indices:
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]
                
                # Skip special tokens if requested
                if remove_special_tokens and token in [self.start_token, self.end_token, self.pad_token]:
                    continue
                    
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab)


def build_vocabulary(texts: List[str], min_freq: int = 2, 
                    max_vocab_size: Optional[int] = None,
                    special_tokens: List[str] = None) -> Dict[str, int]:
    """
    Build vocabulary from list of texts
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a token to be included
        max_vocab_size: Maximum vocabulary size (including special tokens)
        special_tokens: List of special tokens to add
        
    Returns:
        Dictionary mapping tokens to indices
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
    
    # Count token frequencies
    tokenizer = Tokenizer({})  # Empty vocab for tokenization
    counter = Counter()
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        counter.update(tokens)
    
    # Filter by minimum frequency
    vocab_tokens = [token for token, count in counter.items() if count >= min_freq]
    
    # Limit vocabulary size if specified
    if max_vocab_size is not None:
        max_regular_tokens = max_vocab_size - len(special_tokens)
        vocab_tokens = vocab_tokens[:max_regular_tokens]
    
    # Build vocabulary dictionary
    vocab = {}
    
    # Add special tokens first
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    # Add regular tokens
    for i, token in enumerate(vocab_tokens):
        vocab[token] = i + len(special_tokens)
    
    return vocab


def create_sample_data() -> Tuple[List[str], List[str]]:
    """
    Create sample translation data for testing
    
    Returns:
        Tuple of (source_texts, target_texts)
    """
    source_texts = [
        "hello world",
        "the cat sat on the mat",
        "i love you",
        "good morning",
        "how are you",
        "thank you",
        "goodbye",
        "please help",
        "where is the",
        "what time is it"
    ]
    
    target_texts = [
        "bonjour monde",
        "le chat s'est assis sur le tapis",
        "je t'aime",
        "bonjour",
        "comment allez-vous",
        "merci",
        "au revoir",
        "s'il vous plaît aidez",
        "où est le",
        "quelle heure est-il"
    ]
    
    return source_texts, target_texts


if __name__ == "__main__":
    # Test tokenization
    print("🧠 Testing Tokenization")
    print("=" * 40)
    
    # Create sample data
    source_texts, target_texts = create_sample_data()
    
    # Build vocabulary from source texts
    vocab = build_vocabulary(source_texts, min_freq=1, max_vocab_size=50)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: {list(vocab.keys())[:4]}")
    print(f"Regular tokens: {list(vocab.keys())[4:10]}")
    
    # Create tokenizer
    tokenizer = Tokenizer(vocab)
    
    # Test encoding/decoding
    test_text = "hello world"
    print(f"\nTest text: '{test_text}'")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    print("\n✅ Tokenization test completed!")

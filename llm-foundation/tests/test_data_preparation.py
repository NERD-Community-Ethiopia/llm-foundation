"""
Tests for Data Preparation Module (Checkpoint 6)
"""
import numpy as np
import pytest
from src.data_preparation.tokenization import Tokenizer, build_vocabulary, create_sample_data
from src.data_preparation.dataset import TranslationDataset, create_data_loader, split_dataset
from src.data_preparation.collate import collate_batch, create_padding_mask, create_causal_mask


class TestTokenizer:
    """Test tokenizer functionality"""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "hello": 4, "world": 5}
        tokenizer = Tokenizer(vocab)
        
        assert len(tokenizer) == 6
        assert tokenizer.pad_idx == 0
        assert tokenizer.start_idx == 1
        assert tokenizer.end_idx == 2
        assert tokenizer.unk_idx == 3
    
    def test_tokenize(self):
        """Test text tokenization"""
        vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "hello": 4, "world": 5}
        tokenizer = Tokenizer(vocab)
        
        tokens = tokenizer.tokenize("Hello World!")
        assert tokens == ["hello", "world"]
    
    def test_encode_decode(self):
        """Test encoding and decoding"""
        vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "hello": 4, "world": 5}
        tokenizer = Tokenizer(vocab)
        
        text = "hello world"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert encoded == [1, 4, 5, 2]  # START, hello, world, END
        assert decoded == "hello world"
    
    def test_unknown_tokens(self):
        """Test handling of unknown tokens"""
        vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "hello": 4}
        tokenizer = Tokenizer(vocab)
        
        encoded = tokenizer.encode("hello unknown")
        assert encoded == [1, 4, 3, 2]  # START, hello, UNK, END


class TestVocabulary:
    """Test vocabulary building"""
    
    def test_build_vocabulary(self):
        """Test vocabulary building from texts"""
        texts = ["hello world", "hello there", "good morning"]
        vocab = build_vocabulary(texts, min_freq=1, max_vocab_size=10)
        
        assert "<PAD>" in vocab
        assert "<START>" in vocab
        assert "<END>" in vocab
        assert "<UNK>" in vocab
        assert "hello" in vocab
        assert len(vocab) <= 10
    
    def test_minimum_frequency(self):
        """Test minimum frequency filtering"""
        texts = ["hello world", "hello there", "good morning"]
        vocab = build_vocabulary(texts, min_freq=2, max_vocab_size=20)
        
        # "hello" appears twice, should be included
        assert "hello" in vocab
        # "world", "there", "good", "morning" appear once, should be excluded
        assert "world" not in vocab
        assert "there" not in vocab


class TestDataset:
    """Test dataset functionality"""
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        source_texts = ["hello world", "good morning"]
        target_texts = ["bonjour monde", "bonjour"]
        
        source_vocab = build_vocabulary(source_texts, min_freq=1, max_vocab_size=20)
        target_vocab = build_vocabulary(target_texts, min_freq=1, max_vocab_size=20)
        
        source_tokenizer = Tokenizer(source_vocab)
        target_tokenizer = Tokenizer(target_vocab)
        
        dataset = TranslationDataset(
            source_texts, target_texts,
            source_tokenizer, target_tokenizer,
            max_source_length=5,
            max_target_length=6
        )
        
        assert len(dataset) == 2
        assert dataset.get_vocab_sizes() == (len(source_vocab), len(target_vocab))
    
    def test_dataset_getitem(self):
        """Test dataset indexing"""
        source_texts = ["hello world"]
        target_texts = ["bonjour monde"]
        
        source_vocab = build_vocabulary(source_texts, min_freq=1, max_vocab_size=20)
        target_vocab = build_vocabulary(target_texts, min_freq=1, max_vocab_size=20)
        
        source_tokenizer = Tokenizer(source_vocab)
        target_tokenizer = Tokenizer(target_vocab)
        
        dataset = TranslationDataset(
            source_texts, target_texts,
            source_tokenizer, target_tokenizer
        )
        
        src_seq, tgt_seq = dataset[0]
        assert isinstance(src_seq, list)
        assert isinstance(tgt_seq, list)
        assert len(src_seq) > 0
        assert len(tgt_seq) > 0


class TestDataLoader:
    """Test data loader functionality"""
    
    def test_create_data_loader(self):
        """Test data loader creation"""
        source_texts = ["hello world", "good morning", "how are you"]
        target_texts = ["bonjour monde", "bonjour", "comment allez-vous"]
        
        source_vocab = build_vocabulary(source_texts, min_freq=1, max_vocab_size=20)
        target_vocab = build_vocabulary(target_texts, min_freq=1, max_vocab_size=20)
        
        source_tokenizer = Tokenizer(source_vocab)
        target_tokenizer = Tokenizer(target_vocab)
        
        dataset = TranslationDataset(
            source_texts, target_texts,
            source_tokenizer, target_tokenizer
        )
        
        data_loader = create_data_loader(dataset, batch_size=2, shuffle=False)
        
        assert len(data_loader) > 0
        
        # Test first batch
        src_batch, tgt_batch = data_loader[0]
        assert isinstance(src_batch, np.ndarray)
        assert isinstance(tgt_batch, np.ndarray)
        assert src_batch.shape[0] <= 2  # batch_size
        assert tgt_batch.shape[0] <= 2  # batch_size


class TestCollate:
    """Test collation functions"""
    
    def test_collate_batch(self):
        """Test batch collation"""
        batch_data = [
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2], [4, 5]),
            ([1, 2, 3, 4], [4, 5, 6, 7, 8])
        ]
        
        src_batch, tgt_batch = collate_batch(batch_data, source_pad_idx=0, target_pad_idx=0)
        
        assert src_batch.shape == (3, 4)  # batch_size, max_source_len
        assert tgt_batch.shape == (3, 5)  # batch_size, max_target_len
        assert src_batch.dtype == np.int64
        assert tgt_batch.dtype == np.int64
    
    def test_create_padding_mask(self):
        """Test padding mask creation"""
        sequences = np.array([
            [1, 2, 3, 0],
            [1, 2, 0, 0],
            [1, 2, 3, 4]
        ])
        
        mask = create_padding_mask(sequences, pad_idx=0)
        
        expected = np.array([
            [False, False, False, True],
            [False, False, True, True],
            [False, False, False, False]
        ])
        
        np.testing.assert_array_equal(mask, expected)
    
    def test_create_causal_mask(self):
        """Test causal mask creation"""
        mask = create_causal_mask(4)
        
        expected = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False]
        ])
        
        np.testing.assert_array_equal(mask, expected)


class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self):
        """Test complete data preparation pipeline"""
        # Create sample data
        source_texts, target_texts = create_sample_data()
        
        # Build vocabularies
        source_vocab = build_vocabulary(source_texts, min_freq=1, max_vocab_size=30)
        target_vocab = build_vocabulary(target_texts, min_freq=1, max_vocab_size=30)
        
        # Create tokenizers
        source_tokenizer = Tokenizer(source_vocab)
        target_tokenizer = Tokenizer(target_vocab)
        
        # Create dataset
        dataset = TranslationDataset(
            source_texts, target_texts,
            source_tokenizer, target_tokenizer,
            max_source_length=10,
            max_target_length=12
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        
        # Create data loader
        train_loader = create_data_loader(train_dataset, batch_size=2)
        
        # Test a batch
        src_batch, tgt_batch = train_loader[0]
        
        assert src_batch.shape[0] <= 2
        assert tgt_batch.shape[0] <= 2
        assert src_batch.shape[1] > 0
        assert tgt_batch.shape[1] > 0
        
        # Test tokenization round-trip
        sample_idx = 0
        src_seq = source_tokenizer.decode(src_batch[sample_idx], remove_special_tokens=True)
        tgt_seq = target_tokenizer.decode(tgt_batch[sample_idx], remove_special_tokens=True)
        
        assert isinstance(src_seq, str)
        assert isinstance(tgt_seq, str)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

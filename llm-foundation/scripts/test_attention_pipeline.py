#!/usr/bin/env python3
"""
Quick test to verify attention pipeline works
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_attention_imports():
    """Test if all attention modules can be imported"""
    print("🧪 Testing Attention Module Imports...")
    
    try:
        from attention.basic_attention.attention import BasicAttention
        print("✅ BasicAttention imported successfully")
        
        from attention.self_attention.self_attention import SelfAttention
        print("✅ SelfAttention imported successfully")
        
        from attention.multi_head_attention.multi_head_attention import MultiHeadAttention
        print("✅ MultiHeadAttention imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False

def test_basic_attention():
    """Test basic attention mechanism"""
    print("\n🧪 Testing Basic Attention...")
    
    try:
        from attention.basic_attention.attention import BasicAttention
        
        # Create attention mechanism
        attention = BasicAttention(query_dim=5, key_dim=5, value_dim=6)
        
        # Create test data
        queries = np.random.randn(3, 5)
        keys = np.random.randn(4, 5)
        values = np.random.randn(4, 6)
        
        # Forward pass
        output, weights = attention.forward(queries, keys, values)
        
        print(f"✅ Basic attention works!")
        print(f"   Input shapes: Q{queries.shape}, K{keys.shape}, V{values.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Weights shape: {weights.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Basic attention error: {e}")
        return False

def test_self_attention():
    """Test self-attention mechanism"""
    print("\n🧪 Testing Self-Attention...")
    
    try:
        from attention.self_attention.self_attention import SelfAttention
        
        # Create self-attention mechanism
        self_attn = SelfAttention(input_dim=8, num_heads=2)
        
        # Create test data
        x = np.random.randn(5, 8)
        
        # Forward pass
        output, weights = self_attn.forward(x)
        
        print(f"✅ Self-attention works!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Weights shape: {weights.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Self-attention error: {e}")
        return False

def test_multi_head_attention():
    """Test multi-head attention mechanism"""
    print("\n🧪 Testing Multi-Head Attention...")
    
    try:
        from attention.multi_head_attention.multi_head_attention import MultiHeadAttention
        
        # Create multi-head attention mechanism
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        
        # Create test data
        queries = np.random.randn(2, 4, 8)  # batch_size=2, seq_len=4, d_model=8
        keys = np.random.randn(2, 4, 8)
        values = np.random.randn(2, 4, 8)
        
        # Forward pass
        output, weights = mha.forward(queries, keys, values)
        
        print(f"✅ Multi-head attention works!")
        print(f"   Input shapes: Q{queries.shape}, K{keys.shape}, V{values.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Weights shape: {weights.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Multi-head attention error: {e}")
        return False

def main():
    print("🚀 Testing Attention Pipeline\n")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_attention_imports,
        test_basic_attention,
        test_self_attention,
        test_multi_head_attention
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Attention pipeline is ready to run!")
        print("\n🚀 You can now run:")
        print("   python run_attention_pipeline.py")
        print("   python src/attention/examples/attention_demo.py")
    else:
        print("❌ Some tests failed. Let's debug the issues.")
    
    return passed == total

if __name__ == "__main__":
    main() 
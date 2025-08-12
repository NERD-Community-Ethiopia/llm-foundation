#!/usr/bin/env python3
"""
Simple test for NetworkX visualization functionality
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_networkx_import():
    """Test if NetworkX can be imported"""
    try:
        import networkx as nx
        print("✅ NetworkX imported successfully")
        print(f"   Version: {nx.__version__}")
        return True
    except ImportError as e:
        print(f"❌ NetworkX import failed: {e}")
        return False

def test_visualizer_import():
    """Test if NetworkXVisualizer can be imported"""
    try:
        from neural_nets import NetworkXVisualizer, FeedforwardNeuralNetwork
        # Test that we can actually use the imported classes
        _ = NetworkXVisualizer
        _ = FeedforwardNeuralNetwork
        print("✅ NetworkXVisualizer imported successfully")
        return True
    except ImportError as e:
        print(f"❌ NetworkXVisualizer import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic NetworkX visualization functionality"""
    try:
        from neural_nets import NetworkXVisualizer, FeedforwardNeuralNetwork
        
        # Create a simple network
        network = FeedforwardNeuralNetwork([2, 3, 1], activation='sigmoid')
        
        # Create visualizer
        visualizer = NetworkXVisualizer(network)
        
        # Test network stats
        stats = visualizer.get_network_stats()
        print(f"✅ Network stats: {stats['num_layers']} layers, {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        # Test forward pass simulation
        X = np.array([[0.5], [0.8]])
        node_values = visualizer.simulate_forward_pass(X)
        print(f"✅ Forward pass simulation: {len(node_values)} node values computed")
        
        # Test prediction
        prediction = network.predict(X)[0, 0]
        print(f"✅ Network prediction: {prediction:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 NetworkX Visualization Test")
    print("=" * 40)
    
    tests = [
        ("NetworkX Import", test_networkx_import),
        ("Visualizer Import", test_visualizer_import),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! NetworkX visualization is ready to use.")
        print("\n🚀 You can now run:")
        print("   python simple_networkx_example.py")
        print("   python demo_networkx_visualization.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main()) 
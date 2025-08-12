#!/usr/bin/env python3
"""
Test NetworkX visualization with proper backend handling
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_visualization_with_save():
    """Test visualization by saving to files instead of showing"""
    try:
        from neural_nets import NetworkXVisualizer, FeedforwardNeuralNetwork
        
        print("🧠 Testing NetworkX Visualization with File Output")
        print("=" * 50)
        
        # Create a simple network
        network = FeedforwardNeuralNetwork([2, 4, 1], activation='sigmoid')
        visualizer = NetworkXVisualizer(network)
        
        # Print network info
        visualizer.print_network_info()
        
        # Create test input
        X = np.array([[0.5], [0.8]])
        node_values = visualizer.simulate_forward_pass(X)
        
        print(f"\n📈 Forward Pass Results:")
        for node, value in node_values.items():
            print(f"  {node}: {value:.4f}")
        
        # Test XOR inputs
        XOR_inputs = [
            (np.array([[0], [0]]), "XOR_00"),
            (np.array([[0], [1]]), "XOR_01"),
            (np.array([[1], [0]]), "XOR_10"),
            (np.array([[1], [1]]), "XOR_11")
        ]
        
        print(f"\n🎨 Creating visualizations...")
        
        # Create visualizations for each XOR input
        for X_test, label in XOR_inputs:
            node_values = visualizer.simulate_forward_pass(X_test)
            prediction = network.predict(X_test)[0, 0]
            
            print(f"  Creating visualization for {label} (prediction: {prediction:.4f})")
            
            # Save visualization
            plt.figure(figsize=(10, 6))
            visualizer.visualize_network(
                node_values, 
                title=f"XOR Network - {label} (Pred: {prediction:.4f})",
                show_weights=True
            )
            plt.savefig(f'network_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create architecture visualization
        plt.figure(figsize=(10, 6))
        visualizer.visualize_network(title="Neural Network Architecture")
        plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ All visualizations saved successfully!")
        print(f"   Files created:")
        print(f"   - network_XOR_00.png")
        print(f"   - network_XOR_01.png") 
        print(f"   - network_XOR_10.png")
        print(f"   - network_XOR_11.png")
        print(f"   - network_architecture.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_network_stats():
    """Test network statistics and analysis"""
    try:
        from neural_nets import NetworkXVisualizer, FeedforwardNeuralNetwork
        
        print("\n📊 Testing Network Statistics")
        print("=" * 30)
        
        # Test different architectures
        architectures = [
            ([2, 2, 1], "Simple"),
            ([2, 4, 1], "XOR"),
            ([3, 5, 3, 1], "Deep"),
            ([2, 8, 8, 4, 1], "Very Deep")
        ]
        
        for arch, name in architectures:
            network = FeedforwardNeuralNetwork(arch, activation='sigmoid')
            visualizer = NetworkXVisualizer(network)
            stats = visualizer.get_network_stats()
            
            print(f"\n{name} Network ({arch}):")
            print(f"  Layers: {stats['num_layers']}")
            print(f"  Nodes: {stats['num_nodes']}")
            print(f"  Connections: {stats['num_edges']}")
            print(f"  Parameters: {stats['total_parameters']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 NetworkX Visualization Backend Test")
    print("=" * 50)
    
    tests = [
        ("Visualization with File Output", test_visualization_with_save),
        ("Network Statistics", test_network_stats),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! NetworkX visualization is working correctly.")
        print("\n💡 The visualizations are saved as PNG files instead of displayed interactively.")
        print("   This is normal for non-interactive environments.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main()) 
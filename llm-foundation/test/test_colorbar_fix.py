#!/usr/bin/env python3
"""
Test the colorbar fix for NetworkX visualization
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_colorbar_fix():
    """Test that the colorbar works without errors"""
    try:
        from neural_nets import NetworkXVisualizer, FeedforwardNeuralNetwork
        
        # Create a simple network
        network = FeedforwardNeuralNetwork([2, 3, 1], activation='sigmoid')
        visualizer = NetworkXVisualizer(network)
        
        # Create test input
        X = np.array([[0.5], [0.8]])
        node_values = visualizer.simulate_forward_pass(X)
        
        print("✅ Testing colorbar fix...")
        print("   This should open a matplotlib window with a working colorbar")
        
        # Test visualization with colorbar
        visualizer.visualize_network(
            node_values, 
            title="Test: Network with Colorbar",
            show_weights=True
        )
        
        print("✅ Colorbar test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Colorbar test failed: {e}")
        return False

if __name__ == "__main__":
    test_colorbar_fix() 
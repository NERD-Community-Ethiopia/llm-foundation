#!/usr/bin/env python3
"""
Demo script for NetworkX Neural Network Visualization
Run this to see neural networks visualized as graphs
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from neural_nets import FeedforwardNeuralNetwork, NetworkXVisualizer
from neural_nets.examples.simple_examples import xor_example

def demo_basic_visualization():
    """Demo basic network visualization"""
    print("🔍 Basic NetworkX Visualization Demo")
    print("=" * 50)
    
    # Create a simple network
    network = FeedforwardNeuralNetwork([2, 4, 3, 1], activation='sigmoid')
    
    # Create visualizer
    visualizer = NetworkXVisualizer(network)
    
    # Print network info
    visualizer.print_network_info()
    
    # Create sample input
    X = np.array([[0.5, 0.8]]).T
    
    # Simulate forward pass
    node_values = visualizer.simulate_forward_pass(X)
    
    print(f"\n📈 Forward Pass Results:")
    for node, value in node_values.items():
        print(f"  {node}: {value:.4f}")
    
    # Visualize network with activation values
    print("\n🎨 Visualizing network with activation values...")
    visualizer.visualize_network(node_values, 
                                title="Neural Network with Activation Values",
                                show_weights=True)
    
    # Visualize architecture only
    print("\n🏗️ Visualizing network architecture...")
    visualizer.visualize_network(title="Neural Network Architecture")
    
    return visualizer, network

def demo_xor_visualization():
    """Demo visualization with XOR network"""
    print("\n🔍 XOR Network Visualization Demo")
    print("=" * 50)
    
    # Create XOR data
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    # Create network: 2 input -> 4 hidden -> 1 output
    network = FeedforwardNeuralNetwork([2, 4, 1], activation='sigmoid')
    
    # Create visualizer
    visualizer = NetworkXVisualizer(network)
    
    # Test different inputs
    test_inputs = [
        (np.array([[0], [0]]), "XOR(0,0)"),
        (np.array([[0], [1]]), "XOR(0,1)"),
        (np.array([[1], [0]]), "XOR(1,0)"),
        (np.array([[1], [1]]), "XOR(1,1)")
    ]
    
    for X_test, label in test_inputs:
        print(f"\n📊 {label}:")
        
        # Get predictions
        prediction = network.predict(X_test)[0, 0]
        print(f"  Prediction: {prediction:.4f}")
        
        # Simulate forward pass
        node_values = visualizer.simulate_forward_pass(X_test)
        
        # Show activation values for key nodes
        print(f"  Input neurons: {node_values['L0_N0']:.4f}, {node_values['L0_N1']:.4f}")
        print(f"  Hidden neurons: {[node_values[f'L1_N{i}'] for i in range(4)]}")
        print(f"  Output neuron: {node_values['L2_N0']:.4f}")
    
    # Visualize with first input
    X_first = test_inputs[0][0]
    node_values = visualizer.simulate_forward_pass(X_first)
    visualizer.visualize_network(node_values, 
                                title=f"XOR Network - {test_inputs[0][1]}",
                                show_weights=True)
    
    return visualizer, network

def demo_animation():
    """Demo animated forward pass"""
    print("\n🎬 Animated Forward Pass Demo")
    print("=" * 50)
    
    # Create a network
    network = FeedforwardNeuralNetwork([2, 3, 2, 1], activation='sigmoid')
    visualizer = NetworkXVisualizer(network)
    
    # Create input
    X = np.array([[0.7], [0.3]])
    
    print("Creating animation... (this may take a moment)")
    print("The animation will show how activations flow through each layer")
    
    # Create animation
    anim = visualizer.animate_forward_pass(X, interval=1500)
    
    return visualizer, network, anim

def main():
    """Run all NetworkX visualization demos"""
    print("🧠 NetworkX Neural Network Visualization Demo")
    print("=" * 60)
    
    try:
        # # Basic visualization
        # visualizer1, network1 = demo_basic_visualization()
        
        # XOR visualization
        visualizer2, network2 = demo_xor_visualization()
        
        # Animation demo
        visualizer3, network3, anim = demo_animation()
        
        print("\n✅ All NetworkX visualization demos completed!")
        print("\n💡 Key Features Demonstrated:")
        print("  • Graph representation of neural networks")
        print("  • Node-based visualization of neurons")
        print("  • Edge-based visualization of connections")
        print("  • Activation value coloring")
        print("  • Weight visualization")
        print("  • Animated forward pass")
        
    except Exception as e:
        print(f"❌ Error running demos: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
#!/usr/bin/env python3
"""
Simple example of using NetworkX to visualize neural networks
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from neural_nets import FeedforwardNeuralNetwork, NetworkXVisualizer

def main():
    """Simple NetworkX neural network visualization example"""
    print("🧠 Simple NetworkX Neural Network Example")
    print("=" * 50)
    
    # 1. Create a neural network
    print("1. Creating neural network...")
    network = FeedforwardNeuralNetwork([2, 3, 1], activation='sigmoid')
    
    # 2. Create NetworkX visualizer
    print("2. Creating NetworkX visualizer...")
    visualizer = NetworkXVisualizer(network)
    
    # 3. Print network information
    print("3. Network information:")
    visualizer.print_network_info()
    
    # 4. Create test input
    print("4. Creating test input...")
    X = np.array([[0.5], [0.8]])  # 2 input features
    print(f"   Input: [{X[0,0]:.1f}, {X[1,0]:.1f}]")
    
    # 5. Simulate forward pass
    print("5. Simulating forward pass...")
    node_values = visualizer.simulate_forward_pass(X)
    
    # 6. Show results
    print("6. Forward pass results:")
    for node, value in node_values.items():
        print(f"   {node}: {value:.4f}")
    
    # 7. Get prediction
    prediction = network.predict(X)[0, 0]
    print(f"   Final prediction: {prediction:.4f}")
    
    # 8. Visualize the network
    print("7. Visualizing network...")
    print("   (This will open a matplotlib window)")
    
    # Visualize with activation values
    visualizer.visualize_network(
        node_values, 
        title="Neural Network with Activation Values",
        show_weights=True
    )
    
    print("\n✅ Example completed!")
    print("\n💡 What you learned:")
    print("   • How to represent neural networks as graphs")
    print("   • How neurons become nodes in the graph")
    print("   • How connections become edges in the graph")
    print("   • How to visualize activation values")
    print("   • How to see the flow of information through the network")

if __name__ == "__main__":
    main() 
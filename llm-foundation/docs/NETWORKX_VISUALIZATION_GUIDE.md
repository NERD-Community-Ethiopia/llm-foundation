# 🧠 NetworkX Neural Network Visualization Guide

This guide shows you how to simulate neural network forward passes using NetworkX graphs, where **neurons become nodes** and **connections become edges**.

## 🎯 Overview

NetworkX is a powerful Python library for creating, manipulating, and visualizing complex networks. By representing neural networks as graphs, you can:

- **Visualize network architecture** with nodes and edges
- **Simulate forward passes** by tracking activation values
- **Animate information flow** through the network
- **Analyze network structure** and connectivity

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install networkx matplotlib numpy
```

### 2. Basic Usage

```python
from neural_nets import FeedforwardNeuralNetwork, NetworkXVisualizer
import numpy as np

# Create a neural network
network = FeedforwardNeuralNetwork([2, 3, 1], activation='sigmoid')

# Create visualizer
visualizer = NetworkXVisualizer(network)

# Simulate forward pass
X = np.array([[0.5], [0.8]])
node_values = visualizer.simulate_forward_pass(X)

# Visualize
visualizer.visualize_network(node_values, show_weights=True)
```

## 📊 How It Works

### Graph Representation

```
Neural Network:          NetworkX Graph:
┌─────────┐             ┌─────────┐
│ Input   │             │ Node    │
│ Layer   │    →        │ (L0_N0) │
└─────────┘             └─────────┘
     │                        │
     ▼                        ▼
┌─────────┐             ┌─────────┐
│ Hidden  │             │ Edge    │
│ Layer   │    →        │ (weight)│
└─────────┘             └─────────┘
     │                        │
     ▼                        ▼
┌─────────┐             ┌─────────┐
│ Output  │             │ Node    │
│ Layer   │    →        │ (L1_N0) │
└─────────┘             └─────────┘
```

### Node Naming Convention

- **Input neurons**: `L0_N0`, `L0_N1`, `L0_N2`, ...
- **Hidden neurons**: `L1_N0`, `L1_N1`, `L1_N2`, ...
- **Output neurons**: `L2_N0`, `L2_N1`, `L2_N2`, ...

Where:
- `L{layer_index}_N{neuron_index}`
- Layer 0 = Input layer
- Layer 1 = First hidden layer
- Layer 2 = Output layer

## 🎨 Visualization Features

### 1. Basic Network Architecture

```python
# Show network structure without activation values
visualizer.visualize_network(title="Neural Network Architecture")
```

### 2. Activation Value Visualization

```python
# Color nodes based on their activation values
node_values = visualizer.simulate_forward_pass(X)
visualizer.visualize_network(node_values, 
                            title="Network with Activation Values")
```

### 3. Weight Visualization

```python
# Show connection weights on edges
visualizer.visualize_network(node_values, 
                            show_weights=True,
                            title="Network with Weights")
```

### 4. Animated Forward Pass

```python
# Animate how activations flow through layers
anim = visualizer.animate_forward_pass(X, interval=1000)
```

## 🔍 Detailed Examples

### Example 1: XOR Network Visualization

```python
# Create XOR network
network = FeedforwardNeuralNetwork([2, 4, 1], activation='sigmoid')
visualizer = NetworkXVisualizer(network)

# Test XOR inputs
XOR_inputs = [
    (np.array([[0], [0]]), "XOR(0,0)"),
    (np.array([[0], [1]]), "XOR(0,1)"),
    (np.array([[1], [0]]), "XOR(1,0)"),
    (np.array([[1], [1]]), "XOR(1,1)")
]

for X, label in XOR_inputs:
    node_values = visualizer.simulate_forward_pass(X)
    prediction = network.predict(X)[0, 0]
    
    print(f"{label}: Prediction = {prediction:.4f}")
    print(f"  Input activations: {node_values['L0_N0']:.4f}, {node_values['L0_N1']:.4f}")
    print(f"  Hidden activations: {[node_values[f'L1_N{i}'] for i in range(4)]}")
    print(f"  Output activation: {node_values['L2_N0']:.4f}")
```

### Example 2: Multi-Layer Network

```python
# Create deeper network
network = FeedforwardNeuralNetwork([3, 5, 4, 2, 1], activation='sigmoid')
visualizer = NetworkXVisualizer(network)

# Print network statistics
stats = visualizer.get_network_stats()
print(f"Layers: {stats['num_layers']}")
print(f"Nodes: {stats['num_nodes']}")
print(f"Connections: {stats['num_edges']}")
print(f"Parameters: {stats['total_parameters']:,}")

# Visualize with different color schemes
node_values = visualizer.simulate_forward_pass(X)
visualizer.visualize_network(node_values, color_map='plasma')
```

## 🛠️ Advanced Features

### Custom Node Positioning

```python
# The visualizer automatically positions nodes in layers
# You can access positions:
positions = visualizer.node_positions
print(f"Node L0_N0 position: {positions['L0_N0']}")
```

### Network Analysis

```python
# Get network statistics
stats = visualizer.get_network_stats()

# Analyze connectivity
G = visualizer.G
print(f"Network density: {nx.density(G):.4f}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt

# Create custom visualization
fig, ax = plt.subplots(figsize=(12, 8))
pos = visualizer.node_positions

# Draw with custom styling
nx.draw(visualizer.G, pos, 
        node_color='lightblue',
        node_size=800,
        font_size=10,
        font_weight='bold',
        arrows=True,
        edge_color='gray',
        width=2,
        alpha=0.8,
        ax=ax)

plt.title("Custom Network Visualization")
plt.show()
```

## 🎬 Animation Examples

### Forward Pass Animation

```python
# Create animation showing how activations flow
X = np.array([[0.7], [0.3]])
anim = visualizer.animate_forward_pass(X, interval=1500)

# Save animation
anim.save('forward_pass.gif', writer='pillow')
```

### Training Progress Animation

```python
# Animate training progress
for epoch in range(100):
    # Train one step
    trainer._train_step(X, y)
    
    # Visualize current state
    node_values = visualizer.simulate_forward_pass(X)
    visualizer.visualize_network(node_values, 
                                title=f"Training Epoch {epoch}")
    plt.pause(0.1)
```

## 🔧 Integration with Existing Code

### With Training

```python
from neural_nets import FeedforwardNeuralNetwork, NeuralNetworkTrainer, NetworkXVisualizer

# Create and train network
network = FeedforwardNeuralNetwork([2, 4, 1], activation='sigmoid')
trainer = NeuralNetworkTrainer(network, learning_rate=0.1)
history = trainer.train(X, y, epochs=1000)

# Visualize trained network
visualizer = NetworkXVisualizer(network)
node_values = visualizer.simulate_forward_pass(X)
visualizer.visualize_network(node_values, title="Trained Network")
```

### With Different Architectures

```python
# Test different network architectures
architectures = [
    [2, 2, 1],      # Simple
    [2, 4, 1],      # XOR
    [3, 5, 3, 1],   # Deep
    [2, 8, 8, 4, 1] # Very deep
]

for arch in architectures:
    network = FeedforwardNeuralNetwork(arch, activation='sigmoid')
    visualizer = NetworkXVisualizer(network)
    
    print(f"\nArchitecture: {arch}")
    visualizer.print_network_info()
    
    # Visualize
    X = np.random.rand(arch[0], 1)
    node_values = visualizer.simulate_forward_pass(X)
    visualizer.visualize_network(node_values, 
                                title=f"Architecture: {arch}")
```

## 🎯 Educational Benefits

Using NetworkX for neural network visualization helps you understand:

1. **Network Structure**: How layers and neurons are connected
2. **Information Flow**: How data moves through the network
3. **Activation Patterns**: How different inputs activate neurons
4. **Weight Importance**: Which connections are stronger/weaker
5. **Layer Interactions**: How each layer transforms the input

## 🚀 Running the Examples

```bash
# Simple example
python simple_networkx_example.py

# Full demo
python demo_networkx_visualization.py

# With existing neural network demos
python demo_neural_nets.py
```

## 📚 Further Reading

- [NetworkX Documentation](https://networkx.org/)
- [Matplotlib Animation](https://matplotlib.org/stable/api/animation_api.html)
- [Neural Network Visualization Techniques](https://distill.pub/2017/feature-visualization/)

---

**💡 Tip**: The NetworkX visualization is perfect for educational purposes and debugging neural network behavior. It makes the abstract concept of neural networks concrete and visual! 
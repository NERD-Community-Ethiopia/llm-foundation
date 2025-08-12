"""
Neural Network Visualization using NetworkX
Simulates forward pass using graph representation
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import matplotlib.animation as animation
from ..basic.feedforward import FeedforwardNeuralNetwork

class NetworkXVisualizer:
    """
    Visualizes neural networks using NetworkX graphs
    """
    
    def __init__(self, network: FeedforwardNeuralNetwork):
        """
        Initialize visualizer with a neural network
        
        Args:
            network: FeedforwardNeuralNetwork instance
        """
        self.network = network
        self.G = nx.DiGraph()
        self.node_positions = {}
        self.node_values = {}
        self.edge_weights = {}
        self._build_graph()
    
    def _build_graph(self):
        """Build the NetworkX graph from the neural network"""
        node_id = 0
        
        # Add nodes for each layer
        for layer_idx, layer_size in enumerate(self.network.layer_sizes):
            layer_nodes = []
            
            for neuron_idx in range(layer_size):
                node_name = f"L{layer_idx}_N{neuron_idx}"
                layer_nodes.append(node_name)
                
                # Store node position for visualization
                self.node_positions[node_name] = (layer_idx, -neuron_idx + layer_size/2)
                
                # Add node attributes
                self.G.add_node(node_name, 
                               layer=layer_idx,
                               neuron=neuron_idx,
                               layer_size=layer_size)
            
            # Add edges between layers
            if layer_idx < len(self.network.layer_sizes) - 1:
                weight_matrix = self.network.weights[layer_idx]
                
                for from_neuron in range(layer_size):
                    for to_neuron in range(self.network.layer_sizes[layer_idx + 1]):
                        from_node = f"L{layer_idx}_N{from_neuron}"
                        to_node = f"L{layer_idx + 1}_N{to_neuron}"
                        
                        weight = weight_matrix[to_neuron, from_neuron]
                        self.G.add_edge(from_node, to_node, weight=weight)
                        self.edge_weights[(from_node, to_node)] = weight
    
    def simulate_forward_pass(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, float]:
        """
        Simulate forward pass and store node values
        
        Args:
            X: Input data
            sample_idx: Index of sample to visualize
            
        Returns:
            Dictionary mapping node names to their activation values
        """
        # Get activations from the network
        activations, _ = self.network.forward(X)
        
        # Store node values
        self.node_values = {}
        
        for layer_idx, activation in enumerate(activations):
            for neuron_idx in range(activation.shape[0]):
                node_name = f"L{layer_idx}_N{neuron_idx}"
                value = activation[neuron_idx, sample_idx]
                self.node_values[node_name] = value
        
        return self.node_values
    
    def visualize_network(self, 
                         node_values: Optional[Dict[str, float]] = None,
                         title: str = "Neural Network Architecture",
                         figsize: Tuple[int, int] = (12, 8),
                         node_size: int = 1000,
                         font_size: int = 8,
                         show_weights: bool = False,
                         color_map: str = 'viridis'):
        """
        Visualize the neural network
        
        Args:
            node_values: Dictionary of node values for coloring
            title: Plot title
            figsize: Figure size
            node_size: Size of nodes
            font_size: Font size for labels
            show_weights: Whether to show edge weights
            color_map: Color map for node values
        """
        plt.figure(figsize=figsize)
        
        # Set up the layout
        pos = self.node_positions
        
        # Determine node colors
        if node_values:
            # Color nodes based on their values
            node_colors = [node_values.get(node, 0) for node in self.G.nodes()]
            vmin, vmax = min(node_colors), max(node_colors)
        else:
            # Color nodes by layer
            node_colors = [self.G.nodes[node]['layer'] for node in self.G.nodes()]
            vmin, vmax = 0, len(self.network.layer_sizes) - 1
        
        # Draw the network
        nx.draw(self.G, pos, 
                node_color=node_colors,
                cmap=plt.cm.get_cmap(color_map),
                vmin=vmin, vmax=vmax,
                node_size=node_size,
                font_size=font_size,
                font_color='white',
                font_weight='bold',
                arrows=True,
                edge_color='gray',
                width=1,
                alpha=0.7)
        
        # Add edge labels if requested
        if show_weights:
            edge_labels = {(u, v): f'{d["weight"]:.2f}' 
                          for u, v, d in self.G.edges(data=True)}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels, font_size=6)
        
        # Add layer labels
        for layer_idx in range(len(self.network.layer_sizes)):
            layer_name = ['Input', 'Hidden', 'Output'][min(layer_idx, 2)]
            if layer_idx > 2:
                layer_name = f'Hidden {layer_idx}'
            plt.text(layer_idx, self.network.layer_sizes[layer_idx]/2 + 0.5, 
                    layer_name, ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Create colorbar with proper axes context
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(color_map), norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Activation Value' if node_values else 'Layer')
        
        plt.axis('off')
        
        # Use constrained_layout instead of tight_layout for better compatibility
        plt.subplots_adjust(right=0.85)  # Make room for colorbar
        
        # Try to show the plot, but handle non-interactive backends gracefully
        try:
            plt.show()
        except Exception as e:
            print(f"Note: Could not display plot interactively: {e}")
            print("   The visualization is working correctly, but you may need to save it to a file.")
            # Save the plot instead
            plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
            print("   Plot saved as 'network_visualization.png'")
        
        plt.close()  # Clean up the figure
    
    def animate_forward_pass(self, X: np.ndarray, 
                           sample_idx: int = 0,
                           interval: int = 1000,
                           save_path: Optional[str] = None):
        """
        Animate the forward pass through the network
        
        Args:
            X: Input data
            sample_idx: Index of sample to visualize
            interval: Time between frames (ms)
            save_path: Path to save animation (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            ax.clear()
            
            # Simulate forward pass up to current layer
            activations, _ = self.network.forward(X)
            
            # Store node values up to current layer
            node_values = {}
            for layer_idx in range(min(frame + 1, len(activations))):
                activation = activations[layer_idx]
                for neuron_idx in range(activation.shape[0]):
                    node_name = f"L{layer_idx}_N{neuron_idx}"
                    value = activation[neuron_idx, sample_idx]
                    node_values[node_name] = value
            
            # Color nodes based on their values
            node_colors = [node_values.get(node, 0) for node in self.G.nodes()]
            
            # Draw the network
            nx.draw(self.G, self.node_positions, 
                    node_color=node_colors,
                    cmap=plt.cm.viridis,
                    node_size=1000,
                    font_size=8,
                    font_color='white',
                    font_weight='bold',
                    arrows=True,
                    edge_color='gray',
                    width=1,
                    alpha=0.7,
                    ax=ax)
            
            ax.set_title(f'Forward Pass - Layer {frame}', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, 
                                     frames=len(self.network.layer_sizes),
                                     interval=interval, 
                                     repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        # Try to show the animation, but handle non-interactive backends gracefully
        try:
            plt.show()
        except Exception as e:
            print(f"Note: Could not display animation interactively: {e}")
            print("   The animation is working correctly, but you may need to save it to a file.")
            # Save the animation instead
            anim.save('forward_pass_animation.gif', writer='pillow')
            print("   Animation saved as 'forward_pass_animation.gif'")
        
        return anim
    
    def get_network_stats(self) -> Dict:
        """Get statistics about the network"""
        return {
            'num_layers': len(self.network.layer_sizes),
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'layer_sizes': self.network.layer_sizes,
            'total_parameters': sum(w.size + b.size for w, b in zip(self.network.weights, self.network.biases))
        }
    
    def print_network_info(self):
        """Print detailed network information"""
        stats = self.get_network_stats()
        
        print("🧠 Neural Network Architecture")
        print("=" * 40)
        print(f"Layers: {stats['num_layers']}")
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Connections: {stats['num_edges']}")
        print(f"Parameters: {stats['total_parameters']:,}")
        print(f"Layer sizes: {stats['layer_sizes']}")
        
        print("\n📊 Layer Details:")
        for layer_idx, size in enumerate(stats['layer_sizes']):
            layer_name = ['Input', 'Hidden', 'Output'][min(layer_idx, 2)]
            if layer_idx > 2:
                layer_name = f'Hidden {layer_idx}'
            print(f"  {layer_name}: {size} neurons")


def demo_networkx_visualization():
    """Demo function showing NetworkX visualization"""
    print("🔍 NetworkX Neural Network Visualization Demo")
    
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
    
    # Visualize network
    visualizer.visualize_network(node_values, 
                                title="Neural Network with Activation Values",
                                show_weights=True)
    
    # Visualize without values (architecture only)
    visualizer.visualize_network(title="Neural Network Architecture")
    
    return visualizer, network


if __name__ == "__main__":
    demo_networkx_visualization() 
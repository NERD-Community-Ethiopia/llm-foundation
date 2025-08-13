"""
Neural Network Training Module
"""
import numpy as np
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt

class NeuralNetworkTrainer:
    """
    Trainer class for neural networks
    """
    
    def __init__(self, network, learning_rate: float = 0.01):
        """
        Initialize trainer
        
        Args:
            network: Neural network instance
            learning_rate: Learning rate for gradient descent
        """
        self.network = network
        self.learning_rate = learning_rate
        self.training_history = {
            'loss': [],
            'accuracy': []
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        loss_type: str = 'mse',
        verbose: bool = True,
        early_stopping: bool = False,
        patience: int = 10
    ) -> dict:
        """
        Train the neural network
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            loss_type: Type of loss function
            verbose: Whether to print training progress
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Mini-batch training
            if batch_size and batch_size < X_train.shape[1]:
                indices = np.random.permutation(X_train.shape[1])
                for i in range(0, X_train.shape[1], batch_size):
                    batch_indices = indices[i:i + batch_size]
                    X_batch = X_train[:, batch_indices]
                    y_batch = y_train[:, batch_indices]
                    
                    self._train_step(X_batch, y_batch, loss_type)
            else:
                self._train_step(X_train, y_train, loss_type)
            
            # Compute training loss
            train_loss = self._compute_loss(X_train, y_train, loss_type)
            self.training_history['loss'].append(train_loss)
            
            # Compute validation loss if validation data provided
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val, loss_type)
                
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            
            # Print progress
            if verbose and epoch % 100 == 0:
                val_info = f", Val Loss: {val_loss:.6f}" if X_val is not None else ""
                print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}{val_info}")
        
        return self.training_history
    
    def _train_step(self, X: np.ndarray, y: np.ndarray, loss_type: str):
        """Single training step"""
        # Forward pass
        activations, z_values = self.network.forward(X)
        
        # Backward pass
        weight_gradients, bias_gradients = self.network.backward_pass(
            activations, z_values, self.network.weights, y, self.network, loss_type
        )
        
        # Update weights and biases
        for i in range(len(self.network.weights)):
            self.network.weights[i] -= self.learning_rate * weight_gradients[i]
            self.network.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray, loss_type: str) -> float:
        """Compute loss for given data"""
        y_pred = self.network.predict(X)
        return self.network.compute_loss(y, y_pred, loss_type)
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if self.training_history['accuracy']:
            plt.subplot(1, 2, 2)
            plt.plot(self.training_history['accuracy'])
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

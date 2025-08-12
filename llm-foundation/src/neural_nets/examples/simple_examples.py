"""
Simple Training Examples for Neural Networks
"""
import numpy as np
import matplotlib.pyplot as plt
from src.neural_nets.basic.feedforward import FeedforwardNeuralNetwork
from src.neural_nets.training.trainer import NeuralNetworkTrainer

def xor_example():
    """
    Train a neural network to learn XOR function
    """
    print("🔍 Training Neural Network on XOR Problem")
    
    # XOR data
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    # Create network: 2 input -> 4 hidden -> 1 output
    network = FeedforwardNeuralNetwork([2, 4, 1], activation='sigmoid')
    
    # Create trainer
    trainer = NeuralNetworkTrainer(network, learning_rate=0.1)
    
    # Train the network
    history = trainer.train(X, y, epochs=5000, verbose=True)
    
    # Test predictions
    predictions = network.predict(X)
    print("\n📊 XOR Results:")
    print("Input\t\tPredicted\tExpected")
    print("-" * 40)
    for i in range(X.shape[1]):
        print(f"({X[0, i]}, {X[1, i]})\t\t{predictions[0, i]:.4f}\t\t{y[0, i]}")
    
    # Plot training history
    trainer.plot_training_history()
    
    return network, trainer

def linear_regression_example():
    """
    Train a neural network for linear regression
    """
    print("\n🔍 Training Neural Network for Linear Regression")
    
    # Generate synthetic data: y = 2x + 1 + noise
    np.random.seed(42)
    X = np.random.rand(1, 100) * 10
    y = 2 * X + 1 + np.random.normal(0, 0.1, (1, 100))
    
    # Create network: 1 input -> 1 output (linear)
    network = FeedforwardNeuralNetwork([1, 1], activation='linear')
    
    # Create trainer
    trainer = NeuralNetworkTrainer(network, learning_rate=0.01)
    
    # Train the network
    history = trainer.train(X, y, epochs=1000, verbose=True)
    
    # Test predictions
    X_test = np.array([[0, 2, 4, 6, 8, 10]])
    predictions = network.predict(X_test)
    
    print("\n📊 Linear Regression Results:")
    print("Input\tPredicted\tExpected (2x + 1)")
    print("-" * 40)
    for i in range(X_test.shape[1]):
        expected = 2 * X_test[0, i] + 1
        print(f"{X_test[0, i]:.1f}\t{predictions[0, i]:.4f}\t\t{expected:.1f}")
    
    # Plot results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[0], y[0], alpha=0.6, label='Training Data')
    plt.plot(X_test[0], predictions[0], 'r-', linewidth=2, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return network, trainer

def classification_example():
    """
    Train a neural network for binary classification
    """
    print("\n🔍 Training Neural Network for Binary Classification")
    
    # Generate synthetic data for classification
    np.random.seed(42)
    n_samples = 200
    
    # Class 0: centered at (0, 0)
    class0_x = np.random.normal(0, 1, n_samples // 2)
    class0_y = np.random.normal(0, 1, n_samples // 2)
    
    # Class 1: centered at (3, 3)
    class1_x = np.random.normal(3, 1, n_samples // 2)
    class1_y = np.random.normal(3, 1, n_samples // 2)
    
    # Combine data
    X = np.vstack([np.hstack([class0_x, class1_x]),
                   np.hstack([class0_y, class1_y])])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).reshape(1, -1)
    
    # Create network: 2 input -> 8 hidden -> 1 output
    network = FeedforwardNeuralNetwork([2, 8, 1], activation='sigmoid')
    
    # Create trainer
    trainer = NeuralNetworkTrainer(network, learning_rate=0.1)
    
    # Train the network
    history = trainer.train(X, y, epochs=2000, verbose=True)
    
    # Test predictions
    predictions = network.predict(X)
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y)
    
    print(f"\n📊 Classification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[0, :n_samples//2], X[1, :n_samples//2], 
                c='blue', alpha=0.6, label='Class 0')
    plt.scatter(X[0, n_samples//2:], X[1, n_samples//2:], 
                c='red', alpha=0.6, label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Classification Data')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return network, trainer

if __name__ == "__main__":
    # Run all examples
    print("🚀 Running Neural Network Examples\n")
    
    # XOR example
    xor_network, xor_trainer = xor_example()
    
    # Linear regression example
    lr_network, lr_trainer = linear_regression_example()
    
    # Classification example
    clf_network, clf_trainer = classification_example()
    
    print("\n✅ All examples completed!")

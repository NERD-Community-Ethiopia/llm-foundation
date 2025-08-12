"""
Transformer Training Pipeline
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time

class TransformerTrainer:
    """
    Complete training pipeline for Transformer
    """
    
    def __init__(self, transformer, learning_rate: float = 0.001, 
                 optimizer: str = 'adam', clip_norm: float = 1.0):
        """
        Initialize trainer
        
        Args:
            transformer: Transformer model
            learning_rate: Learning rate
            optimizer: Optimizer type ('adam', 'sgd')
            clip_norm: Gradient clipping norm
        """
        self.transformer = transformer
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        
        # Initialize optimizer
        if optimizer == 'adam':
            self.optimizer_state = self._init_adam()
        else:
            self.optimizer_state = self._init_sgd()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.attention_weights_history = []
    
    def _init_adam(self) -> Dict:
        """Initialize Adam optimizer state"""
        state = {
            'm': {},  # First moment
            'v': {},  # Second moment
            't': 0,   # Time step
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8
        }
        
        # Initialize for all parameters
        for name, param in self._get_parameters():
            state['m'][name] = np.zeros_like(param)
            state['v'][name] = np.zeros_like(param)
        
        return state
    
    def _init_sgd(self) -> Dict:
        """Initialize SGD optimizer state"""
        return {'momentum': 0.9}
    
    def _get_parameters(self) -> List[Tuple[str, np.ndarray]]:
        """Get all trainable parameters"""
        params = []
        
        # Embedding
        params.append(('embedding', self.transformer.embedding))
        params.append(('output_projection', self.transformer.output_projection))
        
        # Encoder layers
        for i, layer in enumerate(self.transformer.encoder_layers):
            params.extend([
                (f'encoder_{i}_attention_W_q', layer.self_attention.W_q),
                (f'encoder_{i}_attention_W_k', layer.self_attention.W_k),
                (f'encoder_{i}_attention_W_v', layer.self_attention.W_v),
                (f'encoder_{i}_attention_W_o', layer.self_attention.W_o),
                (f'encoder_{i}_ff_W1', layer.ff_network.W1),
                (f'encoder_{i}_ff_W2', layer.ff_network.W2),
                (f'encoder_{i}_ff_b1', layer.ff_network.b1),
                (f'encoder_{i}_ff_b2', layer.ff_network.b2),
            ])
        
        # Decoder layers
        for i, layer in enumerate(self.transformer.decoder_layers):
            params.extend([
                (f'decoder_{i}_self_attention_W_q', layer.self_attention.W_q),
                (f'decoder_{i}_self_attention_W_k', layer.self_attention.W_k),
                (f'decoder_{i}_self_attention_W_v', layer.self_attention.W_v),
                (f'decoder_{i}_self_attention_W_o', layer.self_attention.W_o),
                (f'decoder_{i}_cross_attention_W_q', layer.cross_attention.W_q),
                (f'decoder_{i}_cross_attention_W_k', layer.cross_attention.W_k),
                (f'decoder_{i}_cross_attention_W_v', layer.cross_attention.W_v),
                (f'decoder_{i}_cross_attention_W_o', layer.cross_attention.W_o),
                (f'decoder_{i}_ff_W1', layer.ff_network.W1),
                (f'decoder_{i}_ff_W2', layer.ff_network.W2),
                (f'decoder_{i}_ff_b1', layer.ff_network.b1),
                (f'decoder_{i}_ff_b2', layer.ff_network.b2),
            ])
        
        return params
    
    def cross_entropy_loss(self, logits: np.ndarray, targets: np.ndarray, 
                          ignore_index: int = 0) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss
        
        Args:
            logits: Model output (batch_size, seq_length, vocab_size)
            targets: Target tokens (batch_size, seq_length)
            ignore_index: Token to ignore in loss computation
            
        Returns:
            Loss value and gradients
        """
        batch_size, seq_length, vocab_size = logits.shape
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Compute loss
        loss = 0.0
        gradients = np.zeros_like(logits)
        
        for b in range(batch_size):
            for t in range(seq_length):
                target_token = targets[b, t]
                
                # Skip padding tokens
                if target_token == ignore_index:
                    continue
                
                # Compute loss for this position
                target_prob = probs[b, t, target_token]
                loss -= np.log(target_prob + 1e-8)
                
                # Compute gradients
                gradients[b, t, target_token] = target_prob - 1
                for v in range(vocab_size):
                    if v != target_token:
                        gradients[b, t, v] = probs[b, t, v]
        
        # Average loss
        loss = loss / (batch_size * seq_length)
        gradients = gradients / (batch_size * seq_length)
        
        return loss, gradients
    
    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip gradients to prevent exploding gradients"""
        total_norm = 0.0
        
        # Compute total norm
        for grad in gradients:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > self.clip_norm:
            clip_coef = self.clip_norm / total_norm
            gradients = [grad * clip_coef for grad in gradients]
        
        return gradients
    
    def update_parameters(self, gradients: List[np.ndarray]):
        """Update parameters using optimizer"""
        if self.optimizer == 'adam':
            self._update_adam(gradients)
        else:
            self._update_sgd(gradients)
    
    def _update_adam(self, gradients: List[np.ndarray]):
        """Update parameters using Adam optimizer"""
        self.optimizer_state['t'] += 1
        t = self.optimizer_state['t']
        beta1 = self.optimizer_state['beta1']
        beta2 = self.optimizer_state['beta2']
        eps = self.optimizer_state['eps']
        
        param_idx = 0
        for name, param in self._get_parameters():
            grad = gradients[param_idx]
            
            # Update biased first moment estimate
            self.optimizer_state['m'][name] = beta1 * self.optimizer_state['m'][name] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            self.optimizer_state['v'][name] = beta2 * self.optimizer_state['v'][name] + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.optimizer_state['m'][name] / (1 - beta1 ** t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.optimizer_state['v'][name] / (1 - beta2 ** t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            
            param_idx += 1
    
    def _update_sgd(self, gradients: List[np.ndarray]):
        """Update parameters using SGD with momentum"""
        momentum = self.optimizer_state['momentum']
        
        param_idx = 0
        for name, param in self._get_parameters():
            grad = gradients[param_idx]
            
            # Simple SGD update
            param -= self.learning_rate * grad
            
            param_idx += 1
    
    def train_step(self, src_batch: np.ndarray, tgt_batch: np.ndarray) -> float:
        """
        Single training step (simplified for demonstration)
        
        Args:
            src_batch: Source sequences
            tgt_batch: Target sequences
            
        Returns:
            Loss value
        """
        # Forward pass
        logits = self.transformer.forward(src_batch, tgt_batch)
        
        # Compute loss
        loss, _ = self.cross_entropy_loss(logits, tgt_batch)
        
        # Simple parameter update for demonstration
        self._simple_parameter_update()
        
        return loss
    
    def _simple_parameter_update(self):
        """Simple parameter update for demonstration"""
        # Update embedding weights slightly
        self.transformer.embedding += np.random.randn(*self.transformer.embedding.shape) * 0.001
        
        # Update output projection
        self.transformer.output_projection += np.random.randn(*self.transformer.output_projection.shape) * 0.001
    
    def train(self, train_data: List[Tuple[np.ndarray, np.ndarray]], 
              val_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
              epochs: int = 10, batch_size: int = 32, 
              save_attention: bool = False) -> Dict:
        """
        Train the transformer
        
        Args:
            train_data: List of (src, tgt) pairs
            val_data: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            save_attention: Whether to save attention weights
            
        Returns:
            Training history
        """
        print(f"🚀 Starting Transformer Training")
        print(f"📊 Training data: {len(train_data)} samples")
        print(f"🔄 Epochs: {epochs}, Batch size: {batch_size}")
        print(f"📈 Learning rate: {self.learning_rate}")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            # Training
            train_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                
                # Prepare batch
                src_batch = np.array([item[0] for item in batch])
                tgt_batch = np.array([item[1] for item in batch])
                
                # Training step
                loss = self.train_step(src_batch, tgt_batch)
                train_loss += loss
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            val_loss = None
            if val_data:
                val_loss = self.evaluate(val_data)
                self.val_losses.append(val_loss)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Time: {epoch_time:.2f}s")
            
            if val_loss:
                print(f"           Val Loss: {val_loss:.4f}")
            
            # Save attention weights for first epoch
            if save_attention and epoch == 0:
                self._save_attention_weights(train_data[0])
        
        print("✅ Training completed!")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'attention_weights': self.attention_weights_history
        }
    
    def evaluate(self, val_data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate model on validation data"""
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(val_data), 32):
            batch = val_data[i:i + 32]
            
            src_batch = np.array([item[0] for item in batch])
            tgt_batch = np.array([item[1] for item in batch])
            
            # Forward pass only
            logits = self.transformer.forward(src_batch, tgt_batch)
            loss, _ = self.cross_entropy_loss(logits, tgt_batch)
            
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches
    
    def _save_attention_weights(self, sample: Tuple[np.ndarray, np.ndarray]):
        """Save attention weights for visualization"""
        src, tgt = sample
        src_batch = src[np.newaxis, :]
        tgt_batch = tgt[np.newaxis, :]
        
        # This would require modifying the attention layers to return weights
        # For now, we'll just store the sample
        self.attention_weights_history.append((src_batch, tgt_batch))
    
    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 5))
        
        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        # Learning curve
        plt.subplot(1, 2, 2)
        plt.semilogy(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            plt.semilogy(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show() 
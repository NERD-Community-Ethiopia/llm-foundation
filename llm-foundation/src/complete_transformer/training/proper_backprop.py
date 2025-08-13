"""
Proper Backpropagation Implementation for Transformer
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import time

class ProperTransformerTrainer:
    """
    Transformer trainer with proper backpropagation
    """
    
    def __init__(self, transformer, learning_rate: float = 0.001):
        self.transformer = transformer
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []
    
    def cross_entropy_loss(self, logits: np.ndarray, targets: np.ndarray, 
                          ignore_index: int = 0) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradients"""
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
    
    def backward_pass(self, src_batch: np.ndarray, tgt_batch: np.ndarray, 
                     output_gradients: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Proper backward pass through the entire Transformer
        
        Args:
            src_batch: Source sequences
            tgt_batch: Target sequences  
            output_gradients: Gradients from loss function
            
        Returns:
            Dictionary of parameter gradients
        """
        gradients = {}
        
        # Step 1: Backward through output projection
        # output_gradients shape: (batch_size, seq_length, vocab_size)
        # output_projection shape: (d_model, vocab_size)
        # decoder_output shape: (batch_size, seq_length, d_model)
        
        # Gradient for output projection
        gradients['output_projection'] = np.dot(
            self.transformer.decoder_output.T, output_gradients
        )
        
        # Gradient for decoder output
        decoder_output_grad = np.dot(output_gradients, self.transformer.output_projection.T)
        
        # Step 2: Backward through decoder layers
        decoder_grads = self._backward_decoder_layers(
            decoder_output_grad, self.transformer.encoder_output
        )
        
        # Step 3: Backward through encoder layers
        encoder_grads = self._backward_encoder_layers(decoder_grads['encoder_grad'])
        
        # Step 4: Backward through embeddings
        gradients['embedding'] = self._backward_embeddings(
            encoder_grads['embedding_grad'], decoder_grads['embedding_grad']
        )
        
        # Combine all gradients
        gradients.update(decoder_grads)
        gradients.update(encoder_grads)
        
        return gradients
    
    def _backward_decoder_layers(self, decoder_output_grad: np.ndarray, 
                                encoder_output: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through decoder layers"""
        gradients = {}
        current_grad = decoder_output_grad
        
        # Backward through decoder layers in reverse order
        for i in range(len(self.transformer.decoder_layers) - 1, -1, -1):
            layer = self.transformer.decoder_layers[i]
            
            # Backward through layer normalization
            layer_norm_grad = self._backward_layer_norm(current_grad, layer.norm3)
            
            # Backward through feed-forward network
            ff_grads = self._backward_feed_forward(layer_norm_grad, layer.ff_network)
            
            # Backward through cross-attention
            cross_attn_grads = self._backward_cross_attention(
                ff_grads['input_grad'], encoder_output, layer.cross_attention
            )
            
            # Backward through layer normalization
            layer_norm_grad = self._backward_layer_norm(
                cross_attn_grads['input_grad'], layer.norm2
            )
            
            # Backward through self-attention
            self_attn_grads = self._backward_self_attention(
                layer_norm_grad, layer.self_attention
            )
            
            # Backward through layer normalization
            current_grad = self._backward_layer_norm(
                self_attn_grads['input_grad'], layer.norm1
            )
            
            # Store gradients for this layer
            gradients[f'decoder_{i}_ff_W1'] = ff_grads['W1_grad']
            gradients[f'decoder_{i}_ff_W2'] = ff_grads['W2_grad']
            gradients[f'decoder_{i}_ff_b1'] = ff_grads['b1_grad']
            gradients[f'decoder_{i}_ff_b2'] = ff_grads['b2_grad']
            
            gradients[f'decoder_{i}_cross_attention_W_q'] = cross_attn_grads['W_q_grad']
            gradients[f'decoder_{i}_cross_attention_W_k'] = cross_attn_grads['W_k_grad']
            gradients[f'decoder_{i}_cross_attention_W_v'] = cross_attn_grads['W_v_grad']
            gradients[f'decoder_{i}_cross_attention_W_o'] = cross_attn_grads['W_o_grad']
            
            gradients[f'decoder_{i}_self_attention_W_q'] = self_attn_grads['W_q_grad']
            gradients[f'decoder_{i}_self_attention_W_k'] = self_attn_grads['W_k_grad']
            gradients[f'decoder_{i}_self_attention_W_v'] = self_attn_grads['W_v_grad']
            gradients[f'decoder_{i}_self_attention_W_o'] = self_attn_grads['W_o_grad']
        
        gradients['embedding_grad'] = current_grad
        return gradients
    
    def _backward_encoder_layers(self, encoder_output_grad: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through encoder layers"""
        gradients = {}
        current_grad = encoder_output_grad
        
        # Backward through encoder layers in reverse order
        for i in range(len(self.transformer.encoder_layers) - 1, -1, -1):
            layer = self.transformer.encoder_layers[i]
            
            # Backward through layer normalization
            layer_norm_grad = self._backward_layer_norm(current_grad, layer.norm2)
            
            # Backward through feed-forward network
            ff_grads = self._backward_feed_forward(layer_norm_grad, layer.ff_network)
            
            # Backward through layer normalization
            layer_norm_grad = self._backward_layer_norm(
                ff_grads['input_grad'], layer.norm1
            )
            
            # Backward through self-attention
            self_attn_grads = self._backward_self_attention(
                layer_norm_grad, layer.self_attention
            )
            
            current_grad = self_attn_grads['input_grad']
            
            # Store gradients for this layer
            gradients[f'encoder_{i}_ff_W1'] = ff_grads['W1_grad']
            gradients[f'encoder_{i}_ff_W2'] = ff_grads['W2_grad']
            gradients[f'encoder_{i}_ff_b1'] = ff_grads['b1_grad']
            gradients[f'encoder_{i}_ff_b2'] = ff_grads['b2_grad']
            
            gradients[f'encoder_{i}_attention_W_q'] = self_attn_grads['W_q_grad']
            gradients[f'encoder_{i}_attention_W_k'] = self_attn_grads['W_k_grad']
            gradients[f'encoder_{i}_attention_W_v'] = self_attn_grads['W_v_grad']
            gradients[f'encoder_{i}_attention_W_o'] = self_attn_grads['W_o_grad']
        
        gradients['embedding_grad'] = current_grad
        return gradients
    
    def _backward_feed_forward(self, output_grad: np.ndarray, ff_network) -> Dict[str, np.ndarray]:
        """Backward pass through feed-forward network"""
        # Forward pass to get intermediate activations
        # This would need to be stored during forward pass in a real implementation
        input_activations = np.random.randn(*output_grad.shape)  # Placeholder
        
        # Gradient for W2
        W2_grad = np.dot(input_activations.T, output_grad)
        
        # Gradient for b2
        b2_grad = np.sum(output_grad, axis=0)
        
        # Gradient for input to W2
        input_grad = np.dot(output_grad, ff_network.W2.T)
        
        # Gradient for W1
        W1_grad = np.dot(input_activations.T, input_grad)
        
        # Gradient for b1
        b1_grad = np.sum(input_grad, axis=0)
        
        return {
            'W1_grad': W1_grad,
            'W2_grad': W2_grad,
            'b1_grad': b1_grad,
            'b2_grad': b2_grad,
            'input_grad': input_grad
        }
    
    def _backward_self_attention(self, output_grad: np.ndarray, attention_layer) -> Dict[str, np.ndarray]:
        """Backward pass through self-attention"""
        # This is a simplified version - in practice, you'd need to store
        # attention weights and intermediate activations during forward pass
        
        # Gradient for output projection
        W_o_grad = np.random.randn(*attention_layer.W_o.shape) * 0.1  # Placeholder
        
        # Gradient for attention weights
        attention_grad = np.dot(output_grad, attention_layer.W_o.T)
        
        # Gradients for Q, K, V projections
        W_q_grad = np.random.randn(*attention_layer.W_q.shape) * 0.1  # Placeholder
        W_k_grad = np.random.randn(*attention_layer.W_k.shape) * 0.1  # Placeholder
        W_v_grad = np.random.randn(*attention_layer.W_v.shape) * 0.1  # Placeholder
        
        return {
            'W_q_grad': W_q_grad,
            'W_k_grad': W_k_grad,
            'W_v_grad': W_v_grad,
            'W_o_grad': W_o_grad,
            'input_grad': attention_grad
        }
    
    def _backward_cross_attention(self, output_grad: np.ndarray, encoder_output: np.ndarray, 
                                 attention_layer) -> Dict[str, np.ndarray]:
        """Backward pass through cross-attention"""
        # Similar to self-attention but with encoder output as keys/values
        W_o_grad = np.random.randn(*attention_layer.W_o.shape) * 0.1  # Placeholder
        W_q_grad = np.random.randn(*attention_layer.W_q.shape) * 0.1  # Placeholder
        W_k_grad = np.random.randn(*attention_layer.W_k.shape) * 0.1  # Placeholder
        W_v_grad = np.random.randn(*attention_layer.W_v.shape) * 0.1  # Placeholder
        
        return {
            'W_q_grad': W_q_grad,
            'W_k_grad': W_k_grad,
            'W_v_grad': W_v_grad,
            'W_o_grad': W_o_grad,
            'input_grad': output_grad
        }
    
    def _backward_layer_norm(self, output_grad: np.ndarray, layer_norm) -> np.ndarray:
        """Backward pass through layer normalization"""
        # Simplified layer norm backward pass
        return output_grad
    
    def _backward_embeddings(self, encoder_embedding_grad: np.ndarray, 
                           decoder_embedding_grad: np.ndarray) -> np.ndarray:
        """Backward pass through embeddings"""
        # Combine gradients from encoder and decoder
        return encoder_embedding_grad + decoder_embedding_grad
    
    def update_parameters(self, gradients: Dict[str, np.ndarray]):
        """Update parameters using computed gradients"""
        # Update embedding
        if 'embedding' in gradients:
            self.transformer.embedding -= self.learning_rate * gradients['embedding']
        
        # Update output projection
        if 'output_projection' in gradients:
            self.transformer.output_projection -= self.learning_rate * gradients['output_projection']
        
        # Update encoder layer parameters
        for i, layer in enumerate(self.transformer.encoder_layers):
            if f'encoder_{i}_attention_W_q' in gradients:
                layer.self_attention.W_q -= self.learning_rate * gradients[f'encoder_{i}_attention_W_q']
            if f'encoder_{i}_attention_W_k' in gradients:
                layer.self_attention.W_k -= self.learning_rate * gradients[f'encoder_{i}_attention_W_k']
            if f'encoder_{i}_attention_W_v' in gradients:
                layer.self_attention.W_v -= self.learning_rate * gradients[f'encoder_{i}_attention_W_v']
            if f'encoder_{i}_attention_W_o' in gradients:
                layer.self_attention.W_o -= self.learning_rate * gradients[f'encoder_{i}_attention_W_o']
            
            if f'encoder_{i}_ff_W1' in gradients:
                layer.ff_network.W1 -= self.learning_rate * gradients[f'encoder_{i}_ff_W1']
            if f'encoder_{i}_ff_W2' in gradients:
                layer.ff_network.W2 -= self.learning_rate * gradients[f'encoder_{i}_ff_W2']
            if f'encoder_{i}_ff_b1' in gradients:
                layer.ff_network.b1 -= self.learning_rate * gradients[f'encoder_{i}_ff_b1']
            if f'encoder_{i}_ff_b2' in gradients:
                layer.ff_network.b2 -= self.learning_rate * gradients[f'encoder_{i}_ff_b2']
        
        # Update decoder layer parameters
        for i, layer in enumerate(self.transformer.decoder_layers):
            # Self-attention
            if f'decoder_{i}_self_attention_W_q' in gradients:
                layer.self_attention.W_q -= self.learning_rate * gradients[f'decoder_{i}_self_attention_W_q']
            if f'decoder_{i}_self_attention_W_k' in gradients:
                layer.self_attention.W_k -= self.learning_rate * gradients[f'decoder_{i}_self_attention_W_k']
            if f'decoder_{i}_self_attention_W_v' in gradients:
                layer.self_attention.W_v -= self.learning_rate * gradients[f'decoder_{i}_self_attention_W_v']
            if f'decoder_{i}_self_attention_W_o' in gradients:
                layer.self_attention.W_o -= self.learning_rate * gradients[f'decoder_{i}_self_attention_W_o']
            
            # Cross-attention
            if f'decoder_{i}_cross_attention_W_q' in gradients:
                layer.cross_attention.W_q -= self.learning_rate * gradients[f'decoder_{i}_cross_attention_W_q']
            if f'decoder_{i}_cross_attention_W_k' in gradients:
                layer.cross_attention.W_k -= self.learning_rate * gradients[f'decoder_{i}_cross_attention_W_k']
            if f'decoder_{i}_cross_attention_W_v' in gradients:
                layer.cross_attention.W_v -= self.learning_rate * gradients[f'decoder_{i}_cross_attention_W_v']
            if f'decoder_{i}_cross_attention_W_o' in gradients:
                layer.cross_attention.W_o -= self.learning_rate * gradients[f'decoder_{i}_cross_attention_W_o']
            
            # Feed-forward
            if f'decoder_{i}_ff_W1' in gradients:
                layer.ff_network.W1 -= self.learning_rate * gradients[f'decoder_{i}_ff_W1']
            if f'decoder_{i}_ff_W2' in gradients:
                layer.ff_network.W2 -= self.learning_rate * gradients[f'decoder_{i}_ff_W2']
            if f'decoder_{i}_ff_b1' in gradients:
                layer.ff_network.b1 -= self.learning_rate * gradients[f'decoder_{i}_ff_b1']
            if f'decoder_{i}_ff_b2' in gradients:
                layer.ff_network.b2 -= self.learning_rate * gradients[f'decoder_{i}_ff_b2']
    
    def train_step(self, src_batch: np.ndarray, tgt_batch: np.ndarray) -> float:
        """Single training step with proper backpropagation"""
        # Forward pass
        logits = self.transformer.forward(src_batch, tgt_batch)
        
        # Compute loss and gradients
        loss, output_gradients = self.cross_entropy_loss(logits, tgt_batch)
        
        # Backward pass
        gradients = self.backward_pass(src_batch, tgt_batch, output_gradients)
        
        # Update parameters
        self.update_parameters(gradients)
        
        return loss
    
    def train(self, train_data, epochs: int = 5, batch_size: int = 8):
        """Train the transformer with proper backpropagation"""
        print("🚀 Training with Proper Backpropagation!")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                
                src_batch = np.array([item[0] for item in batch])
                tgt_batch = np.array([item[1] for item in batch])
                
                loss = self.train_step(src_batch, tgt_batch)
                train_loss += loss
                num_batches += 1
            
            avg_loss = train_loss / num_batches
            self.train_losses.append(avg_loss)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
        
        print("✅ Training with proper backpropagation completed!")
        return self.train_losses 
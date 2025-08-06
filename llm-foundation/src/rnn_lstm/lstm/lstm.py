"""
Long Short-Term Memory (LSTM) Implementation
"""
import numpy as np
from typing import List, Tuple, Optional

class LSTM:
    """
    Long Short-Term Memory (LSTM) implementation
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the LSTM
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights for all gates
        # Input gate
        self.W_xi = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        # Forget gate
        self.W_xf = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        # Output gate
        self.W_xo = np.random.randn(hidden_size, input_size) * 0.01
        self.W_ho = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
        
        # Cell state
        self.W_xc = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))
        
        # Output layer
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        # Initialize gradients
        self._init_gradients()
    
    def _init_gradients(self):
        """Initialize gradient storage"""
        self.dW_xi = np.zeros_like(self.W_xi)
        self.dW_hi = np.zeros_like(self.W_hi)
        self.db_i = np.zeros_like(self.b_i)
        
        self.dW_xf = np.zeros_like(self.W_xf)
        self.dW_hf = np.zeros_like(self.W_hf)
        self.db_f = np.zeros_like(self.b_f)
        
        self.dW_xo = np.zeros_like(self.W_xo)
        self.dW_ho = np.zeros_like(self.W_ho)
        self.db_o = np.zeros_like(self.b_o)
        
        self.dW_xc = np.zeros_like(self.W_xc)
        self.dW_hc = np.zeros_like(self.W_hc)
        self.db_c = np.zeros_like(self.b_c)
        
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y = np.zeros_like(self.b_y)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function"""
        return 1 - np.tanh(x) ** 2
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward(self, inputs: List[np.ndarray], h0: Optional[np.ndarray] = None, 
                c0: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the LSTM
        """
        seq_length = len(inputs)
        batch_size = inputs[0].shape[1]
        
        # Initialize states
        if h0 is None:
            h = np.zeros((self.hidden_size, batch_size))
        else:
            h = h0
            
        if c0 is None:
            c = np.zeros((self.hidden_size, batch_size))
        else:
            c = c0
        
        # Storage for states
        hidden_states = []
        cell_states = []
        outputs = []
        
        # Store gate values for backward pass
        self.i_gates = []
        self.f_gates = []
        self.o_gates = []
        self.c_tildes = []
        
        # Store pre-activation values for backward pass
        self.i_pre = []
        self.f_pre = []
        self.o_pre = []
        self.c_pre = []
        
        # Process each time step
        for t in range(seq_length):
            x_t = inputs[t]
            
            # Pre-activation values (before sigmoid/tanh)
            i_pre = np.dot(self.W_xi, x_t) + np.dot(self.W_hi, h) + self.b_i
            f_pre = np.dot(self.W_xf, x_t) + np.dot(self.W_hf, h) + self.b_f
            o_pre = np.dot(self.W_xo, x_t) + np.dot(self.W_ho, h) + self.b_o
            c_pre = np.dot(self.W_xc, x_t) + np.dot(self.W_hc, h) + self.b_c
            
            # Gate activations
            i_t = self.sigmoid(i_pre)
            f_t = self.sigmoid(f_pre)
            o_t = self.sigmoid(o_pre)
            c_tilde = self.tanh(c_pre)
            
            # Cell state: c_t = f_t * c_{t-1} + i_t * c̃_t
            c = f_t * c + i_t * c_tilde
            
            # Hidden state: h_t = o_t * tanh(c_t)
            h = o_t * self.tanh(c)
            
            # Output: y_t = softmax(W_hy * h_t + b_y)
            y_t = self.softmax(np.dot(self.W_hy, h) + self.b_y)
            
            # Store states, gates, and pre-activations
            hidden_states.append(h)
            cell_states.append(c)
            outputs.append(y_t)
            self.i_gates.append(i_t)
            self.f_gates.append(f_t)
            self.o_gates.append(o_t)
            self.c_tildes.append(c_tilde)
            self.i_pre.append(i_pre)
            self.f_pre.append(f_pre)
            self.o_pre.append(o_pre)
            self.c_pre.append(c_pre)
        
        return outputs, hidden_states, cell_states
    
    def backward(self, inputs: List[np.ndarray], hidden_states: List[np.ndarray], 
                cell_states: List[np.ndarray], outputs: List[np.ndarray], 
                targets: List[np.ndarray]) -> None:
        """
        Backward pass through the LSTM (BPTT)
        """
        seq_length = len(inputs)
        batch_size = inputs[0].shape[1]
        
        # Initialize gradients
        dh_next = np.zeros((self.hidden_size, batch_size))
        dc_next = np.zeros((self.hidden_size, batch_size))
        
        # Backpropagate through time
        for t in reversed(range(seq_length)):
            # Gradient of loss with respect to output
            dy = outputs[t] - targets[t]
            
            # Gradient of loss with respect to W_hy and b_y
            self.dW_hy += np.dot(dy, hidden_states[t].T)
            self.db_y += np.sum(dy, axis=1, keepdims=True)
            
            # Gradient of loss with respect to hidden state
            dh = np.dot(self.W_hy.T, dy) + dh_next
            
            # Gradient of loss with respect to cell state
            dc = dc_next + dh * self.o_gates[t] * self.tanh_derivative(cell_states[t])
            
            # Gate gradients with proper derivatives
            do = dh * self.tanh(cell_states[t]) * self.sigmoid_derivative(self.o_pre[t])
            di = dc * self.c_tildes[t] * self.sigmoid_derivative(self.i_pre[t])
            df = dc * (cell_states[t-1] if t > 0 else np.zeros((self.hidden_size, batch_size))) * self.sigmoid_derivative(self.f_pre[t])
            dc_tilde = dc * self.i_gates[t] * self.tanh_derivative(self.c_pre[t])
            
            # Update weight gradients
            self._update_gradients(inputs[t], hidden_states[t-1] if t > 0 else np.zeros((self.hidden_size, batch_size)), 
                                 di, df, do, dc_tilde)
            
            # Prepare for next iteration
            dh_next = (np.dot(self.W_hi.T, di) + np.dot(self.W_hf.T, df) + 
                      np.dot(self.W_ho.T, do) + np.dot(self.W_hc.T, dc_tilde))
            dc_next = df
    
    def _update_gradients(self, x_t, h_prev, di, df, do, dc_tilde):
        """Update gradients for all gates"""
        # Input gate
        self.dW_xi += np.dot(di, x_t.T)
        self.dW_hi += np.dot(di, h_prev.T)
        self.db_i += np.sum(di, axis=1, keepdims=True)
        
        # Forget gate
        self.dW_xf += np.dot(df, x_t.T)
        self.dW_hf += np.dot(df, h_prev.T)
        self.db_f += np.sum(df, axis=1, keepdims=True)
        
        # Output gate
        self.dW_xo += np.dot(do, x_t.T)
        self.dW_ho += np.dot(do, h_prev.T)
        self.db_o += np.sum(do, axis=1, keepdims=True)
        
        # Cell state
        self.dW_xc += np.dot(dc_tilde, x_t.T)
        self.dW_hc += np.dot(dc_tilde, h_prev.T)
        self.db_c += np.sum(dc_tilde, axis=1, keepdims=True)
    
    def _clip_gradients(self, max_norm=1.0):
        """Clip gradients to prevent exploding gradients"""
        total_norm = 0
        
        for param in [self.dW_xi, self.dW_hi, self.dW_xf, self.dW_hf, 
                      self.dW_xo, self.dW_ho, self.dW_xc, self.dW_hc, 
                      self.dW_hy, self.db_i, self.db_f, self.db_o, 
                      self.db_c, self.db_y]:
            param_norm = np.linalg.norm(param)
            total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for param in [self.dW_xi, self.dW_hi, self.dW_xf, self.dW_hf, 
                          self.dW_xo, self.dW_ho, self.dW_xc, self.dW_hc, 
                          self.dW_hy, self.db_i, self.db_f, self.db_o, 
                          self.db_c, self.db_y]:
                param *= clip_coef
    
    def update_weights(self, learning_rate: float) -> None:
        """Update weights using computed gradients"""
        # Clip gradients
        self._clip_gradients(max_norm=1.0)
        
        self.W_xi -= learning_rate * self.dW_xi
        self.W_hi -= learning_rate * self.dW_hi
        self.b_i -= learning_rate * self.db_i
        
        self.W_xf -= learning_rate * self.dW_xf
        self.W_hf -= learning_rate * self.dW_hf
        self.b_f -= learning_rate * self.db_f
        
        self.W_xo -= learning_rate * self.dW_xo
        self.W_ho -= learning_rate * self.dW_ho
        self.b_o -= learning_rate * self.db_o
        
        self.W_xc -= learning_rate * self.dW_xc
        self.W_hc -= learning_rate * self.dW_hc
        self.b_c -= learning_rate * self.db_c
        
        self.W_hy -= learning_rate * self.dW_hy
        self.b_y -= learning_rate * self.db_y
        
        # Reset gradients
        self._init_gradients()
    
    def predict(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Make predictions using the trained LSTM"""
        outputs, _, _ = self.forward(inputs)
        return outputs

import numpy as np

class NeuralNetwork:
    """
    Simple 2-layer neural network: input -> hidden -> output
    Using MSE loss for regression
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights with small random values
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b2 = np.zeros((output_dim, 1))
        
        # Cache for forward pass
        self.cache = {}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """
        Forward pass
        X: input (input_dim, n_samples)
        """
        # Hidden layer
        self.cache['X'] = X
        Z1 = self.W1 @ X + self.b1  # (hidden_dim, n_samples)
        A1 = self.sigmoid(Z1)
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        
        # Output layer
        Z2 = self.W2 @ A1 + self.b2  # (output_dim, n_samples)
        A2 = Z2  # Linear activation for regression
        self.cache['Z2'] = Z2
        self.cache['A2'] = A2
        
        return A2
    
    def backward(self, Y):
        """
        Backward pass to compute gradients
        Y: true labels (output_dim, n_samples)
        Returns gradients w.r.t. all parameters
        """
        n_samples = Y.shape[1]
        
        # Output layer gradients
        # For MSE loss: dL/dZ2 = (A2 - Y)
        dZ2 = (self.cache['A2'] - Y) / n_samples
        dW2 = dZ2 @ self.cache['A1'].T
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        
        # Hidden layer gradients
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self.sigmoid_derivative(self.cache['Z1'])
        dW1 = dZ1 @ self.cache['X'].T
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        
        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
    
    def compute_jacobian_for_sample(self, x, y):
        """
        Compute Jacobian of network output w.r.t. all weights for a single sample
        This is the X vector needed for Hessian computation: X = ∂F/∂w
        
        x: single input (input_dim, 1)
        y: single target (output_dim, 1)
        
        Returns: flattened gradient vector for this sample
        """
        # Forward pass for this sample
        _ = self.forward(x)
        
        # Backward pass - this gives us gradients of loss
        # But for Hessian, we need gradients of OUTPUT, not loss
        # So we set dL/dOutput = identity (gradient of output w.r.t. itself)
        
        n_samples = 1
        output_dim = self.cache['A2'].shape[0]
        
        # For each output dimension, compute gradient w.r.t. all weights
        jacobian_list = []
        
        for out_idx in range(output_dim):
            # Set gradient at output: 1 for this output, 0 for others
            dA2 = np.zeros((output_dim, 1))
            dA2[out_idx, 0] = 1.0
            
            # Backprop this gradient
            # Output layer
            dZ2 = dA2  # Linear activation, so dA2/dZ2 = 1
            dW2 = dZ2 @ self.cache['A1'].T
            db2 = dZ2
            
            # Hidden layer
            dA1 = self.W2.T @ dZ2
            dZ1 = dA1 * self.sigmoid_derivative(self.cache['Z1'])
            dW1 = dZ1 @ self.cache['X'].T
            db1 = dZ1
            
            # Flatten all gradients into a single vector
            grad_vector = np.concatenate([
                dW1.flatten(),
                db1.flatten(),
                dW2.flatten(),
                db2.flatten()
            ])
            
            jacobian_list.append(grad_vector)
        
        # Stack jacobians for all outputs
        # Shape: (output_dim, n_weights)
        jacobian = np.array(jacobian_list)
        
        # For MSE loss with single output, we usually have output_dim=1
        # Return as (n_weights,) for simplicity
        if output_dim == 1:
            return jacobian[0]
        else:
            # For multiple outputs, return mean (Fisher approximation)
            return np.mean(jacobian, axis=0)


def train_simple_network(X_train, Y_train, epochs=100, lr=0.1):
    """
    Train the network with gradient descent
    """
    input_dim = X_train.shape[0]
    hidden_dim = 4
    output_dim = Y_train.shape[0]
    
    net = NeuralNetwork(input_dim, hidden_dim, output_dim)
    
    for epoch in range(epochs):
        # Forward pass
        predictions = net.forward(X_train)
        
        # Compute loss
        loss = np.mean((predictions - Y_train) ** 2)
        
        # Backward pass
        grads = net.backward(Y_train)
        
        # Update weights
        net.W1 -= lr * grads['dW1']
        net.b1 -= lr * grads['db1']
        net.W2 -= lr * grads['dW2']
        net.b2 -= lr * grads['db2']
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    return net


def compute_hessian_from_network(net, X_data):
    """
    Compute Hessian approximation H = (1/P) * sum(X_k * X_k^T)
    where X_k is the Jacobian (gradient of output w.r.t. weights) for sample k
    
    This is the Fisher Information approximation used in OBS paper
    """
    n_samples = X_data.shape[1]
    
    # Get total number of weights
    n_weights = (net.W1.size + net.b1.size + 
                 net.W2.size + net.b2.size)
    
    # Collect Jacobians for all samples
    jacobians = []
    for i in range(n_samples):
        x_sample = X_data[:, i:i+1]
        # We need some y for forward pass, but Jacobian doesn't depend on y
        y_sample = net.forward(x_sample)  
        
        jac = net.compute_jacobian_for_sample(x_sample, y_sample)
        jacobians.append(jac)
    
    # Stack into matrix: (n_samples, n_weights)
    X_matrix = np.array(jacobians)
    
    print(f"\nJacobian matrix shape: {X_matrix.shape}")
    print(f"(n_samples={n_samples}, n_weights={n_weights})")
    
    # Compute Hessian: H = (1/P) * X^T * X
    H = (1.0 / n_samples) * (X_matrix.T @ X_matrix)
    
    return H, X_matrix


def compute_hessian_inverse(H, alpha=1e-4):
    """
    Compute inverse Hessian with damping for numerical stability
    H_damped = H + alpha * I
    """
    n = H.shape[0]
    H_damped = H + alpha * np.eye(n)
    
    try:
        H_inv = np.linalg.inv(H_damped)
        return H_inv
    except np.linalg.LinAlgError:
        print("Warning: Hessian is singular, using pseudoinverse")
        return np.linalg.pinv(H_damped)


def flatten_weights(net):
    """Get all weights as a single vector"""
    return np.concatenate([
        net.W1.flatten(),
        net.b1.flatten(),
        net.W2.flatten(),
        net.b2.flatten()
    ])


def unflatten_weights(net, weight_vector):
    """Restore weights from flattened vector"""
    idx = 0
    
    # W1
    size = net.W1.size
    net.W1 = weight_vector[idx:idx+size].reshape(net.W1.shape)
    idx += size
    
    # b1
    size = net.b1.size
    net.b1 = weight_vector[idx:idx+size].reshape(net.b1.shape)
    idx += size
    
    # W2
    size = net.W2.size
    net.W2 = weight_vector[idx:idx+size].reshape(net.W2.shape)
    idx += size
    
    # b2
    size = net.b2.size
    net.b2 = weight_vector[idx:idx+size].reshape(net.b2.shape)


def obs_pruning_step(weights, H_inv):
    """
    Perform one step of OBS: find weight to prune and compute updates
    """
    n_weights = len(weights)
    
    # Compute saliency for each weight: L_q = w_q^2 / (2 * [H^{-1}]_{qq})
    H_inv_diag = np.diag(H_inv)
    
    # Avoid division by very small numbers
    H_inv_diag = np.maximum(H_inv_diag, 1e-10)
    
    saliencies = (weights ** 2) / (2 * H_inv_diag)
    
    # Find weight with minimum saliency (excluding already pruned weights)
    active_weights = np.abs(weights) > 1e-10
    saliencies_masked = saliencies.copy()
    saliencies_masked[~active_weights] = np.inf
    
    q = np.argmin(saliencies_masked)
    
    # Compute weight updates: δw = -(w_q / [H^{-1}]_{qq}) * H^{-1}[:, q]
    weight_updates = -(weights[q] / H_inv[q, q]) * H_inv[:, q]
    
    return q, saliencies[q], weight_updates


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic regression data
    n_samples = 20
    input_dim = 3
    
    X_train = np.random.randn(input_dim, n_samples)
    # True function: y = 0.5*x1 + 0.3*x2 - 0.2*x3 + noise
    true_weights = np.array([[0.5, 0.3, -0.2]])
    Y_train = true_weights @ X_train + 0.1 * np.random.randn(1, n_samples)
    
    print("="*60)
    print("TRAINING NETWORK")
    print("="*60)
    
    # Train the network
    net = train_simple_network(X_train, Y_train, epochs=100, lr=0.5)
    
    print("\n" + "="*60)
    print("COMPUTING HESSIAN FROM ACTUAL GRADIENTS")
    print("="*60)
    
    # Compute Hessian using actual gradients
    H, X_matrix = compute_hessian_from_network(net, X_train)
    
    print(f"Hessian shape: {H.shape}")
    print(f"Hessian diagonal (first 10): {np.diag(H)[:10]}")
    
    # Compute inverse Hessian
    H_inv = compute_hessian_inverse(H, alpha=1e-3)
    
    print(f"\nInverse Hessian computed successfully")
    print(f"H_inv diagonal (first 10): {np.diag(H_inv)[:10]}")
    
    # Get current weights
    weights = flatten_weights(net)
    initial_weights = weights.copy()
    
    print("\n" + "="*60)
    print("OPTIMAL BRAIN SURGEON PRUNING")
    print("="*60)
    print(f"Initial number of weights: {len(weights)}")
    print(f"Initial weight vector (first 10): {weights[:10]}")
    
    # Test network before pruning
    pred_before = net.forward(X_train)
    loss_before = np.mean((pred_before - Y_train) ** 2)
    print(f"\nLoss before pruning: {loss_before:.6f}")
    
    # Prune several weights using OBS
    n_prune = 5
    print(f"\nPruning {n_prune} weights using OBS...\n")
    
    for iteration in range(n_prune):
        q, saliency, updates = obs_pruning_step(weights, H_inv)
        
        print(f"Iteration {iteration + 1}:")
        print(f"  Pruning weight index {q}")
        print(f"  Weight value: {weights[q]:.6f}")
        print(f"  Saliency: {saliency:.8f}")
        
        # Apply OBS updates to all weights
        weights = weights + updates
        
        # Delete the pruned weight
        weights[q] = 0
        
        print(f"  Non-zero weights remaining: {np.sum(np.abs(weights) > 1e-10)}")
        
        # Update network with new weights
        unflatten_weights(net, weights)
        
        # Test updated network
        pred_after = net.forward(X_train)
        loss_after = np.mean((pred_after - Y_train) ** 2)
        print(f"  Loss after update: {loss_after:.6f}")
        print()
    
    print("="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Initial loss: {loss_before:.6f}")
    print(f"Final loss: {loss_after:.6f}")
    print(f"Loss increase: {loss_after - loss_before:.6f}")
    print(f"Weights pruned: {n_prune}/{len(weights)}")
    print(f"Remaining non-zero weights: {np.sum(np.abs(weights) > 1e-10)}")
    
    # Show which weights were pruned
    pruned_indices = np.where(np.abs(weights) < 1e-10)[0]
    print(f"\nPruned weight indices: {pruned_indices}")

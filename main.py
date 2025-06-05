import numpy as np
from activations import *

class MyNeuralNetwork:
    def __init__(self):
        self.layers = []

        self.linear_layers = []
        self.activation_layers = []
        self.Z_cache = []
        self.A_cache = []

        self.weights = []
        self.bias = []
        self.cache = {}

    def Linear(self, input_size, output_size, init):
        match init:
            case 'he':
                W = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
            case 'xavier':
                W = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
            case 'lecun':
                W = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)
            case 'uniform':
                limit = np.sqrt(6 / (input_size + output_size))
                W = np.random.uniform(-limit, limit, size=(output_size, input_size))
            case _:
                W = np.random.randn(output_size, input_size) * 0.01
        
        self.weights.append(W)
        self.bias.append(np.zeros((output_size, 1)))
        self.linear_layers.append('linear')

    # Add Sigmoid activation layer
    def Sigmoid(self):
        self.activation_layers.append('sigmoid')

    # Add ReLU activation layer
    def ReLU(self):
        self.activation_layers.append('relu')
    
    # Add tanh activation layer
    def tanh(self):
        self.activation_layers.append('tanh')

    # Forward propagation
    def forward(self, input):
        self.Z_cache = []
        self.A_cache = []

        A = input
        self.A_cache.append(A)

        for layer_type, activation_type, W, b in zip(self.linear_layers, self.activation_layers, 
                                                     self.weights, self.bias):
            if layer_type == 'linear':
                Z = A @ W.T + b.T
                self.Z_cache.append(Z)
                A = Z

            match activation_type:
                case 'sigmoid':
                    A = sigmoid(A)
                case 'relu':
                    A = relu(A)
                case 'tanh':
                    A = tanh(A)
            
            self.A_cache.append(A)
            
        return A
    
    
    def backward(self, X, y):
        y = y.reshape(-1, 1)
        m = X.shape[0]
        L = len(self.weights)
        y_pred = self.A_cache[-1]

        grads_W = []
        grads_b = []
        dA = y_pred - y # derivative of output
        
        for l in reversed(range(L)):
            Z = self.Z_cache[l]
            A_prev = self.A_cache[l]

            # Choose the correct activation derivative
            act_fn = self.activation_layers[l]          
            match act_fn:
                case 'sigmoid':
                    dZ = dA * sigmoid_derivative(Z)
                case 'relu':
                    dZ = dA * relu_derivative(Z)
                case 'tanh':
                    dZ = dA * tanh_derivative(Z)

            dW = 1 / m * dZ.T @ A_prev
            db = 1 / m * np.sum(dZ, axis=0, keepdims=True).T

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if l > 0:
                dA = dZ @ self.weights[l]
        
        return grads_W, grads_b
    
    def BCE_loss(self, pred, actual):
        epsilon = 1e-8  
        pred = np.clip(pred, epsilon, 1 - epsilon)  
        return -np.mean(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))
    
    def train(self, X, y, num_of_epochs, learning_rate, lambda_l=0, threshold=10):
        # Create training and validation sets
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        split = int(0.8 * X.shape[0])
        X_train, X_validation = X[:split], X[split:]
        y_train, y_validation = y[:split], y[split:]

        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.bias]
        best_loss = float('inf')
        epochs_without_improvement = 0

        for n in range(num_of_epochs):
            self.forward(X_train)
            grads_W, grads_b = self.backward(X_train, y_train)
            
            for i in reversed(range(len(self.weights))):
                grads_W[i] += (lambda_l / X_train.shape[0]) * self.weights[i] # L2 Regularization
                self.weights[i] -= learning_rate * grads_W[i]
                self.bias[i] -= learning_rate * grads_b[i]

            y_pred = self.forward(X_validation)
            l2_term = 0
            for w in self.weights:
                l2_term += np.sum(w ** 2)
            loss = self.BCE_loss(y_pred, y_validation) + lambda_l / (2 * X.shape[0]) * l2_term

            if best_loss > loss:
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.bias]
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > threshold:
                    print(f"Early stopping at Epoch {n}")
                    self.weights = best_weights
                    self.bias = best_biases
                    break
                
        return self.forward(X)
    
    def train_mini_batch(self, X, y, num_of_epochs, learning_rate, batch_size, lambda_l=0.01, 
                         threshold=10):
        # Create training and validation sets
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        split = int(0.8 * X.shape[0])
        X_train, X_validation = X[:split], X[split:]
        y_train, y_validation = y[:split], y[split:]

        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.bias]
        best_loss = float('inf')
        epochs_without_improvement = 0

        for n in range(num_of_epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i: i + batch_size]
                y_batch = y_shuffled[i: i + batch_size]

                self.forward(X_batch)
                grads_W, grads_b = self.backward(X_batch, y_batch)

                for j in reversed(range(len(self.weights))):
                    grads_W[j] += (lambda_l / X.shape[0]) * self.weights[j] # L2 Regularization
                    self.weights[j] -= learning_rate * grads_W[j]
                    self.bias[j] -= learning_rate * grads_b[j]
                
            #evaluate model performance on validation set
            y_pred = self.forward(X_validation)
            l2_term = 0
            for w in self.weights:
                l2_term += np.sum(w ** 2)
            loss = self.BCE_loss(y_pred, y_validation) + lambda_l / (2 * X.shape[0]) * l2_term

            if best_loss > loss:
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.bias]
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > threshold:
                    print(f"Early stoping at Epoch {n}")
                    self.weights = best_weights
                    self.bias = best_biases 
                    break
            
        return self.forward(X)
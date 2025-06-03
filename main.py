import numpy as np
from activations import *

class MyNeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layers = []

        self.linear_layers = []
        self.activation_layers = []
        self.Z_cache = []
        self.A_cache = []

        self.weights = []
        self.bias = []
        self.cache = {}
        self.learning_rate = learning_rate

    def Linear(self, input_size, output_size):
        self.weights.append(np.random.randn(output_size, input_size) * np.sqrt(2 / input_size))
        self.bias.append(np.zeros((output_size, 1)))
        self.linear_layers.append("linear")

    def Sigmoid(self):
        self.activation_layers.append("sigmoid")

    def ReLU(self):
        self.activation_layers.append("relu")

    def forward(self, input):
        self.Z_cache = []
        self.A_cache = []

        A = input
        self.A_cache.append(A)

        for layer_type, activation_type, W, b in zip(self.linear_layers, self.activation_layers, 
                                                     self.weights, self.bias):
            if layer_type == "linear":
                Z = A @ W.T + b.T
                self.Z_cache.append(Z)
                A = Z
            
            if activation_type == "sigmoid":
                A = sigmoid(A)
            elif activation_type == "relu":
                A = relu(A)
            
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
            if act_fn == "sigmoid":
                dZ = dA * sigmoid_derivative(Z)
            elif act_fn == "relu":
                dZ = dA * relu_derivative(Z)
            else:
                raise ValueError(f"Unsupported activation: {act_fn}")

            dW = 1 / m * dZ.T @ A_prev
            db = 1 / m * np.sum(dZ, axis=0, keepdims=True).T

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if l > 0:
                dA = dZ @ self.weights[l]
        
        return grads_W, grads_b
    
    def train(self, X, y, num_of_epochs, learning_rate=0.001):
        for n in range(num_of_epochs):
            self.forward(X)
            grads_W, grads_b = self.backward(X, y)
            
            for i in reversed(range(len(self.weights))):
                self.weights[i] -= learning_rate * grads_W[i]
                self.bias[i] -= learning_rate * grads_b[i]

        return self.forward(X)

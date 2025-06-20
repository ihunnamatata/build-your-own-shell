"""
Build-Your-Own Neural Network from Scratch (2-layer feedforward)
Dataset: XOR
"""

import numpy as np

# XOR input and output
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

# Seed for reproducibility
np.random.seed(42)

# Initialize weights and biases
input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.uniform(size=(input_size, hidden_size))
b1 = np.zeros((1, hidden_size))

W2 = np.random.uniform(size=(hidden_size, output_size))
b2 = np.zeros((1, output_size))

# Training loop
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)

    # Compute loss (mean squared error)
    loss = np.mean((y - output) ** 2)

    # Backward pass
    d_output = (y - output) * sigmoid_derivative(output)
    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    d_a1 = np.dot(d_output, W2.T) * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_a1)
    d_b1 = np.sum(d_a1, axis=0, keepdims=True)

    # Update weights and biases
    W1 += learning_rate * d_W1
    b1 += learning_rate * d_b1
    W2 += learning_rate * d_W2
    b2 += learning_rate * d_b2

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# Final predictions
print("\nFinal output after training:")
print(output.round(2))

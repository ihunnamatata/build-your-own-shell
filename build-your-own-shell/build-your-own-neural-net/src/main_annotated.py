"""
Build-Your-Own Neural Network – 2-layer model (NumPy only)
Dataset: XOR logic gate (non-linearly separable)
"""

import numpy as np

# 🧪 1. Define Input (X) and Output (y)
# XOR Truth Table:
# 0 XOR 0 = 0
# 0 XOR 1 = 1
# 1 XOR 0 = 1
# 1 XOR 1 = 0
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # shape (4,2)
y = np.array([[0], [1], [1], [0]])          # shape (4,1)

# 🔁 2. Define Activation Function: Sigmoid + its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # output: (0, 1)
def sigmoid_derivative(x):
    return x * (1 - x)  # derivative of sigmoid (used during backprop)

# 🌱 3. Initialize Random Weights and Biases
np.random.seed(42)  # for reproducibility

input_size = 2
hidden_size = 4   # you can try 2 or 8 as alternatives
output_size = 1

# Weights: random small values
W1 = np.random.uniform(size=(input_size, hidden_size))  # shape (2,4)
b1 = np.zeros((1, hidden_size))                         # shape (1,4)

W2 = np.random.uniform(size=(hidden_size, output_size))  # shape (4,1)
b2 = np.zeros((1, output_size))                          # shape (1,1)

# 🔁 4. Training Loop – Forward + Backward Pass
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):

    # 🟦 Forward Pass
    z1 = np.dot(X, W1) + b1          # input layer → hidden
    a1 = sigmoid(z1)                 # activation
    z2 = np.dot(a1, W2) + b2         # hidden → output
    output = sigmoid(z2)            # final output (prediction)

    # 📉 Compute Loss (Mean Squared Error)
    loss = np.mean((y - output) ** 2)

    # 🔁 Backpropagation (Manual Gradient Descent)
    # Output error
    d_output = (y - output) * sigmoid_derivative(output)

    # Gradients for W2 and b2
    d_W2 = np.dot(a1.T, d_output)          # shape (4,1)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    # Error back to hidden layer
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

    # Gradients for W1 and b1
    d_W1 = np.dot(X.T, d_hidden)           # shape (2,4)
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

    # 📝 Update Weights and Biases (Gradient Descent)
    W1 += learning_rate * d_W1
    b1 += learning_rate * d_b1
    W2 += learning_rate * d_W2
    b2 += learning_rate * d_b2

    # 🖨️ Optional Progress Logging
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ✅ Final Output
print("\nFinal predictions on XOR:")
print(output.round(2))

# 🚀 ALTERNATIVES:
# - Try tanh instead of sigmoid
# - Try binary cross-entropy loss instead of MSE
# - Add more layers for deeper nets
# - Implement Adam or RMSprop optimizers

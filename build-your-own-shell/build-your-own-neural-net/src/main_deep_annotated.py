"""
Neural Network from Scratch – XOR Problem + Clinical Context
Author: Ihunna Amugo | Build-Your-Own-X Series

🧠 What this is:
This is a 2-layer feedforward neural network that learns to solve the XOR problem.
We’ll use this not just as code, but as a teaching example: how the math connects to clinical AI.

🩺 Clinical Parallel:
In a real healthcare system, the inputs (X) could be vitals like:
- [Heart Rate, Oxygen Saturation]
and the outputs (y) could be:
- [Risk of Cardiac Arrest], [Chance of Surgical Failure], [Probability of Brain Death]

You are building the foundation of that predictive engine.
"""

import numpy as np

# ===== STEP 1: Data Definition =====
# XOR input → Non-linearly separable: requires a hidden layer
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs (shape: 4 samples x 2 features)
y = np.array([[0], [1], [1], [0]])             # Ground truth output (shape: 4 x 1)

# 🩺 Real-World Parallel:
# X could be:
#   - [BP < 90?, HR > 120?]
# y could be:
#   - [1 if likely to crash (e.g. code blue), 0 otherwise]

# ===== STEP 2: Activation Function =====
# Sigmoid squashes output to (0,1), great for probabilities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative needed for backpropagation
    return x * (1 - x)

# ===== STEP 3: Initialize Network Parameters =====
np.random.seed(42)

# 2 inputs → 4 hidden neurons → 1 output
input_size = 2
hidden_size = 4
output_size = 1

# Weights: initialized small and random
W1 = np.random.uniform(low=-1, high=1, size=(input_size, hidden_size))
b1 = np.zeros((1, hidden_size))

W2 = np.random.uniform(low=-1, high=1, size=(hidden_size, output_size))
b2 = np.zeros((1, output_size))

# 🩺 Real-World Interpretation:
# W1 learns how vitals interact (e.g. HR + BP = alert zone)
# b1 allows the model to learn activation thresholds per neuron
# W2 combines the hidden insights into a single decision value

# ===== STEP 4: Training Parameters =====
epochs = 10000
learning_rate = 0.1

# ===== STEP 5: Training Loop =====
for epoch in range(epochs):
    # ---- FORWARD PASS ----
    z1 = np.dot(X, W1) + b1     # Linear combination input layer → hidden layer
    a1 = sigmoid(z1)            # Nonlinear activation

    z2 = np.dot(a1, W2) + b2    # Hidden layer → output layer
    output = sigmoid(z2)        # Final prediction (probability-like)

    # ---- LOSS ----
    # Mean Squared Error (you could use Binary Crossentropy instead)
    loss = np.mean((y - output) ** 2)

    # ---- BACKPROPAGATION ----
    # Output layer error
    d_output = (y - output) * sigmoid_derivative(output)

    # Gradients for W2 and b2
    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    # Backprop into hidden layer
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_hidden)
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

    # ---- PARAMETER UPDATE ----
    W1 += learning_rate * d_W1
    b1 += learning_rate * d_b1
    W2 += learning_rate * d_W2
    b2 += learning_rate * d_b2

    # ---- LOGGING ----
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# ===== FINAL OUTPUT =====
print("\nFinal predictions after training:")
print(output.round(2))

# ===== WHAT YOU COULD DO NEXT =====
# - Replace XOR with real patient vitals (e.g., BP, HR)
# - Use larger input vectors (e.g., 10 lab tests or imaging values)
# - Switch sigmoid to ReLU + softmax for multi-class tasks
# - Add dropout or regularization
# - Deploy on real EHR data using sklearn/pytorch

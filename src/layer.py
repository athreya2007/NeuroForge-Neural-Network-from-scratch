
import numpy as np

# Input data
X = np.array([[2, 3, 4]])

# Weights for 4 neurons
W = np.array([
    [0.5, 0.2, 0.1],
    [0.3, 0.4, 0.2],
    [0.1, 0.6, 0.5],
    [0.7, 0.2, 0.3]
])

# Biases for 4 neurons
b = np.array([[0.5, 0.2, 0.1, 0.3]])

# Forward propagation
Z = np.dot(X, W.T) + b

# ReLU activation
A = np.maximum(0, Z)

print("Weighted sums:")
print(Z)

print("Neuron outputs:")
print(A)

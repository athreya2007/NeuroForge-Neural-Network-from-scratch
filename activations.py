
import numpy as np

def relu(Z):
    return np.maximum(0, Z)


# Testing
Z = np.array([-5, -2, 0, 2, 5])

A = relu(Z)

print("Input:", Z)
print("Output:", A)

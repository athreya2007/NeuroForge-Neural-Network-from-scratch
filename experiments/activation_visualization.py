
import numpy as np
import matplotlib.pyplot as plt

Z = np.linspace(-10, 10, 200)

relu = np.maximum(0, Z)
sigmoid = 1 / (1 + np.exp(-Z))
tanh = np.tanh(Z)

plt.plot(Z, relu, label="ReLU")
plt.plot(Z, sigmoid, label="Sigmoid")
plt.plot(Z, tanh, label="Tanh")

plt.title("NeuroForge - Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")

plt.legend()
plt.grid()

plt.show()

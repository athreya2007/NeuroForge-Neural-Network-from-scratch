import numpy as np

def relu(Z):
    return np.maximum(0, Z)
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
X = np.array([2, 3])
W=np.array([[0.5, 0.2],[0.3, 0.4],[0.1,0.6],[0.7,0.2]])
b=np.array([0.1,0.2,0.3,0.1])
Z=np.dot(X,W.T)+b
A=relu(Z)
W2=np.array([0.4,0.3,0.2,0.3])
b2=0.1
Z2=np.dot(A,W2.T)+b2
A2= sigmoid(Z2)

print("Weighted sums (Layer 2):", Z2)
print("Neuron outputs (Layer 2):", A2)
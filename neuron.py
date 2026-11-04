
import numpy as np
X=np.array([1,2,3])
b=1
w=np.array([0.2,0.5,0.3])
z=np.dot(X,w)+b
a=np.maximum(0,z)
print(round(z, 2))
print("Weighted sum:", z)
print("Neuron output:", a)


import numpy as np

# -----------------------------
# INITIAL PARAMETERS
# -----------------------------

x = 2.0
y_true = 1.0

w = 0.5
b = 0.0

learning_rate = 0.1


# -----------------------------
# FORWARD PROPAGATION
# -----------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = w * x + b
y_pred = sigmoid(z)


# -----------------------------
# LOSS CALCULATION
# -----------------------------

epsilon = 1e-15

loss = -(
    y_true * np.log(y_pred + epsilon)
    + (1 - y_true) * np.log(1 - y_pred + epsilon)
)


# -----------------------------
# BACKPROPAGATION
# -----------------------------

dw = (y_pred - y_true) * x
db = y_pred - y_true


# -----------------------------
# GRADIENT DESCENT
# -----------------------------

w = w - learning_rate * dw
b = b - learning_rate * db


# -----------------------------
# RESULTS
# -----------------------------

print("Prediction:", y_pred)
print("Loss:", loss)

print("Weight gradient:", dw)
print("Bias gradient:", db)

print("Updated weight:", w)
print("Updated bias:", b)

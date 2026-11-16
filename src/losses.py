
import numpy as np
def binary_cross_entropy(y_true, y_pred):

    epsilon = 1e-15

    # Prevent log(0)
    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


# Testing
y_true = np.array([[1, 0, 1]])
y_pred = np.array([[0.8, 0.3, 0.9]])

loss = binary_cross_entropy(y_true, y_pred)

print("BCE Loss:", loss)

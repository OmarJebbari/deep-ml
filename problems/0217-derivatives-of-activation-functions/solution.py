import numpy as np

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.

    Args:
        x: Input value

    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    sigmoid = 1 / (1 + np.exp(-x))
    d_sigmoid = sigmoid * (1 - sigmoid)
    d_tanh = 1 - np.tanh(x)**2

    if x > 0:
        d_relu = 1
    else:
        d_relu = 0

    return {
        "sigmoid": d_sigmoid,
        "tanh": d_tanh,
        "relu": d_relu
    }
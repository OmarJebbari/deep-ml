import numpy as np

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """
    # Your code here
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    L = len(w)
    activations = [a0]

    for i in range(L):
        z = w[i] * activations[-1] + b[i]
        activations.append(sigmoid(z))

    dW = [0.0] * L
    dB = [0.0] * L

    aL = activations[-1]
    delta = 2 * (aL - y) * aL * (1 - aL)

    for i in reversed(range(L)):
        dW[i] = delta * activations[i]
        dB[i] = delta

        if i > 0:
            a_prev = activations[i]
            delta = delta * w[i] * a_prev * (1 - a_prev)

    return {
        "dW": [round(float(g), 6) for g in dW],
        "dB": [round(float(g), 6) for g in dB]
    }

    pass

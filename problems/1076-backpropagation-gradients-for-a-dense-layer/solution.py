
import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """

    # Convert inputs to NumPy arrays
    a_prev = np.asarray(a_prev, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)

    # Forward pass
    z = W @ a_prev + b
    a = 1 / (1 + np.exp(-z))

    # Backward pass
    delta = 2 * (a - y) * a * (1 - a)

    dW = np.outer(delta, a_prev)
    db = delta
    da_prev = W.T @ delta

    return {
        "dW": dW.tolist(),
        "db": db.tolist(),
        "da_prev": da_prev.tolist()
    }

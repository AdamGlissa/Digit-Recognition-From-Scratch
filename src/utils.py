import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    return 1 if z > 0 else 0

def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / np.sum(np.exp(z))
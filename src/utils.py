import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    #return 1 if z > 0 else 0
    return np.where(z > 0, 1, 0)

def softmax(z: np.ndarray) -> np.ndarray:
    res = 0
    if z.ndim == 1:
        e_z = np.exp(z - np.max(z))
        res = e_z / e_z.sum()
    else:
        e_z = np.exp(z - np.max(z, azis=1, keepdims=True))
        res = e_z / e_z.sum(azis=1, keepdims=True)
        
    return res

print(softmax(np.array([1.0, 2.0, 3.0])))
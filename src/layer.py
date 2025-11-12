import numpy as np
from typing import Tuple 

class Layer:

    """
    Represents a layer in a neural network with its weights and biases 

    Args: 
        input_size (int): Number of input features to the layer
        output_size (int): Number of neurons in the layer
        random_seed (int, optional): Seed for random number generator for reproducibility. Defaults to 42.
    
    """

    def __init__(self, input_size: int, output_size: int, random_seed: int = 42):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = None
        self.biases = None 

        self._initialize_parameters()

        self.input_cache = None # Pour la rétropropagation 

    def _initialize_parameters(self):
        limit = np.sqrt(6.0 / (self.input_size + self.output_size))
        self.weights = np.random.uniform(-limit, limit, size=(self.input_size, self.output_size))

        self.biases = np.zeros((1, self.output_size))

    def forward(self, X : np.ndarray) -> np.ndarray: 
        """
        Perform the forward pass through the layer

        Args:
            X (np.ndarray): Input data of shape (m, input_size)

        Returns:
            np.ndarray: Output data of shape (m, output_size)
        """
        self.input_cache = X
        Z = np.dot(X, self.weights) + self.biases
        return Z

    def backward(self, dz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the backward pass through the layer

        Args:
            dz (np.ndarray): Gradient of the loss with respect to the output of the layer, shape (m, output_size)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - Gradient with respect to input data, shape (m, input_size)
                - Gradient with respect to weights, shape (input_size, output_size)
                - Gradient with respect to biases, shape (1, output_size)
        """

        X = self.input_cache

        m = X.shape[0]  # Number of samples

        dw = (X.T @ dz) * (1 / m)
        assert dw.shape == self.weights.shape, \
        f"dw shape {dw.shape} != W shape {self.weights.shape}"

        db = np.sum(dz, axis=0, keepdims=True) * (1 / m)
        assert db.shape == self.biases.shape, \
        f"db shape {db.shape} != b shape {self.biases.shape}"

        dx = dz @ self.weights.T
        assert dx.shape == X.shape, \
        f"dx shape {dx.shape} != X shape {X.shape}"

        return dx, dw, db
     
    def update_parameters(self, dw: np.ndarray, db: np.ndarray, learning_rate: float):
        """
        Update the layer's weights and biases using gradient descent

        Args:
            dw (np.ndarray): Gradient with respect to weights, shape (input_size, output_size)
            db (np.ndarray): Gradient with respect to biases, shape (1, output_size)
            learning_rate (float): Learning rate for the update
        """
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db




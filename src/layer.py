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
        np.random.seed(random_seed)
        self.input_size = input_size
        self.output_size = output_size

        self.weights = None
        self.biases = None 

        self._initialize_parameters()

        self.input_cache = None # Pour la rétropropagation 

    def _initialize_parameters(self):
        limit = np.sqrt(6.0 / (self.input_size + self.output_size))
        self.weights = np.random.uniform(-limit, limit, size=(self.input_size, self.output_size))
        print(self.weights)

        self.biases = np.zeros((self.input_size, self.output_size))
        print(self.biases)

    def forward(self, X : np.ndarray) -> np.ndarray: 
        """
        Perform the forward pass through the layer

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_size)

        Returns:
            np.ndarray: Output data of shape (num_samples, output_size)
        """
        self.input_cache = X
        output = np.dot(X, self.weights) + self.biases
        return output

    def backward(self, dz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the backward pass through the layer

        Args:
            dz (np.ndarray): Gradient of the loss with respect to the output of the layer, shape (num_samples, output_size)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - Gradient with respect to input data, shape (num_samples, input_size)
                - Gradient with respect to weights, shape (input_size, output_size)
                - Gradient with respect to biases, shape (1, output_size)
        """

        X = self.input_cache

        m = X.shape[0]  # Number of samples

        dw = np.dot(self.input_cache.T, dz) / num_samples
        assert dW.shape == self.weights.shape, \ 
        f"dW shape {dW.shape} != W shape {self.weights.shape}

        db = np.sum(dz, axis=0, keepdims=True) / num_samples
        dx = np.dot(dz, self.weights.T)

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

    

a = Layer(3,5)





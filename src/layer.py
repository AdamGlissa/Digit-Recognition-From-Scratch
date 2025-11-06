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

    def _initialize_parameters(self):
        limit = np.sqrt(6.0 / (self.input_size + self.output_size))
        self.weights = np.random.uniform(-limit, limit, size=(self.input_size, self.output_size))

        self.biases = np.zeros((self.input_size, self.output_size))

firstLayer = Layer(7,8)
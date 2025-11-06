import numpy as np
from typing import List, Tuple, Dict
from src.layer import Layer
from src.utils import relu, relu_derivative, softmax

class NeuralNetwork:
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01,
                 random_seed: int = 42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        np.random.seed(random_seed)
        
        # Créer les couches
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], random_seed))
        
        # Cache pour la rétropropagation
        self.activations = []
        self.z_values = []
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        self.z_values.append(
            self.layers[0].forward(X)
            )
        self.activations.append(
            relu(self.z_values[-1])
            )
        for layer in self.layers[1:-1]:
            self.z_values.append(
                layer.forward(self.activations[-1])
                )
            self.activations.append(
                relu(self.z_values[-1])
                )
        
        self.z_values.append(
            self.layers[-1].forward(self.activations[-1])
            )
        output = softmax(self.z_values[-1])
        
        return output
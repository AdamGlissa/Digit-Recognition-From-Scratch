import numpy as np
from typing import List, Tuple, Dict
from .layer import Layer
from ..utils.math_funtions import relu, relu_derivative, sigmoid, sigmoid_derivative

class NeuralNetwork:
    
    def __init__(self, layer_sizes: List[int], learning_rate,
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
        self.activations = []
        self.z_values = []

        # Clear caches for a fresh forward pass
        self.z_values = []
        self.activations = []

        # First hidden layer
        self.z_values.append(self.layers[0].forward(X))
        self.activations.append(relu(self.z_values[-1]))
        for layer in self.layers[1:-1]:
            self.z_values.append(
                layer.forward(self.activations[-1])
                )
            self.activations.append(
                relu(self.z_values[-1])
                )
        
        # Output layer
        self.z_values.append(self.layers[-1].forward(self.activations[-1]))
        y_final = sigmoid(self.z_values[-1])
        # store final activation for backward pass
        self.activations.append(y_final)
        return y_final
    
    def compute_loss(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        m = Y_true.shape[0]

        squarred_error = np.sum((Y_pred - Y_true) ** 2)
        loss = (1 / 2 ) * ( 1 / m ) * squarred_error

        return loss
    
    def backward_propagation(self, X: np.ndarray, Y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward propagation through the network
        
        Args:
            X (np.ndarray): Input data of shape (m, input_size)
            Y (np.ndarray): True labels of shape (m, 10) (one-hot encoded)

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: 
                - List of gradients with respect to weights for each layer
                - List of gradients with respect to biases for each layer
        """
        
        m = X.shape[0]

        gradients_W = []
        gradients_B = []

        # Gradient de la couche de sortie 

        Y_pred = self.activations[-1]
        dz = (Y_pred - Y) * sigmoid_derivative(self.z_values[-1])

        # Gradients de la boucle de rétropropagation

        for i in reversed(range(len(self.layers))):
            # Récupérer l'activation de la couche précédente
            if i == 0:
                prev_activation = X
            else:
                prev_activation = self.activations[i-1]

            dX, dW, dB = self.layers[i].backward(dz)

            gradients_W.insert(0, dW)
            gradients_B.insert(0, dB)

            if i > 0:
                Z_prev = self.z_values[i-1]
                dz = dX * relu_derivative(Z_prev)

        return gradients_W, gradients_B
    
    def update_parameters(self, gradients_W: List[np.ndarray], gradients_B: List[np.ndarray]):
        for i, layer in enumerate(self.layers):
            layer.update_parameters(gradients_W[i], gradients_B[i], self.learning_rate)

    def predict(self, X: np.ndarray, return_probs: bool = False):
        """
        Make predictions for input X.

        Args:
            X (np.ndarray): Input data of shape (m, input_size)
            return_probs (bool): If True, also return the probability matrix (soft outputs).

        Returns:
            If return_probs is False: array of predicted class indices (m,)
            If return_probs is True: (preds, probs) where probs has shape (m, n_classes)
        """
        Y_pred = self.forward_propagation(X)
        predictions = np.argmax(Y_pred, axis=1)
        if return_probs:
            return predictions, Y_pred
        return predictions
    
    def train(self, X_train: np.ndarray, Y_train: np.ndarray,
              epochs: int = 1000, batch_size: int = 32, 
               X_val: np.ndarray = None, y_val: np.ndarray = None ) -> dict:
    
        """
        Train the neural network using mini-batch gradient descent
        
        Args:
            X_train (np.ndarray): Training input data of shape (m, input_size)
            Y_train (np.ndarray): Training true labels of shape (m, 10) (one-hot encoded)
            epochs (int): Number of epochs to train
            batch_size (int): Size of each mini-batch
            X_val (np.ndarray, optional): Validation input data
            y_val (np.ndarray, optional): Validation true labels
            
        Returns:
            dict: Training history containing loss values
        """

        # Initialize history

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }

        # Number of training samples

        m = X_train.shape[0]

        n_batches = int(np.ceil(m / batch_size))

        print(f"=== Début de l'entraînement ===")
        print(f"Exemples d'entraînement: {m}")
        print(f"Batch size: {batch_size}")
        print(f"Batches par epoch: {n_batches}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.learning_rate}\n")

        # =============================================
        #                TRAINING LOOP 
        # =============================================
        

        for epoch in range(epochs):
            
            # Melange des données 

            indices = np.arange(m)
            np.random.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = Y_train[indices]

            epoch_loss = 0 

            # ----------------------------------------
            #      Mini-batch gradient descent 
            # ----------------------------------------

            for batch_idx in range(n_batches):

                # Calcul des indices de début et de fin de batch 
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, m)

                # Extraire le mini-batch 
                X_batch = X_train_shuffled[start_idx:end_idx]
                Y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward propagation 
                Y_pred = self.forward_propagation(X_batch)

                # Calcul de la perte
                batch_loss = self.compute_loss(Y_batch, Y_pred)
                epoch_loss += batch_loss

                # Backward propagation
                gradients_W, gradients_B = self.backward_propagation(X_batch, Y_batch)

                # Mettre à jour les paramètres
                self.update_parameters(gradients_W, gradients_B)
            
            # Calculer la perte moyenne de l'epoch
            avg_train_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_train_loss)

            
            # ----------------------------------------
            #   Evaluation sur le set de validation 
            # ----------------------------------------

            if X_val is not None and y_val is not None:

                # Forward propagation sur le set de validation
                Y_val_pred = self.forward_propagation(X_val)

                # Calcul de la perte de validation
                val_loss = self.compute_loss(y_val, Y_val_pred)
                history["val_loss"].append(val_loss)

                # Calcul de l'accuracy de validation
                val_accuracy = self.evaluate(X_val, y_val)
                history["val_accuracy"].append(val_accuracy)

                # Affichage des métriques
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}")
    
        print("=== Fin de l'entraînement ===")
        return history
    
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y_pred = self.forward_propagation(X)

        predictions = self.predict(X)
        true_labels = np.argmax(Y, axis=1)

        accuracy = np.mean(predictions == true_labels)
        return accuracy


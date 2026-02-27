import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from src.models.neural_network import NeuralNetwork
from src.data.data_loader import load_and_prepare_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_training_history(history: dict, save_path='results/training_history.png'):

    fig, axes = plt.subplots(2, 1, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot de la loss
    axes[0].plot(epochs, history['train_loss'], 
                label='Train loss', linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs, history['val_loss'], 
                label='Validation Loss', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Perte', fontsize=12)
    axes[0].set_title('Évolution de la perte pendant l\'entraînement', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Marquer le minimum de val loss
    min_val_loss_epoch = np.argmin(history['val_loss']) + 1
    min_val_loss = np.min(history['val_loss'])
    axes[0].axvline(x=min_val_loss_epoch, color='red', linestyle='--', 
                   alpha=0.5, label=f'Best Validation Loss (Epoch {min_val_loss_epoch})')
    axes[0].legend(fontsize=11)

    # Graphique 2 : Accuracy
    axes[1].plot(epochs, history['val_accuracy'], 
                label='Validation Accuracy', linewidth=2, 
                marker='o', markersize=3, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Évolution de la précision sur le set de validation', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Marquer le maximum d'accuracy
    max_val_acc_epoch = np.argmax(history['val_accuracy']) + 1
    max_val_acc = np.max(history['val_accuracy'])
    axes[1].axvline(x=max_val_acc_epoch, color='red', linestyle='--', 
                   alpha=0.5, label=f'Best Validation Accuracy (Epoch {max_val_acc_epoch})')
    axes[1].axhline(y=max_val_acc, color='blue', linestyle=':', 
                   alpha=0.3, label=f'Max: {max_val_acc:.2%}')
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique sauvegardé : {save_path}")
    plt.show()

def print_training_summary(history: dict):

    print("\n" + "="*60)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*60)
    
    # Loss finale
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    # Meilleure val loss
    best_val_loss = np.min(history['val_loss'])
    best_val_loss_epoch = np.argmin(history['val_loss']) + 1
    
    # Meilleure accuracy
    best_val_acc = np.max(history['val_accuracy'])
    best_val_acc_epoch = np.argmax(history['val_accuracy']) + 1
    
    # Accuracy finale
    final_val_acc = history['val_accuracy'][-1]
    
    print(f"\n Loss finale :")
    print(f"   Train Loss : {final_train_loss:.4f}")
    print(f"   Val Loss   : {final_val_loss:.4f}")
    
    print(f"\n Meilleure Val Loss :")
    print(f"   {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
    
    print(f"\n Meilleure Val Accuracy :")
    print(f"   {best_val_acc:.2%} (Epoch {best_val_acc_epoch})")
    
    print(f"\n Accuracy finale :")
    print(f"   {final_val_acc:.2%}")

    if final_train_loss < final_val_loss * 0.5:
        print(f"\n Possible overfitting détecté")
        print(f"   (Train Loss beaucoup plus petite que Val Loss)")

def save_model_weights(nn: NeuralNetwork, filename='model_weights.npz'):

    weights_dict = {}

    for i, layer in enumerate(nn.layers):
        weights_dict[f'W{i}'] = layer.weights
        weights_dict[f'b{i}'] = layer.biases

    np.savez(filename, **weights_dict)
    print(f"\nPoids du modèle sauvegardés dans : {filename}")

def main():
    # Charger les données
    print("Chargement et préparation des données...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_and_prepare_data()

    print(f"\n Données chargées avec succès :")
    print(f"   Train      : {X_train.shape[0]} exemples")
    print(f"   Validation : {X_val.shape[0]} exemples")
    print(f"   Test       : {X_test.shape[0]} exemples")
    print(f"   Features   : {X_train.shape[1]} (pixels par image)")
    print(f"   Classes    : {Y_train.shape[1]} (chiffres 0-9)")

    # ==========================================
    #     Creation du réseau  de neurones 
    # ==========================================

    # Hyperparamètres
    architecture = [64, 128, 64, 10]
    learning_rate = 0.05
    epochs = 500
    batch_size = 32
    random_seed = 42

    print(f"\n Hyperparamètres du modèle :")
    print(f"   Architecture  : {architecture}")
    print(f"   Learning Rate : {learning_rate}")
    print(f"   Epochs        : {epochs}")
    print(f"   Batch Size    : {batch_size}")   

    #Créer le réseau de neurones
    nn = NeuralNetwork(layer_sizes=architecture,
                        learning_rate=learning_rate,
                        random_seed=random_seed)
    print(f"\n Réseau de neurones créé avec {len(nn.layers)} couches")

    # Calculer le nombre total de paramètres
    total_params = 0
    for i, layer in enumerate(nn.layers):
        layer_params = layer.weights.size + layer.biases.size
        total_params += layer_params
        print(f"   Couche {i} : {layer_params} paramètres")   
    
    print(f"\n Nombre total de paramètres : {total_params}\n")

    # ==========================================
    #          Entraînement du modèle 
    # ==========================================

    start_time = datetime.now()

    history = nn.train(X_train=X_train,
                        Y_train=Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        X_val=X_val,
                        y_val=Y_val)
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    print(f"\nDurée totale de l'entraînement : {training_duration:.2f} secondes")
    print(f"Durée totale de l'entraînement : {training_duration/60:.2f} minutes")

    # ==========================================
    #          Evaluation du modèle 
    # ==========================================

    train_acc= nn.evaluate(X_train, Y_train)
    val_acc = nn.evaluate(X_val, Y_val)
    test_acc = nn.evaluate(X_test, Y_test)

    print(f"\n Précision du modèle :")  
    print(f"   Train Accuracy      : {train_acc:.2%}")
    print(f"   Validation Accuracy : {val_acc:.2%}")
    print(f"   Test Accuracy       : {test_acc:.2%}")

    diff = train_acc - test_acc
    if diff > 0.1:
        print(f"\n Possible overfitting détecté")
        print(f"   (Train Accuracy beaucoup plus élevée que Test Accuracy)")

    # ==========================================

    # Résumé et visualisation
    print_training_summary(history)

    # Sauvegarde 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    weights_filename = f'models/model_weights_{timestamp}.npz'
    save_model_weights(nn, filename=weights_filename)

    plot_filename = f'results/training_history_{timestamp}.png'
    plot_training_history(history, save_path=plot_filename)

    history_filename = f'results/training_history_{timestamp}.npz'
    np.savez(history_filename,
             train_loss=history["train_loss"],
                val_loss=history["val_loss"],
                val_accuracy=history["val_accuracy"],
                train_acc=train_acc,
                val_acc=val_acc,
                test_acc=test_acc)

    print(f"\nHistorique d'entraînement sauvegardé dans : {history_filename}")

    print("\n=== Fin de l'entraînement ===\n")

    print(f"\n Fichiers sauvegardés :")
    print(f"   - Poids du modèle : {weights_filename}")
    print(f"   - Graphique      : {plot_filename}")
    print(f"   - Historique     : {history_filename}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Entraînement interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n Une erreur est survenue : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    
"""
evaluate.py
Script d'évaluation détaillée du réseau de neurones.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from src.models.neural_network import NeuralNetwork
from src.data.data_loader import load_and_prepare_data

# Racine du projet, déduite de l'emplacement de ce fichier (src/models/evaluate.py)
# pour que le script fonctionne quel que soit le répertoire de travail courant.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'


def load_model_weights(nn, filename):
    """
    Charger les poids sauvegardés dans le modèle.
    
    Args:
        nn: Instance du NeuralNetwork
        filename: Fichier .npz contenant les poids
    """
    data = np.load(filename)

    for i, layer in enumerate(nn.layers):
        W, b = data[f'W{i}'], data[f'b{i}']
        # Sans cette vérification, une architecture différente de celle utilisée à
        # l'entraînement passerait silencieusement ici et échouerait plus loin dans
        # un np.dot, avec une erreur de shape impossible à rattacher à sa cause.
        if W.shape != layer.weights.shape or b.shape != layer.biases.shape:
            raise ValueError(
                f"Architecture incompatible à la couche {i} : "
                f"le modèle attend W{layer.weights.shape} et b{layer.biases.shape}, "
                f"le fichier contient W{W.shape} et b{b.shape}. "
                f"Vérifiez que layer_sizes correspond à celui utilisé à l'entraînement."
            )
        layer.weights = W
        layer.biases = b

    print(f"✅ Poids chargés depuis : {filename}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=RESULTS_DIR / 'confusion_matrix.png'):
    """
    Afficher et sauvegarder la matrice de confusion.
    
    Args:
        y_true: Labels vrais (array 1D)
        y_pred: Prédictions (array 1D)
        class_names: Noms des classes
        save_path: Chemin pour sauvegarder
    """
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Créer la figure
    plt.figure(figsize=(12, 10))
    
    # Heatmap avec annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Nombre de prédictions'},
                linewidths=0.5, linecolor='gray')
    
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Classe Prédite', fontsize=12, fontweight='bold')
    plt.ylabel('Classe Réelle', fontsize=12, fontweight='bold')
    
    # Calculer l'accuracy par classe et l'afficher
    accuracies_per_class = cm.diagonal() / cm.sum(axis=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Matrice de confusion sauvegardée : {save_path}")
    plt.show()
    
    return cm


def analyze_per_class_performance(y_true, y_pred, class_names):
    """
    Analyser les performances par classe.
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        class_names: Noms des classes
    """
    print("\n" + "="*60)
    print("📊 ANALYSE PAR CLASSE")
    print("="*60)
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'Classe':<10} {'Précision':<12} {'Rappel':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        # True Positives, False Positives, False Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        # Métriques
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name:<10} {precision:<12.2%} {recall:<12.2%} {f1:<12.2%} {support:<10}")
    
    # Accuracy globale
    accuracy = np.trace(cm) / np.sum(cm)
    print("-" * 60)
    print(f"{'Accuracy globale':<34} {accuracy:.2%}")
    print("="*60)


def find_worst_predictions(nn, X, y, n_examples=10):
    """
    Trouver les exemples les plus mal prédits.
    
    Args:
        nn: Réseau de neurones
        X: Données d'entrée
        y: Labels one-hot
        n_examples: Nombre d'exemples à afficher
    
    Returns:
        indices: Indices des pires prédictions
    """
    # Prédictions avec probabilités
    predictions, probs = nn.predict(X, return_probs=True)
    y_true = np.argmax(y, axis=1)
    
    # Trouver les prédictions incorrectes
    incorrect_mask = predictions != y_true
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) == 0:
        print("\n🎉 Aucune erreur trouvée ! Toutes les prédictions sont correctes.")
        return np.array([])
    
    # Calculer la "confiance" dans la mauvaise prédiction
    confidences = []
    for idx in incorrect_indices:
        pred_class = predictions[idx]
        confidence = probs[idx, pred_class]
        confidences.append(confidence)
    
    # Trier par confiance décroissante (les plus "sûres" mais fausses)
    confidences = np.array(confidences)
    sorted_indices = np.argsort(confidences)[::-1]
    
    # Prendre les n pires
    worst_indices = incorrect_indices[sorted_indices[:n_examples]]
    
    return worst_indices


def visualize_predictions(nn, X, y, indices, save_path=RESULTS_DIR / 'worst_predictions.png'):
    """
    Visualiser les prédictions sur des exemples spécifiques.
    
    Args:
        nn: Réseau de neurones
        X: Données d'entrée
        y: Labels one-hot
        indices: Indices des exemples à visualiser
        save_path: Chemin pour sauvegarder
    """
    if len(indices) == 0:
        return
    
    n_examples = len(indices)
    predictions, probs = nn.predict(X[indices], return_probs=True)
    y_true = np.argmax(y[indices], axis=1)
    
    # Créer la grille de subplots
    n_cols = 5
    n_rows = int(np.ceil(n_examples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten() if n_examples > 1 else [axes]
    
    for i, (ax, idx) in enumerate(zip(axes[:n_examples], range(n_examples))):
        # Reshape l'image
        image = X[indices[idx]].reshape(8, 8)
        
        # Afficher l'image
        ax.imshow(image, cmap='gray')
        
        # Informations sur la prédiction
        pred_class = predictions[idx]
        true_class = y_true[idx]
        confidence = probs[idx, pred_class]
        true_confidence = probs[idx, true_class]
        
        # Titre avec couleur
        color = 'green' if pred_class == true_class else 'red'
        ax.set_title(f"Prédit: {pred_class} ({confidence:.1%})\n"
                    f"Vrai: {true_class} ({true_confidence:.1%})",
                    color=color, fontweight='bold', fontsize=10)
        ax.axis('off')
    
    # Cacher les axes vides
    for ax in axes[n_examples:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualisation sauvegardée : {save_path}")
    plt.show()


def analyze_confidence_distribution(nn, X, y, save_path=RESULTS_DIR / 'confidence_distribution.png'):
    """
    Analyser la distribution des confiances de prédiction.
    
    Args:
        nn: Réseau de neurones
        X: Données d'entrée
        y: Labels one-hot
        save_path: Chemin pour sauvegarder
    """
    predictions, probs = nn.predict(X, return_probs=True)
    y_true = np.argmax(y, axis=1)
    
    # Confiance dans la prédiction
    confidences = np.max(probs, axis=1)
    
    # Séparer correctes et incorrectes
    correct_mask = predictions == y_true
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    # Créer la figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme 1 : Toutes les prédictions
    axes[0].hist(correct_confidences, bins=30, alpha=0.7, 
                label=f'Correctes ({len(correct_confidences)})', color='green')
    axes[0].hist(incorrect_confidences, bins=30, alpha=0.7, 
                label=f'Incorrectes ({len(incorrect_confidences)})', color='red')
    axes[0].set_xlabel('Confiance', fontsize=12)
    axes[0].set_ylabel('Fréquence', fontsize=12)
    axes[0].set_title('Distribution des Confiances', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Histogramme 2 : Par intervalles de confiance
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    correct_counts = []
    incorrect_counts = []
    
    for i in range(len(bins)-1):
        mask_correct = (correct_confidences >= bins[i]) & (correct_confidences < bins[i+1])
        mask_incorrect = (incorrect_confidences >= bins[i]) & (incorrect_confidences < bins[i+1])
        correct_counts.append(np.sum(mask_correct))
        incorrect_counts.append(np.sum(mask_incorrect))
    
    x = np.arange(len(bin_labels))
    width = 0.35
    
    axes[1].bar(x - width/2, correct_counts, width, label='Correctes', color='green', alpha=0.7)
    axes[1].bar(x + width/2, incorrect_counts, width, label='Incorrectes', color='red', alpha=0.7)
    axes[1].set_xlabel('Intervalle de Confiance', fontsize=12)
    axes[1].set_ylabel('Nombre de Prédictions', fontsize=12)
    axes[1].set_title('Prédictions par Niveau de Confiance', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_labels)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Distribution des confiances sauvegardée : {save_path}")
    plt.show()
    
    # Statistiques
    print(f"\n📊 Statistiques de confiance :")
    print(f"   Confiance moyenne (correctes)   : {correct_confidences.mean():.2%}")
    print(f"   Confiance moyenne (incorrectes) : {incorrect_confidences.mean():.2%}")
    print(f"   Confiance min (correctes)       : {correct_confidences.min():.2%}")
    print(f"   Confiance max (incorrectes)     : {incorrect_confidences.max():.2%}")


def main():
    """
    Fonction principale d'évaluation.
    """
    print("="*60)
    print("🔍 ÉVALUATION DÉTAILLÉE DU MODÈLE")
    print("="*60)
    
    # ========================================
    # ÉTAPE 1 : Charger les données
    # ========================================
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n📁 Chargement des données...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()
    
    print(f"✅ Données chargées : {X_test.shape[0]} exemples de test")
    
    # ========================================
    # ÉTAPE 2 : Créer et charger le modèle
    # ========================================
    print("\n🧠 Création du réseau de neurones...")
    nn = NeuralNetwork(
        layer_sizes=[64, 128, 64, 10],
        learning_rate=0.05,
        random_seed=42
    )

    # Chercher le fichier de poids le plus récent dans models/
    weight_files = list(MODELS_DIR.glob('model_weights_*.npz'))

    if weight_files:
        latest_weights = max(weight_files, key=lambda p: p.stat().st_ctime)
        print(f"\n📥 Chargement des poids : {latest_weights}")
        load_model_weights(nn, latest_weights)
    else:
        print(f"\n⚠️  Aucun fichier de poids trouvé dans {MODELS_DIR}")
        print("   Veuillez d'abord entraîner le modèle avec main.py")
        return
    
    # ========================================
    # ÉTAPE 3 : Évaluation globale
    # ========================================
    print("\n" + "="*60)
    print("📊 ÉVALUATION GLOBALE")
    print("="*60)
    
    test_acc = nn.evaluate(X_test, y_test)
    print(f"\n🎯 Accuracy sur le test set : {test_acc:.2%}")
    
    # ========================================
    # ÉTAPE 4 : Matrice de confusion
    # ========================================
    print("\n🔍 Calcul de la matrice de confusion...")
    predictions = nn.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    class_names = [str(i) for i in range(10)]
    cm = plot_confusion_matrix(y_true, predictions, class_names)
    
    # ========================================
    # ÉTAPE 5 : Analyse par classe
    # ========================================
    analyze_per_class_performance(y_true, predictions, class_names)
    
    # ========================================
    # ÉTAPE 6 : Distribution des confiances
    # ========================================
    print("\n📊 Analyse de la distribution des confiances...")
    analyze_confidence_distribution(nn, X_test, y_test)
    
    # ========================================
    # ÉTAPE 7 : Pires prédictions
    # ========================================
    print("\n🔍 Identification des pires prédictions...")
    worst_indices = find_worst_predictions(nn, X_test, y_test, n_examples=10)
    
    if len(worst_indices) > 0:
        print(f"\n❌ {len(worst_indices)} exemples mal prédits trouvés")
        visualize_predictions(nn, X_test, y_test, worst_indices)
    
    # ========================================
    # ÉTAPE 8 : Rapport détaillé
    # ========================================
    print("\n" + "="*60)
    print("📋 RAPPORT DE CLASSIFICATION DÉTAILLÉ")
    print("="*60)
    print(classification_report(y_true, predictions, target_names=class_names))
    
    print("\n" + "="*60)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*60)
    print(f"\n📂 Fichiers générés dans {RESULTS_DIR} :")
    print("   - confusion_matrix.png")
    print("   - confidence_distribution.png")
    print("   - worst_predictions.png (si erreurs trouvées)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Évaluation interrompue par l'utilisateur.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Erreur lors de l'évaluation : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
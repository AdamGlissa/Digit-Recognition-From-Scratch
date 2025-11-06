import numpy as np 
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split    

def load_and_prepare_data(): 

    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"\n=== Dataset Information ===")
    print(f"Dataset : {X.shape[0]} images of {X.shape[1]} pixels each")
    print(f"Values of pixels range from {X.min()} to {X.max()}")

    X_normalized = X / 16.0

    print(f"After normalization, pixel values range from {X_normalized.min()} to {X_normalized.max()}")

    n = 10 
    y_one_hot = np.zeros((y.shape[0], n))
    for i in range(y.shape[0]):
        y_one_hot[i, y[i]] = 1.0
    

    X_temp, X_test, y_temp, y_test = train_test_split(X_normalized, y_one_hot, test_size=0.15, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=np.argmax(y_temp, axis=1))  # 0.1765 x 0.85 ≈ 0.15, so validation set is ~15% of total

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

load_and_prepare_data()     

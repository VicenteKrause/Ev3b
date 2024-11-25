# My Utility: auxiliary functions
import numpy as np
import pandas as pd

def load_config_files():
    """Load configuration files and parameters"""
    # Leer configuración SAE
    sae_config = pd.read_csv('config_sae.csv', header=None)
    sae_params = sae_config[0].values  # Leer como una lista de valores en una columna
    
    if len(sae_params) < 4:
        raise ValueError("El archivo config_sae.csv debe contener al menos 4 filas para los parámetros.")
    
    # Leer configuración Softmax
    softmax_config = pd.read_csv('config_softmax.csv', header=None)
    softmax_params = softmax_config[0].values  # Leer como una lista de valores en una columna
    
    if len(softmax_params) < 3:
        raise ValueError("El archivo config_softmax.csv debe contener al menos 3 filas para los parámetros.")
    
    # Leer índices de ganancia
    idx_gain = pd.read_csv('idx_igain.csv', header=None)
    gain_indices = idx_gain[0].values  # Leer como una lista de valores en la primera columna
    
    if len(gain_indices) == 0:
        raise ValueError("El archivo idx_igain.csv no puede estar vacío.")

    return {
        'sae_params': {
            'hidden_nodes': int(sae_params[0]),
            'epochs': int(sae_params[1]),
            'batch_size': int(sae_params[2]),
            'learning_rate': float(sae_params[3])
        },
        'softmax_params': {
            'epochs': int(softmax_params[0]),
            'batch_size': int(softmax_params[1]),
            'learning_rate': float(softmax_params[2])
        },
        'gain_indices': gain_indices - 1  # Convertir a índices base 0
    }



def load_data(file_path, gain_indices=None):
    """
    Load and preprocess data from CSV files
    
    Parameters:
    file_path: path to data file (dtrain.csv or dtest.csv)
    gain_indices: indices for feature selection based on idx_igain.csv
    
    Returns:
    tuple: (features, labels)
    """
    # Cargar datos
    data = pd.read_csv(file_path, header=None)
    print(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")  # Validar cantidad de datos cargados
    
    # Separar características y etiquetas
    X = data.iloc[:, :]  # Todas las columnas excepto la última
    y = data.iloc[:, ]   # Última columna como etiquetas
    
    # Aplicar selección de características si se proporcionan índices
    if gain_indices is not None:
        X = X.iloc[:, gain_indices]
    
    return X.values, y.values


def activation_function(x, activation='sigmoid', alpha=1.0, scale=1.67326324):
    """Apply the selected activation function"""
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'tanh':
        return np.tanh(x)
    elif activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'elu':
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    elif activation == 'selu':
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError("Activation function not supported")

def mtx_confusion(y_true, y_pred):
    """
    Calculate confusion matrix and metrics
    """
    # Convert predictions to binary using 0.5 as threshold
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_true = y_true.astype(int)
    
    # Calculate confusion matrix elements
    TP = np.sum((y_pred_binary == 1) & (y_true == 1))
    TN = np.sum((y_pred_binary == 0) & (y_true == 0))
    FP = np.sum((y_pred_binary == 1) & (y_true == 0))
    FN = np.sum((y_pred_binary == 0) & (y_true == 1))
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score
    }

def normalize_data(X):
    """Normalize features to [0,1] range"""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-10)
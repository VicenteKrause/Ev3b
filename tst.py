import numpy as np
import utility as ut

def forward_edl():
    """
    Forward pass of the ELM model for testing
    Returns:
    tuple: (predictions, metrics)
    """
    # Cargar configuraciones
    config = ut.load_config_files()
    gain_indices = config['gain_indices']
    
    # Cargar datos de prueba
    X_test, y_test = ut.load_data('dtest.csv', gain_indices)
    
    # Normalizar datos de prueba
    X_test = ut.normalize_data(X_test)
    
    # Cargar parámetros del modelo
    weights = np.load('weights.npy')
    bias = np.load('bias.npy')
    beta = np.load('beta.npy')
    
    # Calcular capa oculta para datos de prueba
    H_test = ut.activation_function(X_test @ weights + bias, activation='sigmoid')
    
    # Realizar predicciones
    predictions = H_test @ beta
    
    # Calcular métricas
    metrics = ut.mtx_confusion(y_test, predictions)
    
    return predictions, metrics

def main():
    predictions, metrics = forward_edl()
    
    print("\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {metrics['TP']}")
    print(f"True Negatives: {metrics['TN']}")
    print(f"False Positives: {metrics['FP']}")
    print(f"False Negatives: {metrics['FN']}")

if __name__ == '__main__':
    main()
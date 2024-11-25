# Extreme Deep Learning
import numpy as np
import pandas as pd
import utility as ut

def train_edl():
    """
    Train the ELM model using the specified configuration files
    Returns:
    tuple: (weights, bias, beta, training_cost)
    """
    # Cargar configuraciones
    config = ut.load_config_files()
    
    # Cargar datos de entrenamiento
    X_train, y_train = ut.load_data('dtrain.csv', config['gain_indices'])
    
    # Normalizar datos
    X_train = ut.normalize_data(X_train)
    
    # Parámetros del modelo
    n_hidden = config['sae_params']['hidden_nodes']
    n_features = X_train.shape[1]
    
    # Inicializar pesos y bias con semilla fija para reproducibilidad
    np.random.seed(42)
    weights = np.random.normal(size=(n_features, n_hidden))
    bias = np.random.normal(size=n_hidden)
    
    # Calcular capa oculta
    H = ut.activation_function(X_train @ weights + bias, activation='sigmoid')
    
    # Calcular pesos de salida usando pseudo-inversa
    beta = np.linalg.pinv(H) @ y_train
    
    # Calcular predicciones y métricas de entrenamiento
    y_pred = H @ beta
    metrics = ut.mtx_confusion(y_train, y_pred)
    training_cost = 1 - metrics['accuracy']
    
    # Guardar pesos y bias en archivos numpy
    np.save('weights.npy', weights)
    np.save('bias.npy', bias)
    np.save('beta.npy', beta)
    
    # Crear y guardar la matriz de confusión
    confusion_matrix = [
        [metrics['TP'], metrics['FP']],
        [metrics['FN'], metrics['TN']]
    ]
    df_confusion = pd.DataFrame(confusion_matrix, columns=['Positive', 'Negative'], index=['Actual Positive', 'Actual Negative'])
    df_confusion.to_csv('confusion.csv', index=True)
    
    # Guardar F1-Score en un archivo CSV
    df_fscore = pd.DataFrame([metrics['f1_score']], columns=['F1-Score'])
    df_fscore.to_csv('fscore.csv', index=False)
    
    # Imprimir resultados
    print("\nTraining Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Cost: {training_cost:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {metrics['TP']}")
    print(f"True Negatives: {metrics['TN']}")
    print(f"False Positives: {metrics['FP']}")
    print(f"False Negatives: {metrics['FN']}")
    
    return weights, bias, beta, training_cost

def main():
    weights, bias, beta, cost = train_edl()
    print(f"\nTraining completed successfully!")
    print(f"Model parameters saved to: weights.npy, bias.npy, beta.npy")

if __name__ == '__main__':
    main()

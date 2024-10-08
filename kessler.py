import numpy as np
import sklearn.datasets as skd

# Función para generar los vectores extendidos de Kessler
def generate_kessler_vectors(X, y, M):
    """
    X: Matriz de características (n_samples, n_features)
    y: Vector de etiquetas (n_samples,)
    M: Número de clases
    """
    n_samples, n_features = X.shape
    extended_X = []
    extended_y = []

    for i in range(n_samples):
        x_i = X[i]
        for j in range(M):
            if y[i] == j:
                for k in range(M):
                    if k != j:
                        # Crear vectores extendidos con ceros en todas partes menos en las posiciones de clase
                        x_ij = np.zeros((M, n_features))
                        x_ij[j] = x_i
                        x_ij[k] = -x_i
                        extended_X.append(x_ij.flatten())  # Aplanar la matriz para un vector de tamaño (M * n_features,)
                        extended_y.append(1)  # Clase correcta para este vector
            else:
                extended_y.append(-1)  # Clase incorrecta

    return np.array(extended_X), np.array(extended_y)

# Algoritmo del perceptrón
def perceptron(X, y, n_classes, n_iter=1000, learning_rate=0.01):
    """
    X: Matriz de características extendidas (n_samples_ext, n_features_ext)
    y: Vector de etiquetas extendido (n_samples_ext,)
    n_classes: Número de clases
    n_iter: Número de iteraciones del algoritmo
    learning_rate: Tasa de aprendizaje
    """
    n_samples, n_features = X.shape
    w = np.random.rand(n_features)  # Inicializar vector de pesos aleatoriamente

    for _ in range(n_iter):
        for i in range(n_samples):
            if y[i] * np.dot(w, X[i]) <= 0:  # Verificar si está mal clasificado
                w += learning_rate * y[i] * X[i]  # Actualizar los pesos
    return w

# Función para predecir las clases
def predict(X, w, M):
    """
    X: Matriz de características (n_samples, n_features)
    w: Vector de pesos entrenado
    M: Número de clases
    """
    n_samples, n_features = X.shape
    predictions = []

    for i in range(n_samples):
        scores = []
        for j in range(M):
            w_j = w[j*n_features:(j+1)*n_features]  # Extraer pesos para la clase j
            scores.append(np.dot(w_j, X[i]))
        predictions.append(np.argmax(scores))  # Predecir la clase con el mayor puntaje
    return np.array(predictions)

# Ejemplo de uso:
if __name__ == "__main__":
    
    data_array = skd.load_digits()

    # Obtener los datos como un array de NumPy
    digits = data_array.data  # Este ya es un array de NumPy

# Obtener los targets como un array de NumPy
    labels = data_array.target

    # Número de clases
    M = len(np.unique(labels))

    # Generar los vectores extendidos de Kessler
    X_ext, y_ext = generate_kessler_vectors(digits, labels, M)

    # Entrenar el perceptrón
    w = perceptron(X_ext, y_ext, n_classes=M)

    # Predecir las clases
    y_pred = predict(digits, w, M)

    print("Predicciones:", y_pred)
    print("Clases reales:", labels)
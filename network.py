import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.core.multiarray import array as array


class NeuralNetwork:
    def __init__(self, layers:np.array):
        """
        Inicializa la red neuronal y genera su estructura.
        :param layers: Un iterador donde cada elemento es el número de neuronas en esa capa.
        """
        self.graph = nx.DiGraph()  # Grafo dirigido
        self.edges = []
        self.layers = layers
        self._generate()  # Llama al método privado _generate
        self.activation_function = None

    def _generate(self):
        """
        Genera la estructura de la red neuronal.
        """
        current_node = 0
        for i, layer_size in enumerate(self.layers):
            for j in range(layer_size):
                self.graph.add_node(current_node + j, subset=i)
            
            if i < len(self.layers) - 1:
                layer1_size = self.layers[i]
                layer2_size = self.layers[i + 1]
                
                for j in range(layer1_size):
                    for k in range(layer2_size):
                        self.edges.append((current_node + j, current_node + layer1_size + k))
            
            current_node += layer_size
        
        self.graph.add_edges_from(self.edges)
    
    
    def plot(self, palette: str = 'summer', edge_label_pos: float = 0.8, activated_node = None ):
        """
        Visualizes the structure of the neural network, showing weights on the edges and thresholds on the nodes.
        
        :param palette: Color palette for connections between layers.
        :param edge_label_pos: Position of edge labels along the edges.
        """
        # Get color map based on the number of subsets (layers) in the graph
        num_subsets = len(set(nx.get_node_attributes(self.graph, 'subset').values()))
        cmap = plt.cm.get_cmap(palette, num_subsets - 1)
        
        # Generate node positions for multipartite layout
        pos = nx.multipartite_layout(self.graph, subset_key="subset")
        
        plt.figure(figsize=(8, 8))
        
        node_colors = ['white' if node != activated_node else 'red' for node in self.graph.nodes]
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, edgecolors='black', linewidths=1, node_size=800)
        
        current_edge = 0
        edges = list(self.graph.edges())  # Ensure `edges` is correctly defined
        
        # Draw edges and their labels
        for i in range(num_subsets - 1):
            layer1_size = sum(1 for _, attr in self.graph.nodes(data=True) if attr['subset'] == i)
            layer2_size = sum(1 for _, attr in self.graph.nodes(data=True) if attr['subset'] == i + 1)
            # Avoid division by zero for color indexing
            
            if num_subsets - 1 > 0:
                color_index = i / (num_subsets - 1)
            else:
                color_index = 0  
            
            color = cmap(color_index)  
            
            # Define the edges to draw for this layer
            edges_to_draw = edges[current_edge:current_edge + layer1_size * layer2_size]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges_to_draw, arrows=True, arrowstyle='->', edge_color=color)
            
            # Draw edge labels
            edge_labels = {}
            for u, v in edges_to_draw:
                weight = self.graph[u][v].get("weight", 0.0)
                
                # Convert weight to a float if it is an ndarray
                if isinstance(weight, np.ndarray):
                    weight = float(weight)
                
                edge_labels[(u, v)] = f'{weight:.2f}'
            
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, label_pos=edge_label_pos)
            
            current_edge += layer1_size * layer2_size
        
        # Draw node labels with thresholds
        threshold_labels = {}
        for node in self.graph.nodes:
            threshold_value = self.graph.nodes[node].get("threshold", 0)
            
            # Convert threshold to a scalar if it's an ndarray
            if isinstance(threshold_value, np.ndarray):
                threshold_value = float(threshold_value[0])
            
            threshold_labels[node] = f'{threshold_value:.2f}'
        
        nx.draw_networkx_labels(self.graph, pos, labels=threshold_labels, font_color='black', font_size=10)
        
        plt.show()

    def weights(self, weights=None, random=False, num = 0):
        """
        Asigna pesos a las aristas de la red neuronal.
        :param weights: Puede ser un diccionario donde las claves son tuplas (nodo1, nodo2) y los valores son los pesos.
        :param random: Booleano que indica si se deben generar pesos aleatorios si no se proporciona un diccionario.
        """
        all_edges = list(self.graph.edges)

        if isinstance(weights, dict):
            # Usar el diccionario de pesos proporcionado por el usuario
            weights_dict = weights
        elif random:
            # Generar pesos aleatorios
            weights_dict = {edge: np.random.uniform(-1, 1) for edge in all_edges}
        else:
            # Establecer todos los pesos en 0 por defecto
            weights_dict = {edge: num for edge in all_edges}

        nx.set_edge_attributes(self.graph, weights_dict, "weight")

    def thresholds(self, thresholds=None, random=False, num = 0):
        """
        Asigna umbrales a las neuronas en las capas ocultas y finales.
        :param thresholds: Puede ser un diccionario donde las claves son los nodos y los valores son los umbrales.
        :param random: Booleano que indica si se deben generar umbrales aleatorios si no se proporciona un diccionario.
        """
        hidden_and_output_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['subset'] > 0]

        if isinstance(thresholds, dict):
            # Usar el diccionario de umbrales proporcionado por el usuario
            thresholds_dict = thresholds
        elif random:
            # Generar umbrales aleatorios
            thresholds_dict = {node: np.random.uniform(-1, 1) for node in hidden_and_output_nodes}
        else:
            # Establecer todos los umbrales en 0 por defecto
            thresholds_dict = {node: num for node in hidden_and_output_nodes}

        nx.set_node_attributes(self.graph, thresholds_dict, name="threshold")

    
    def set_activation_function(self, activation_function):
        """
        Establece la función de activación para las neuronas de la red.
        :param activation_function: Un string que representa el nombre de la función de activación 
        ('sigmoid', 'relu', 'tanh', 'linear', 'step', 'bipolar' ).
        """
        # Diccionario de funciones de activación disponibles usando lambdas
        activation_functions = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'relu': lambda x: np.maximum(0, x),
            'tanh': lambda x: np.tanh(x),
            'linear': lambda x: x,
            'step': lambda x: 1 if x >= 0 else 0,
            'bipolar': lambda x: 1 if x >= 0 else -1,
            'skew' :  lambda x: 1 if x > 0.1 else (0 if -0.1 <= x <= 0.1 else -1)
        }

        # Verificar si el nombre proporcionado está en las funciones disponibles
        if activation_function.lower() in activation_functions:
            self.activation_function = activation_functions[activation_function.lower()]
        else:
            raise ValueError(f"Función de activación '{activation_function}' no es válida. Las opciones disponibles son: {list(activation_functions.keys())}")
        
        
    def evaluate(self, input_data):
        """
        Calcula la salida de la red neuronal dado un conjunto de entradas.
        :param input_data: Una lista o tupla de valores de entrada.
        :return: Una lista con los valores de salida de las neuronas de salida.
        """
        # Convertir la entrada en un array de numpy si no lo es
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        
        # Asegurar que input_data tenga la dimensión correcta
        input_layer_size = self.layers[0]
        if input_data.shape[0] != input_layer_size:
            raise ValueError(f"Se esperaban {input_layer_size} entradas, pero se recibieron {input_data.shape[0]}.")

        # Diccionario para almacenar los valores de salida de cada neurona
        outputs = {}

        # Asignar las entradas a las neuronas de la capa de entrada
        for i in range(input_layer_size):
            outputs[i] = input_data[i]

        # Recorrer cada capa, excepto la capa de entrada
        for layer_index in range(1, len(self.layers)):
            current_layer_nodes = [node for node, attr in self.graph.nodes(data=True) if attr['subset'] == layer_index]

            # Calcular la salida de cada neurona en la capa actual
            for node in current_layer_nodes:
                weighted_sum = 0

                # Calcular la suma ponderada de todas las entradas a esta neurona
                for predecessor in self.graph.predecessors(node):
                    weight = self.graph[predecessor][node].get('weight', 0)
                    weighted_sum += outputs[predecessor] * weight

                # Restar el umbral de la neurona actual
                threshold = self.graph.nodes[node].get('threshold', 0)
                weighted_sum -= threshold

                # Aplicar la función de activación
                outputs[node] = self.activation_function(weighted_sum)

        # Extraer los valores de salida de las neuronas de la última capa
        output_layer_nodes = [node for node, attr in self.graph.nodes(data=True) if attr['subset'] == len(self.layers) - 1]
        final_outputs = np.array([outputs[node] for node in output_layer_nodes])

        return final_outputs
    
    
    def fit(self, input_vector, verbose = False):
        """
        Método que representa el procesamiento de una sola capa.
        Utiliza los pesos y umbrales ya asignados y evalúa la salida.
        :param input_vector: Vector de entrada de cardinalidad igual a la capa de entrada.
        :return: Valor procesado después de aplicar la función de activación.
        """
        # Usar el método evaluate que ya tienes implementado para procesar el vector de entrada
        for input_pair in input_vector:
            output = self.evaluate(input_pair)
            if verbose:
                if output == 1:
                    self.plot(activated_node=len(input_pair))
                else:
                    self.plot(activated_node=None)
                
            
            print(f"Entrada: {input_pair}, Salida: {output}")
    


################################################################################################

class logic_and(NeuralNetwork):
    
    def __init__(self):
        super().__init__(np.array([2, 1])) 
        weights = {(0, 2): 1, (1, 2): 1}  
        self.weights(weights) 

        thresholds = {2: 2}
        self.thresholds(thresholds) 
        self.set_activation_function('step')
        

class logic_or(NeuralNetwork):
    
    def __init__(self):
        super().__init__(np.array([2,1]))
        weights = {(0,2): 1, (1,2): 1}
        self.weights(weights)
        
        thresholds = {2:1}
        self.thresholds(thresholds)
        self.set_activation_function('step')
        

class logic_not(NeuralNetwork):
    
    def __init__(self):
        super().__init__(np.array([1,1]))
        weights = {(0,1): -1}
        self.weights(weights)
        
        thresholds = {1:-0.5}
        self.thresholds(thresholds)
        self.set_activation_function('step')
        
        
class one_Layer(NeuralNetwork):
    
    def __init__(self, n: int):
        
        super().__init__(np.array([n, 1]))  # Una capa con n entradas y 1 salida
        
        # Inicializa pesos y umbrales de manera aleatoria
        self.weights(random=False)
        self.thresholds(random=False)
        self.dimension = n


    def train_perceptron(self, X, y, epochs=10, learning_rate=0.1, verbose=False):
        """
        Trains the perceptron using the basic learning algorithm, with early stopping if no updates are made during an epoch.
        :param X: List of input vectors (each row is an input vector).
        :param y: List of labels (expected output) for each input vector.
        :param epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            no_changes = True  # Initialize a flag to track if changes are made
            for i in range(len(X)):
                input_vector = X[i]
                expected_output = y[i]
                
                # Evaluate the current output of the perceptron
                output = self.evaluate(input_vector)
                
                # Calculate the error
                error = expected_output - output
                
                # Update the weights if there's an error
                if error != 0:
                    no_changes = False  # Set flag to false if changes are made
                    
                    for j in range(len(input_vector)):
                        self.graph[j][len(input_vector)]["weight"] += learning_rate * error * input_vector[j]
                    
                    # Update the threshold
                    threshold = self.graph.nodes[self.dimension].get("threshold", 0)
                    self.graph.nodes[self.dimension]["threshold"] = threshold - learning_rate * error
                    
                    if verbose:
                        self.plot(activated_node=len(input_vector))  # Update the plot with the last activated node
                
            # If no changes were made during this epoch, stop early
            if no_changes:
                print(f"Stopping early at epoch {epoch}")
                break


    def train_hebbian(self, X, y, decimal_precision=6):
        """
        Trains the neural network using the Hebbian learning rule.
        
        Parameters:
        X (ndarray): Training input data
        y (ndarray): Training output labels
        decimal_precision (int): The number of decimal places to round the weights to avoid floating-point precision errors.
        """
        
        for i in range(len(X)):
            input_vector = X[i]
            expected_output = y[i]
            
            # Update weights using Hebbian rule
            for j in range(len(input_vector)):
                self.graph[j][len(input_vector)]["weight"] += input_vector[j] * expected_output
                
                # Round the weight to avoid numerical issues
                self.graph[j][len(input_vector)]["weight"] = round(self.graph[j][len(input_vector)]["weight"], decimal_precision)
            
            # Update the threshold
            self.graph.nodes[self.dimension]["threshold"] -= expected_output
            
            # Round the threshold to avoid numerical issues
            self.graph.nodes[self.dimension]["threshold"] = round(self.graph.nodes[self.dimension]["threshold"], decimal_precision)

        print("Training completed with Hebbian rule.")
        

    def train_adaline(self, X, y, epochs=10, learning_rate=0.1, tolerance=1e-5):
        """
        Trains the neural network using the ADALINE learning rule with an early stopping condition.
        
        Parameters:
        X (ndarray): Training input data
        y (ndarray): Training output labels
        epochs (int): Number of training iterations
        learning_rate (float): The step size for weight updates
        tolerance (float): The minimum change in weight to continue training
        """
        for epoch in range(epochs):
            max_weight_change = 0  # Track the maximum weight change in this epoch
            for i in range(len(X)):
                input_vector = X[i]
                expected_output = y[i]
                
                # Compute the net input (weighted sum of inputs + threshold)
                net_input = self.evaluate(input_vector)
                
                # Compute the error
                error = expected_output - net_input
                
                # Update weights and threshold using the ADALINE rule
                for j in range(len(input_vector)):
                    weight_update = learning_rate * error * input_vector[j]
                    self.graph[j][len(input_vector)]["weight"] += weight_update
                    
                    # Track the maximum weight change
                    max_weight_change = max(max_weight_change, abs(weight_update))
                    
                # Update the threshold
                threshold_update = learning_rate * error
                self.graph.nodes[self.dimension]["threshold"] += threshold_update
                
                # Track the maximum threshold change
                max_weight_change = max(max_weight_change, abs(threshold_update))
            
            # If the maximum change in weights or threshold is smaller than the tolerance, stop early
            if max_weight_change < tolerance:
                print(f"Training stopped early at epoch {epoch} due to small weight change.")
                break


class One_layer_Multiclass:
    def __init__(self, n_features, n_classes, learning_rate=0.1, n_iter=1000, tol=1e-4):
        """
        Inicializa la red con Perceptrón y ADALINE usando la construcción de Kesler.
        :param n_features: Número de características de entrada.
        :param n_classes: Número de clases.
        :param learning_rate: Tasa de aprendizaje.
        :param n_iter: Número de iteraciones.
        :param tol: Tolerancia para la convergencia.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        self.weights = None  # Se inicializan más tarde

    def add_bias_term(self, X):
        """
        Añade una columna de unos (para el término de bias) a la matriz de entrada X.
        :param X: Matriz de entrada (n_samples, n_features)
        :return: Matriz extendida con una columna adicional de unos (n_samples, n_features + 1)
        """
        n_samples = X.shape[0]
        bias = np.ones((n_samples, 1))  # Crear una columna de unos
        X_extended = np.hstack((bias, X))  # Concatenar la columna de unos a X
        return X_extended

    def generate_kesler_matrix(self, X, y):
        """
        Genera la matriz de Kesler a partir de los datos originales y sus etiquetas.
        :param X: Matriz de datos de entrada (n_samples, n_features)
        :param y: Etiquetas de clase (n_samples,)
        :return: Matriz extendida de Kesler (n_samples * (num_classes - 1), n_features * num_classes + num_classes)
        """
        
        # Obtener el número de muestras y características
        n_samples, n_features = X.shape
        
        # Inicializar listas vacías para almacenar la matriz de Kesler y las etiquetas
        kesler_matrix = []
        kesler_labels = []
        
        # Extender las dimensiones de X añadiendo una columna de unos (para manejar el bias)
        X_extended = self.add_bias_term(X)  # Añade una columna de unos al final de X para cada muestra
        
        # Para cada muestra de datos (iteramos sobre todas las filas de X)
        for i in range(n_samples):
            xi = X_extended[i]  
            yi = y[i]           
            
            # Generar vectores de Kesler para cada clase incorrecta
            # Este bucle anidado crea vectores de comparación entre la clase correcta y las demás clases
            for j in range(self.n_classes):
                if j != yi:  # Solo creamos vectores para las clases incorrectas
                    # Crear un vector extendido de ceros, con dimensiones ((n_features + 1) * n_classes)
                    # (n_features + 1) es para incluir el bias, y se multiplica por n_classes porque el vector representa todas las clases
                    xij = np.zeros(((n_features + 1) * self.n_classes, ))  # Inicializamos el vector con ceros
                    
                    # Colocar el vector xi (extendido) en la posición de la clase correcta (yi)
                    # Esto coloca xi (con bias) en el bloque correspondiente a la clase correcta en el vector de Kesler
                    xij[yi * (n_features + 1):(yi + 1) * (n_features + 1)] = xi  # Añadimos xi en el bloque correspondiente a la clase correcta
                    
                    # Colocar -xi en la posición de la clase incorrecta (j)
                    # Esto coloca -xi (con bias) en el bloque correspondiente a la clase incorrecta en el vector de Kesler
                    xij[j * (n_features + 1):(j + 1) * (n_features + 1)] = -xi  # Añadimos -xi en el bloque de la clase incorrecta
                    
                    # Añadir el vector de Kesler a la matriz
                    kesler_matrix.append(xij)
                    
                    kesler_labels.append(1)
    
        return np.array(kesler_matrix), np.array(kesler_labels)

    def train_perceptron_kesler(self, X, y):
        """
        Entrena un Perceptrón utilizando la matriz de Kesler.
        :param X: Datos de entrada (n_samples, n_features)
        :param y: Etiquetas de clase (n_samples,)
        """
        # Generar la matriz extendida de Kesler
        X_kesler, y_kesler = self.generate_kesler_matrix(X, y)
        n_samples = X_kesler.shape[0]

        # Inicialización de los pesos
        self.weights = np.random.rand(X_kesler.shape[1])

        for epoch in range(self.n_iter):
            weight_changes = 0
            for i in range(n_samples):
                output = np.dot(self.weights, X_kesler[i])
                
                # Si la predicción es incorrecta (output <= 0), actualizamos los pesos
                if output <= 0:
                    delta_weight = self.learning_rate * X_kesler[i]
                    self.weights += delta_weight
                    weight_changes += np.sum(np.abs(delta_weight))

            if weight_changes < self.tol:
                print(f"Perceptrón convergió en la época {epoch + 1}")
                break

    def train_adaline_kesler(self, X, y):
        # Generar la matriz extendida de Kesler
        X_kesler, y_kesler = self.generate_kesler_matrix(X, y)
        n_samples = X_kesler.shape[0]

        # Inicialización de los pesos
        self.weights = np.random.rand(X_kesler.shape[1])

        for epoch in range(self.n_iter):
            weight_changes = 0
            for i in range(n_samples):
                output = np.dot(self.weights, X_kesler[i])
                
                # Limitar la salida para evitar overflow
                output = np.clip(output, -500, 500)  # Evitar valores demasiado grandes
                
                # Calcular el error
                error = y_kesler[i] - output
                
                # Actualización de los pesos
                delta_weight = self.learning_rate * error * X_kesler[i]
                self.weights += delta_weight
                weight_changes += np.sum(np.abs(delta_weight))

            # Verificar si hubo convergencia
            if weight_changes < self.tol:
                print(f"ADALINE convergió en la época {epoch + 1}")
                break

    def predict(self, X):
        """
        Realiza predicciones sobre nuevos datos utilizando los pesos entrenados.
        :param X: Nuevos datos de entrada (n_samples, n_features)
        :return: Predicciones (n_samples,)
        """
        n_samples, n_features = X.shape
        predictions = []
        
        for i in range(n_samples):
            xi = X[i]
            predicted_class = None
            max_output = -np.inf
            
            # Evaluar el producto punto para cada clase
            for j in range(self.n_classes):
                xij = np.zeros(((n_features + 1) * self.n_classes, ))  # Crear vector extendido
                xij[j * (n_features + 1):(j + 1) * (n_features + 1)] = np.append(1, xi)  # Asignar xi
                
                output = np.dot(self.weights, xij)
                if output > max_output:
                    max_output = output
                    predicted_class = j

            predictions.append(predicted_class)
        
        return np.array(predictions)


    def _normalize(self, X):
        # Normalizar los datos a media 0 y desviación estándar 1
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Evitar división por cero
        return (X - X_mean) / X_std
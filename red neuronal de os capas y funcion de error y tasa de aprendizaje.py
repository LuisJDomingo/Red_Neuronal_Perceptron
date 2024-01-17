import numpy as np

# Función sigmoide, usada como función de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide, usada en el backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Clase para la Red Neuronal
class RedNeuronal:
    def __init__(self, n_entradas, n_neuronas_capa_oculta, n_neuronas_capa_salida, tasa_aprendizaje=0.1):
        self.tasa_aprendizaje = tasa_aprendizaje
        
        # Inicialización de pesos y bias para la capa oculta
        
        self.pesos_capa_oculta = np.random.randn(n_entradas, n_neuronas_capa_oculta)
        self.bias_capa_oculta = np.random.randn(n_neuronas_capa_oculta)

        # Inicialización de pesos y bias para la capa de salida
        
        self.pesos_capa_salida = np.random.randn(n_neuronas_capa_oculta, n_neuronas_capa_salida)
        self.bias_capa_salida = np.random.randn(n_neuronas_capa_salida)

    def forward(self, entradas):
        
        # Paso hacia adelante a través de la capa oculta
        
        self.salida_capa_oculta = sigmoid(np.dot(entradas, self.pesos_capa_oculta) + self.bias_capa_oculta)

        # Paso hacia adelante a través de la capa de salida
        
        self.salida_red = sigmoid(np.dot(self.salida_capa_oculta, self.pesos_capa_salida) + self.bias_capa_salida)
        return self.salida_red

    def backpropagation(self, entradas, salidas_esperadas):
        
        # Forward pass
        
        self.forward(entradas)

        # Error en la capa de salida
        
        error = salidas_esperadas - self.salida_red

        # Calcular el error cuadrático medio
        
        error_mse = np.mean(np.square(error))

        d_pesos_salida = np.dot(self.salida_capa_oculta.T, 2 * error * sigmoid_derivative(self.salida_red))

        # Error en la capa oculta
        
        error_capa_oculta = np.dot(2 * error * sigmoid_derivative(self.salida_red), self.pesos_capa_salida.T)
        d_pesos_oculta = np.dot(entradas.T, error_capa_oculta * sigmoid_derivative(self.salida_capa_oculta))

        # Actualizar los pesos y bias
        
        self.pesos_capa_oculta += self.tasa_aprendizaje * d_pesos_oculta
        self.pesos_capa_salida += self.tasa_aprendizaje * d_pesos_salida
        self.bias_capa_oculta += self.tasa_aprendizaje * np.sum(error_capa_oculta * sigmoid_derivative(self.salida_capa_oculta), axis=0)
        self.bias_capa_salida += self.tasa_aprendizaje * np.sum(2 * error * sigmoid_derivative(self.salida_red), axis=0)

        return error_mse


# Ejemplo de uso
red = RedNeuronal(2, 4, 1, tasa_aprendizaje=0.1)

# Datos de entrenamiento

entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
salidas_esperadas = np.array([[0], [1], [1], [0]])

# Entrenamiento

for i in range(1000):  # Número de iteraciones de entrenamiento
    
    error_mse = red.backpropagation(entradas, salidas_esperadas)
    print(f'Iteración {i+1}, Tasa de Aprendizaje: {red.tasa_aprendizaje}, Error MSE: {error_mse}')
    print('Pesos de la Capa Oculta:\n', red.pesos_capa_oculta)
    print('Bias de la Capa Oculta:\n', red.bias_capa_oculta)
    print('Pesos de la Capa de Salida:\n', red.pesos_capa_salida)
    print('Bias de la Capa de Salida:\n', red.bias_capa_salida)

# Probar la red después del entrenamiento

print(red.forward(entradas))

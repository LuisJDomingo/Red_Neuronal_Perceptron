# Red_Neuronal_Perceptron
Código en Python para implementar una red neuronal básica

# Red Neuronal de Dos Capas en Python

Este proyecto es una implementación sencilla de una red neuronal en Python. Utiliza la función de activación sigmoide y su derivada para el proceso de backpropagation. El proyecto está diseñado para ser un punto de partida para aquellos que se inician en el aprendizaje automático y las redes neuronales.

## Características

- Función de activación sigmoide.
- Derivada de la función sigmoide para el backpropagation.
- Configuración de una red neuronal con una capa oculta.

## Requisitos

Para ejecutar este código, necesitarás Python y la biblioteca Numpy. Puedes instalar Numpy usando pip:

```
pip install numpy
```

## Uso

Para usar esta red neuronal, simplemente importa la clase `RedNeuronal` desde el script y crea una instancia de la red con los parámetros deseados: número de entradas, número de neuronas en la capa oculta, número de neuronas en la capa de salida, y la tasa de aprendizaje.

Aquí hay un ejemplo simple de cómo inicializar y utilizar la red:

```python
from red_neuronal import RedNeuronal

# Crear una instancia de la red
# Por ejemplo: 3 neuronas de entrada, 4 en la capa oculta, 1 en la capa de salida
red = RedNeuronal(3, 4, 1)

# Aquí puedes añadir tu código para entrenar la red con tus datos
```

## Contribuciones

Las contribuciones son siempre bienvenidas. Si tienes alguna sugerencia o mejora, por favor, crea un 'pull request' o abre un 'issue' en este repositorio.

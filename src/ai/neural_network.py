from dotenv import load_dotenv
import os

# Cargar las variables de entorno
load_dotenv()

# El resto del código de TensorFlow sigue igual
import tensorflow as tf
from tensorflow.keras import layers, models


# Aquí ya no necesitas establecer 'TF_ENABLE_ONEDNN_OPTS' en el código
class NeuralNetwork:
    def __init__(self, input_shape, output_shape):
        self.model = self._build_model(input_shape, output_shape)

    def _build_model(self, input_shape, output_shape):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Dense(64, activation='relu'))  # Capa 1
        model.add(layers.Dense(128, activation='relu'))  # Capa 2
        model.add(layers.Dense(output_shape, activation='softmax'))  # Capa de salida
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def predict(self, X):
        return self.model.predict(X)

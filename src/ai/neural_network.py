# src/ai/neural_network.py

import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore

class NeuralNetwork:
    def __init__(self, input_shape, output_shape):
        self.model = self._build_model(input_shape, output_shape)

    def _build_model(self, input_shape, output_shape):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Dense(128, activation='relu'))  # Capa 1
        model.add(layers.Dense(64, activation='relu'))   # Capa 2
        model.add(layers.Dense(output_shape, activation='softmax'))  # Capa de salida
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        model = tf.keras.models.load_model(path)
        nn = NeuralNetwork((None,), None)
        nn.model = model
        return nn

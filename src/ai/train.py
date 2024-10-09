# src/ai/train.py

import os
import sys
import tensorflow as tf  # Asegúrate de importar TensorFlow aquí

# Añadir el directorio src al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from db.database import connect_db
from src.ai.neural_network import NeuralNetwork

import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def obtener_interacciones():
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT message FROM interacciones")
        mensajes = cursor.fetchall()
        cursor.close()
        conn.close()
        return [mensaje[0] for mensaje in mensajes]
    else:
        print("❌ No se pudo conectar a la base de datos para obtener las interacciones.")
        return []

def preparar_datos(textos):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(textos)
    sequences = tokenizer.texts_to_sequences(textos)
    maxlen = max(len(seq) for seq in sequences)
    sequences_padded = pad_sequences(sequences, maxlen=maxlen, padding='post')

    # Guardar el tokenizer para uso posterior
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    return sequences_padded, tokenizer, maxlen

def train_model():
    textos = obtener_interacciones()
    if not textos:
        print("❌ No hay datos para entrenar el modelo.")
        return

    sequences_padded, tokenizer, maxlen = preparar_datos(textos)
    vocab_size = len(tokenizer.word_index) + 1

    # Crear los datos de entrada y salida (usaremos un modelo simple para ejemplo)
    X_train = sequences_padded
    y_train = sequences_padded  # En modelos avanzados, esto sería desplazado

    # Convertir y_train a categórico
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=vocab_size)

    # Crear y entrenar el modelo
    nn = NeuralNetwork(input_shape=(maxlen,), output_shape=vocab_size)
    nn.train(X_train, y_train, epochs=10, batch_size=32)

    # Guardar el modelo
    nn.save('models/modelo.h5')
    print("✅ Modelo entrenado y guardado correctamente.")

if __name__ == '__main__':
    train_model()

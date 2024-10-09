# src/ai/predict.py

import os
import sys

# Añadir el directorio src al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

import tensorflow as tf  # type: ignore
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from src.ai.neural_network import NeuralNetwork


# Cargar el modelo y el tokenizer una sola vez
model_path = 'models/modelo.h5'
tokenizer_path = 'models/tokenizer.pkl'

if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    maxlen = model.input_shape[1]
else:
    print("❌ El modelo o el tokenizer no existen. Por favor, entrena el modelo primero ejecutando train.py")
    model = None
    tokenizer = None
    maxlen = None

def predict_response(input_text):
    if model is None or tokenizer is None:
        return "El modelo no está entrenado aún. Por favor, entrena el modelo primero."

    # Preprocesar el texto de entrada
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=maxlen, padding='post')

    # Generar la predicción
    prediction = model.predict(input_seq)
    predicted_seq = np.argmax(prediction, axis=-1)

    # Convertir la secuencia predicha a texto
    response = ''
    for idx in predicted_seq[0]:
        if idx != 0:
            word = tokenizer.index_word.get(idx, '')
            response += word + ' '

    return response.strip()

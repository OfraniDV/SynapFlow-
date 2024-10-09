import os
import tensorflow as tf

def save_model(model, path='model/synapflow_model.h5'):
    if not os.path.exists('model'):
        os.makedirs('model')
    model.save(path)
    print(f'Modelo guardado en {path}')

def load_model(path='model/synapflow_model.h5'):
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        print(f'Modelo cargado desde {path}')
        return model
    else:
        print("No se encontró el modelo, entrena uno nuevo.")
        return None

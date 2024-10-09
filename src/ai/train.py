from ai.neural_network import NeuralNetwork
from ai.model import save_model
import numpy as np
from db.database import obtener_interacciones  # Importar la función que recupera los datos de la base de datos

# Función de preprocesamiento (debes adaptarla a tu caso de uso)
def preprocess_interaction(message):
    # Un ejemplo básico de preprocesamiento: convertir texto en una representación numérica
    return np.array([ord(char) for char in message])  # Convierte los caracteres a sus códigos numéricos

def train_ai():
    # Obtener las interacciones almacenadas en la base de datos
    interacciones = obtener_interacciones()
    
    if not interacciones:
        print("No hay suficientes datos de interacciones para entrenar el modelo.")
        return
    
    # Preprocesar los datos
    X_train = []
    y_train = []
    
    for interaccion in interacciones:
        # Verificar si interaccion es una tupla o un diccionario
        if isinstance(interaccion, dict):
            message = interaccion['message']
            chat_type = interaccion['chat_type']
        else:
            # Asumimos que es una tupla en el orden (mensaje, tipo_chat)
            message = interaccion[0]
            chat_type = interaccion[1]
        
        # Preprocesar el mensaje (transformar el texto en datos numéricos)
        input_data = preprocess_interaction(message)
        X_train.append(input_data)
        
        # Usar el tipo de chat como una posible etiqueta (privado = 1, grupo = 0)
        if chat_type == 'private':
            y_train.append([1])
        else:
            y_train.append([0])
    
    # Convertir los datos a numpy arrays para entrenar el modelo
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Inicializar la red neuronal
    input_shape = (X_train.shape[1],)  # El tamaño de la entrada depende de cómo preproceses los mensajes
    output_shape = 2  # Binario (1 = privado, 0 = grupo), o ajusta esto según tu caso
    nn = NeuralNetwork(input_shape, output_shape)
    
    # Entrenar la red neuronal
    nn.train(X_train, y_train, epochs=10)
    
    # Guardar el modelo entrenado
    save_model(nn.model)

if __name__ == '__main__':
    train_ai()

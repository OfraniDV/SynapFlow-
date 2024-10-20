import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import time
import pickle
from model import Conversar
from database import Database
from tqdm import tqdm  # Para mostrar el progreso de entrenamiento
from tensorflow.keras.callbacks import TensorBoard

# Configurar el logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Inicializar la base de datos
db = Database()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.callbacks import TensorBoard

def entrenar_modelo(conversar_model, epochs=10, batch_size=32):
    """Función para entrenar el modelo usando el método `train` dentro de la clase `Conversar`"""
    logger.info("Iniciando el proceso de entrenamiento...")
    
    # Llamamos al método train dentro de la clase Conversar
    conversar_model.train(epochs=epochs, batch_size=batch_size)
    
    # Si el entrenamiento fue exitoso, podemos guardar el modelo
    if conversar_model.is_trained:
        guardar_modelo(conversar_model)
    else:
        logger.error("El modelo no fue entrenado correctamente. No se guardará.")

def guardar_modelo(conversar_model):
    """Función para guardar el modelo y su tokenizer"""
    if not conversar_model.is_trained:
        logger.error("El modelo no ha sido entrenado correctamente. No se guardará.")
        return

    logger.info("Guardando el modelo y el tokenizer...")

    # Guardar el modelo entrenado
    conversar_model.model.save('conversational_model.keras')
    logger.info("Modelo guardado exitosamente en 'conversational_model.keras'.")

    # Guardar el tokenizador
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(conversar_model.tokenizer, f)
    logger.info("Tokenizador guardado exitosamente en 'tokenizer.pkl'.")


def main():
    """Función principal para inicializar el modelo y ejecutar el entrenamiento"""
    logger.info("Entrenando el modelo conversacional...")

    # Crear e inicializar el modelo conversacional
    conversar_model = Conversar(db)

    # Construir el modelo antes de entrenar
    conversar_model.build_model()

    # Entrenar el modelo con modularización y seguimiento de progreso
    entrenar_modelo(conversar_model, epochs=10, batch_size=32)

    logger.info("Modelo conversacional entrenado y guardado exitosamente.")



if __name__ == '__main__':
    main()

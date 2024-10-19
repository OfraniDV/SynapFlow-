# train_conversation.py
import logging
from model import Conversar
from database import Database

# Configurar el logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Inicializar la base de datos
db = Database()

def main():
    logger.info("Entrenando el modelo conversacional...")
    
    # Crear e inicializar el modelo conversacional
    conversar_model = Conversar(db)
    
    # Entrenar el modelo
    conversar_model.train()

    # Guardar el modelo entrenado
    conversar_model.save_model('conversational_model.pkl')
    
    logger.info("Modelo conversacional entrenado y guardado exitosamente.")

if __name__ == '__main__':
    main()

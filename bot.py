import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import logging
import asyncio
import importlib
from functools import partial
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

# Importar otros módulos del proyecto
from model import NumerologyModel, Conversar
from database import Database
from scheduler import start_scheduler  # Importar el scheduler

#Importar el Feedback
from commands.feedback import feedback  # Asegúrate de tener el archivo feedback.py en la carpeta commands

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
VIP_GROUP_ID = os.getenv('VIP_GROUP_ID') 

# Establecer el event loop en Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def load_commands(application, db, numerology_model, conversar_model):
    """Carga automáticamente los archivos de la carpeta 'commands' y los registra como comandos en el bot"""
    commands_dir = 'commands'
    
    for filename in os.listdir(commands_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            # Remover la extensión .py para importar como módulo
            command_name = filename[:-3]
            module_path = f"{commands_dir}.{command_name}"
            
            # Cargar el módulo dinámicamente
            module = importlib.import_module(module_path)
            
            # Verificar si el módulo tiene una función con el mismo nombre que el archivo (por convención)
            if hasattr(module, command_name):
                command_function = getattr(module, command_name)

                # Agregar 'feedback' a la lista de comandos que solo necesitan `db`
                if command_name in ['lsgroup', 'addgroup', 'delgroup', 'feedback']:
                    # Comandos que solo necesitan `db`
                    application.add_handler(CommandHandler(command_name, partial(command_function, db=db)))
                else:
                    # Comandos que necesitan `db`, `model`, y `conversar_model`
                    application.add_handler(CommandHandler(command_name, partial(command_function, db=db, model=numerology_model, conversar_model=conversar_model)))
                
                logger.info(f"Comando /{command_name} registrado exitosamente.")
            else:
                logger.warning(f"El archivo {filename} no tiene una función {command_name}. No se pudo registrar el comando.")


def register_message_handler(application, db, conversar_model, numerology_model):
    """Registra el MessageHandler para manejar mensajes de texto generales"""
    from commands.handle_message import handle_message
    # Registramos el handler de mensajes
    message_handler = MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        partial(handle_message, db=db, conversar_model=conversar_model, numerology_model=numerology_model)
    )
    application.add_handler(message_handler)
    logger.info("MessageHandler registrado para mensajes generales.")


def main():
    logger.info("Iniciando el bot...")

    # Inicializar la base de datos
    db = Database()

    # Inicializar el modelo de numerología
    numerology_model = NumerologyModel(db)

    # Verificar si el modelo de numerología ya está entrenado (comprobando si existe el archivo del modelo guardado)
    model_file = 'numerology_model.keras'  # Nombre correcto del modelo de numerología

    if os.path.exists(model_file):
        logger.info(f"Modelo de numerología preentrenado encontrado: {model_file}. Cargando el modelo...")
        numerology_model.load(model_file)  # Cargar el modelo desde el archivo
    else:
        logger.info("No se encontró un modelo de numerología preentrenado. Entrenando el modelo desde cero...")
        numerology_model.train()  # Entrenar el modelo
        numerology_model.save(model_file)  # Guardar el modelo después de entrenarlo

    # Inicializar y cargar el modelo de conversación
    conversar_model = Conversar(db)
    conversar_model.cargar_modelo()

    # Realizar ajuste fino del modelo conversacional al iniciar
    try:
        nuevos_datos = db.get_new_messages()  # Obtener nuevos mensajes de la base de datos
        if nuevos_datos:
            conversar_model.ajuste_fino(nuevos_datos)
            logger.info("Ajuste fino inicial del modelo conversacional completado exitosamente.")
        else:
            logger.info("No se encontraron nuevos datos para el ajuste fino inicial.")
    except Exception as e:
        logger.error(f"Error durante el ajuste fino inicial: {e}")

    # Realizar ajuste fino del modelo de numerología si hay nuevos datos
    try:
        nuevos_datos_numerologia = db.get_numerology_adjustments()  # Obtener nuevos datos específicos para numerología
        if nuevos_datos_numerologia:
            numerology_model.ajuste_fino(nuevos_datos_numerologia)  # Realizar el ajuste fino
            logger.info("Ajuste fino inicial del modelo de numerología completado exitosamente.")
        else:
            logger.info("No se encontraron nuevos datos para el ajuste fino de numerología.")
    except Exception as e:
        logger.error(f"Error durante el ajuste fino del modelo de numerología: {e}")

    # Crear la aplicación de Telegram
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Cargar los comandos dinámicamente desde la carpeta "commands"
    load_commands(application, db, numerology_model, conversar_model)

    # Registrar el MessageHandler para mensajes de texto generales
    register_message_handler(application, db, conversar_model, numerology_model)  # Se agregó numerology_model

    # Iniciar el scheduler para el reentrenamiento periódico
    start_scheduler(numerology_model, conversar_model)

    # Iniciar el bot
    application.run_polling()


if __name__ == '__main__':
    main()

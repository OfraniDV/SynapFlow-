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
from scheduler import start_scheduler




# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

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

                # Algunos comandos solo necesitan `db`, otros también `model`
                if command_name in ['lsgroup', 'addgroup', 'delgroup']:
                    # Comandos que solo necesitan `db`
                    application.add_handler(CommandHandler(command_name, partial(command_function, db=db)))
                else:
                    # Comandos que necesitan `db`, `model`, y `conversar_model`
                    application.add_handler(CommandHandler(command_name, partial(command_function, db=db, model=numerology_model, conversar_model=conversar_model)))
                
                logger.info(f"Comando /{command_name} registrado exitosamente.")
            else:
                logger.warning(f"El archivo {filename} no tiene una función {command_name}. No se pudo registrar el comando.")


def register_message_handler(application, db, conversar_model):
    """Registra el MessageHandler para manejar mensajes de texto generales"""
    from commands.handle_message import handle_message
    # Registramos el handler de mensajes
    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_message, db=db, conversar_model=conversar_model))
    application.add_handler(message_handler)
    logger.info("MessageHandler registrado para mensajes generales.")

def main():
    logger.info("Iniciando el bot...")

    # Inicializar la base de datos
    db = Database()

    # Inicializar el modelo de numerología
    numerology_model = NumerologyModel(db)
    numerology_model.train()

    # Inicializar y cargar el modelo de conversación
    conversar_model = Conversar(db)
    conversar_model.cargar_modelo()

    # Crear la aplicación de Telegram
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Cargar los comandos dinámicamente desde la carpeta "commands"
    load_commands(application, db, numerology_model, conversar_model)

    # Registrar el MessageHandler para mensajes de texto generales
    register_message_handler(application, db, conversar_model)

    # Iniciar el bot
    application.run_polling()

if __name__ == '__main__':
    main()

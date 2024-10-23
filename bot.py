#bot.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import logging
import asyncio
import importlib
from functools import partial
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import telegram
import traceback
from telegram import Update
from telegram.ext import CallbackContext

# Importar otros módulos del proyecto
from model import NumerologyModel, Conversar
from database import Database
from scheduler import start_scheduler  # Importar el scheduler

# Importar el Feedback
from commands.feedback import feedback  # Asegúrate de tener el archivo feedback.py en la carpeta commands

# Cargar variables de entorno
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
VIP_GROUP_ID = os.getenv('VIP_GROUP_ID')
CHANNEL_ERROR_ID = os.getenv('CHANNEL_ERROR_ID')  # ID del canal de error

# Inicializar el bot de Telegram
bot = telegram.Bot(token=BOT_TOKEN)

# Configuración del logger
import codecs

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(log_level)

# Manejador para la consola con codificación UTF-8
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
# Establecer la codificación a UTF-8
console_handler.stream = codecs.getwriter('utf-8')(console_handler.stream.buffer)

# Manejador para archivo de log con codificación UTF-8
file_handler = logging.FileHandler('bot_errors.log', mode='a', encoding='utf-8')
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# Agregar manejadores al logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info('Bot iniciado correctamente.')

# Función asíncrona para enviar errores a un canal de Telegram
async def notify_error_to_channel(error_message):
    try:
        await bot.send_message(chat_id=CHANNEL_ERROR_ID, text=f"🚨 Error en el bot:\n{error_message}")
        logger.info(f"Error notificado al canal de Telegram {CHANNEL_ERROR_ID}")
    except Exception as e:
        logger.error(f"Error al enviar notificación al canal: {e}")

# Manejador de errores de Telegram
async def error_handler(update: Update, context: CallbackContext):
    error_message = f"Error en el bot: {context.error}"
    
    # Registrar el error en los logs
    logger.error(f"Excepción no controlada: {context.error}", exc_info=True)
    
    # Enviar el error al canal de Telegram
    await notify_error_to_channel(error_message)

# Manejador global de excepciones
def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Formatear el error
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(f"Excepción no controlada: {error_msg}")
    
    # Enviar el error al canal de Telegram
    asyncio.run(notify_error_to_channel(error_msg))

# Registrar el manejador global de excepciones
sys.excepthook = global_exception_handler

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
            
            try:
                # Cargar el módulo dinámicamente
                module = importlib.import_module(module_path)
                
                # Verificar si el módulo tiene una función con el mismo nombre que el archivo (por convención)
                if hasattr(module, command_name):
                    command_function = getattr(module, command_name)

                    # Lista de comandos que solo necesitan `db`
                    commands_with_db_only = ['lsgroup', 'addgroup', 'delgroup', 'feedback', 'charada', 'queda']

                    if command_name in commands_with_db_only:
                        # Comandos que solo necesitan `db`
                        application.add_handler(CommandHandler(command_name, partial(command_function, db=db)))
                    else:
                        # Comandos que necesitan `db`, `numerology_model`, y `conversar_model`
                        application.add_handler(CommandHandler(command_name, partial(command_function, db=db, numerology_model=numerology_model, conversar_model=conversar_model)))

                    logger.info(f"Comando /{command_name} registrado exitosamente.")
                else:
                    logger.warning(f"El archivo {filename} no tiene una función '{command_name}'. No se pudo registrar el comando.")
            
            except ImportError as e:
                logger.error(f"Error al importar el módulo {module_path}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error al registrar el comando /{command_name} del archivo {filename}: {e}", exc_info=True)

def register_message_handler(application, db, conversar_model, numerology_model):
    """Registra el MessageHandler para manejar mensajes de texto generales"""
    from commands.handle_message import handle_message
    
    # Utilizamos partial para pasar los modelos y db como argumentos adicionales
    message_handler = MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        partial(handle_message, numerology_model=numerology_model, conversar_model=conversar_model, db=db)  # Usamos partial para pasar los argumentos
    )
    
    application.add_handler(message_handler)
    logger.info("MessageHandler registrado para mensajes generales.")

def main():
    # Iniciar el logger
    logger.info("🚀 [Main - bot.py] Iniciando el bot...")

    # Inicializar la base de datos
    logger.info("🗄️ [Main - bot.py] Inicializando la base de datos...")
    db = Database()

    # Inicializar el modelo de numerología
    logger.info("🔢 [Main - bot.py] Inicializando el modelo de numerología...")
    numerology_model = NumerologyModel(db)

    # Verificar si el modelo de numerología ya está entrenado
    numerology_model_file = 'numerology_model.keras'

    if os.path.exists(numerology_model_file):
        # Cargar el modelo de numerología si ya existe
        logger.info(f"🟢 [Main - bot.py] Modelo de numerología preentrenado encontrado: {numerology_model_file}. Cargando el modelo...")
        numerology_model.load(numerology_model_file)
    else:
        # Entrenar el modelo de numerología si no existe
        logger.info("🟡 [Main - bot.py] No se encontró un modelo de numerología preentrenado. Iniciando entrenamiento...")
        numerology_model.train()

    # Verificar nuevamente si el modelo de numerología está entrenado correctamente
    if not numerology_model.is_trained:
        logger.error("🔴 [Main - bot.py] El modelo de numerología no se pudo entrenar. Por favor, verifica los pasos de entrenamiento.")
        return

    # Inicializar el modelo de conversación
    logger.info("💬 [Main - bot.py] Inicializando el modelo de conversación...")
    conversar_model = Conversar(db)

    # Verificar si el modelo conversacional ya está entrenado
    conversar_model_file = 'conversational_model_conversar.keras'  # Actualizado

    if os.path.exists(conversar_model_file):
        # Cargar el modelo conversacional si ya existe
        logger.info(f"🟢 [Main - bot.py] Modelo conversacional preentrenado encontrado: {conversar_model_file}. Cargando el modelo...")
        conversar_model.cargar_modelo()
    else:
        # Entrenar el modelo conversacional si no existe
        logger.info("🟡 [Main - bot.py] No se encontró un modelo conversacional preentrenado. Iniciando entrenamiento...")
        conversar_model.train()

    # Verificar nuevamente si el modelo de conversación está entrenado correctamente
    if not conversar_model.is_trained:
        logger.error("🔴 [Main - bot.py] El modelo conversacional no se pudo entrenar o cargar. Por favor, verifica los pasos de entrenamiento.")
        return

    # Crear la aplicación de Telegram
    logger.info("🤖 [Main - bot.py] Creando la aplicación de Telegram...")
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Cargar los comandos dinámicamente desde la carpeta "commands"
    logger.info("📦 [Main - bot.py] Cargando comandos...")
    load_commands(application, db, numerology_model, conversar_model)

    # Registrar el MessageHandler para mensajes de texto generales
    logger.info("📨 [Main - bot.py] Registrando manejador de mensajes generales...")
    register_message_handler(application, db, conversar_model, numerology_model)

    # Agregar manejador de errores global
    logger.info("🛠️ [Main - bot.py] Agregando manejador de errores global...")
    application.add_error_handler(error_handler)

    # Iniciar el scheduler para el reentrenamiento periódico
    logger.info("⏰ [Main - bot.py] Iniciando el scheduler para reentrenamiento periódico...")
    start_scheduler(numerology_model, conversar_model)

    # Iniciar el bot
    logger.info("✅ [Main - bot.py] Iniciando el bot con polling...")
    application.run_polling()

if __name__ == '__main__':
    main()

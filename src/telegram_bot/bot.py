# bot.py

import os
import sys
import logging

# Configurar el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Importar las funciones necesarias desde el módulo de base de datos
from src.db.database import guardar_interaccion, crear_tablas, connect_db
# Importar la función predict_response desde el archivo predict
from src.ai.predict import predict_response
# Importar la función de limpieza de datos desde el preprocesador
from src.ai.data_preprocessor import limpiar_texto

# Cargar las variables de entorno
load_dotenv()

# Asignar un string vacío si el token no está configurado, para evitar problemas de tipo
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
if not TOKEN:
    raise ValueError("El token de Telegram no está configurado en el archivo .env")

OWNER_ID = int(os.getenv('OWNER_ID', '0'))  # Asigna 0 si OWNER_ID no está presente

# Crear una variable global para la conexión a la base de datos
db_conn = None

def iniciar_base_de_datos():
    global db_conn
    # Conectar a la base de datos y crear las tablas si no existen
    db_conn = connect_db()
    if db_conn:
        logger.info("Conexión a la base de datos establecida.")
        crear_tablas(db_conn)
        logger.info("Tablas creadas o ya existentes.")
    else:
        logger.error("No se pudo conectar a la base de datos para crear las tablas.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.from_user:
        if update.message.from_user.id == OWNER_ID:
            await update.message.reply_text("¡Hola! El bot está funcionando correctamente.")
        else:
            logger.info(f"Acceso denegado para el usuario: {update.message.from_user.id}")
    else:
        logger.info("El mensaje no contiene un usuario válido.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global db_conn  # Asegurarse de usar la conexión global a la base de datos
    if update.message and update.message.from_user and update.message.text:
        user_message = update.message.text
        user_id = update.message.from_user.id
        chat_type = update.message.chat.type

        # Limpiar el texto del mensaje recibido
        user_message_limpio = limpiar_texto(user_message)

        # Aprender del mensaje, sin importar de quién es o dónde está
        logger.info(f"Aprendiendo del mensaje de {user_id}: {user_message_limpio}")

        # Guardar la interacción en la base de datos usando la conexión global (db_conn)
        if db_conn:  # Verifica que la conexión global a la base de datos esté activa
            guardar_interaccion(db_conn, user_id, user_message_limpio, chat_type)
        else:
            logger.error("No se pudo guardar la interacción, la conexión a la base de datos no está disponible.")

        # Responder solo al owner y solo en chat privado
        if user_id == OWNER_ID and chat_type == 'private':
            logger.info(f"Respondiendo al owner {user_id}: {user_message_limpio}")
            response = predict_response(user_message_limpio)
            await update.message.reply_text(f"{response}")
        else:
            logger.info(f"Mensaje procesado pero no respondido. Usuario: {user_id}")
    else:
        logger.info("El mensaje no contiene un texto válido o no tiene un usuario asociado.")

def start_bot():
    # Iniciar la base de datos y crear tablas
    logger.info("Iniciando el bot de Telegram...")
    iniciar_base_de_datos()

    # Iniciar la aplicación de Telegram
    logger.info("Iniciando aplicación de Telegram...")
    application = ApplicationBuilder().token(TOKEN).build()

    # Comandos básicos
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    # Manejador de mensajes
    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    application.add_handler(message_handler)

    # Ejecutar bot
    application.run_polling()

if __name__ == '__main__':
    start_bot()

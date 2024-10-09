import os
import sys
import logging
import asyncio  # Asegúrate de importar asyncio

# Configurar el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Importar las funciones necesarias
from src.db.database import guardar_interaccion, crear_tablas, connect_db
from src.ai.train import train_model  # Importar la función de entrenamiento
from src.ai.predict import predict_response

# Cargar las variables de entorno
load_dotenv()

# Asigna un string vacío si el token no está configurado, para evitar problemas de tipo
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
if not TOKEN:
    raise ValueError("El token de Telegram no está configurado en el archivo .env")

OWNER_ID = int(os.getenv('OWNER_ID', '0'))  # Asigna 0 si OWNER_ID no está presente

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.from_user:
        if update.message.from_user.id == OWNER_ID:
            await update.message.reply_text("¡Hola! El bot está funcionando correctamente.")
        else:
            logger.info(f"Acceso denegado para el usuario: {update.message.from_user.id}")
    else:
        logger.info("El mensaje no contiene un usuario válido.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Procesar todos los mensajes para aprender de ellos
    if update.message and update.message.from_user and update.message.text:
        user_message = update.message.text
        user_id = update.message.from_user.id
        chat_type = update.message.chat.type

        # Aprender del mensaje, sin importar de quién es o dónde está
        logger.info(f"Aprendiendo del mensaje de {user_id}: {user_message}")

        # Guardar la interacción en la base de datos usando la conexión global (db_conn)
        if db_conn:  # Verifica que la conexión global a la base de datos esté activa
            guardar_interaccion(db_conn, user_id, user_message, chat_type)
        else:
            logger.error("No se pudo guardar la interacción, la conexión a la base de datos no está disponible.")

        # Responder solo al owner y solo en chat privado
        if user_id == OWNER_ID and chat_type == 'private':
            logger.info(f"Respondiendo al owner {user_id}: {user_message}")
            response = predict_response(user_message)
            await update.message.reply_text(f"{response}")
        else:
            logger.info(f"Mensaje procesado pero no respondido. Usuario: {user_id}")
    else:
        logger.info("El mensaje no contiene un texto válido o no tiene un usuario asociado.")

async def entrenar_modelo():
    try:
        train_model()
    except Exception as e:
        logger.error(f"Error durante el entrenamiento del modelo: {e}")

def iniciar_base_de_datos():
    # Conectar a la base de datos y crear las tablas si no existen
    conn = connect_db()
    if conn:
        crear_tablas(conn)
        conn.close()
        logger.info("Las tablas fueron creadas o ya existían.")
    else:
        logger.error("No se pudo conectar a la base de datos para crear las tablas.")

async def start_bot():
    # Iniciar la base de datos y crear tablas
    logger.info("Iniciando el bot de Telegram...")
    iniciar_base_de_datos()

    # Iniciar el modelo de entrenamiento
    logger.info("Iniciando el entrenamiento del modelo...")
    await entrenar_modelo()  # Usar await para asegurarse de que el entrenamiento ocurra antes de que el bot inicie

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
    asyncio.run(start_bot())  # Usar asyncio.run para asegurarse de que todo esté en el contexto asíncrono

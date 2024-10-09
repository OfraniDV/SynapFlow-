import os
import sys

# Añadir el directorio src del proyecto al sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from db.database import guardar_interaccion, crear_tablas, connect_db  # Importar las funciones necesarias

# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.predict import predict_response  # Importar desde la carpeta ai

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
            print("Acceso denegado para el usuario:", update.message.from_user.id)
    else:
        print("El mensaje no contiene un usuario válido.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Procesar todos los mensajes para aprender de ellos
    if update.message and update.message.from_user and update.message.text:
        user_message = update.message.text
        user_id = update.message.from_user.id
        chat_type = update.message.chat.type

        # Aprender del mensaje, sin importar de quién es o dónde está
        print(f"Aprendiendo del mensaje de {user_id}: {user_message}")

        # Guardar la interacción en la base de datos
        guardar_interaccion(user_id, user_message, chat_type)

        # Generar una predicción con la IA (esto es opcional si no se necesita respuesta inmediata)
        predict_response(user_message)

        # Responder solo al owner y solo en chat privado
        if user_id == OWNER_ID and chat_type == 'private':
            print(f"Respondiendo al owner {user_id}: {user_message}")
            response = predict_response(user_message)
            await update.message.reply_text(f"Respuesta IA: {response}")
        else:
            print(f"Mensaje procesado pero no respondido. Usuario: {user_id}")
    else:
        print("El mensaje no contiene un texto válido o no tiene un usuario asociado.")

def iniciar_base_de_datos():
    # Conectar a la base de datos y crear las tablas si no existen
    conn = connect_db()
    if conn:
        crear_tablas(conn)
        conn.close()
    else:
        print("❌ No se pudo conectar a la base de datos para crear las tablas.")

def start_bot():
    # Iniciar la base de datos y crear tablas
    print("Iniciando el bot de Telegram...")
    iniciar_base_de_datos()

    # Iniciar la aplicación de Telegram
    print("Iniciando aplicación de Telegram...")
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

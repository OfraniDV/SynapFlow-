#bot

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import pickle
import logging
import re
import asyncio
import tensorflow as tf  # Importar TensorFlow

from functools import partial
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from telegram.ext import MessageHandler, filters  # Importar MessageHandler y filters para los mensajes

from model import NumerologyModel
from model import Conversar  # Importar la clase del modelo conversacional

from database import Database
from scheduler import start_scheduler

# Establecer la política del event loop para Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configurar el logger con más detalles (DEBUG)
import logging

# Configuración básica del logging
logging.basicConfig(
    level=logging.DEBUG,  # Cambia a DEBUG para más detalles
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Esto mostrará los logs en consola
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
VIP_GROUP_ID = os.getenv('VIP_GROUP_ID')
print(f"El token del bot cargado es: {BOT_TOKEN}")

# Inicializar la base de datos
db = Database()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE, conversar_model: Conversar):
    """Función para manejar cualquier mensaje de texto y generar una respuesta"""
    user_message = update.message.text  # Capturar el mensaje del usuario
    logger.info(f"Mensaje recibido: {user_message}")

    # Verificar si el modelo conversacional está entrenado
    if not conversar_model.is_trained:
        logger.warning("El modelo conversacional no está entrenado o no se cargó correctamente.")
        await update.message.reply_text('El modelo conversacional no está disponible en este momento.')
        return

    # Generar una respuesta utilizando el modelo conversacional
    response = conversar_model.generate_response(user_message)
    
    # Enviar la respuesta al usuario
    await update.message.reply_text(f"🤖 Respuesta: {response}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE, model: NumerologyModel):
    user_id = update.message.from_user.id
    user_first_name = update.message.from_user.first_name  # Obtener el nombre del usuario
    chat_type = update.message.chat.type  # Verificar si es un chat privado o grupal

    logger.info(f"Comando /start recibido de {update.message.from_user.username} en {chat_type}")

    if chat_type == 'private':
        # Obtener el nombre del bot de manera dinámica
        bot_info = await context.bot.get_me()
        bot_name = bot_info.first_name

        # Responder de manera elegante solo en chats privados, incluyendo el nombre del usuario
        await update.message.reply_text(
            f"👋 ¡Hola, {user_first_name}! Bienvenido a **{bot_name}**, tu **asistente de inteligencia artificial** 🤖.\n\n"
            "Soy un bot especializado en **numerología** y **predicciones**. 🌟\n\n"
            "📢 **Desarrollado por @Odulami**. Si tienes dudas o necesitas más información, "
            "no dudes en ponerte en contacto con él. 📞\n\n"
            "⚠️ Este bot está reservado para un **grupo VIP** donde podrás hacer tus consultas. "
            "Si deseas más información, contacta con el desarrollador. 💬\n\n"
            f"¡Gracias por usar **{bot_name}**! 😊",
            parse_mode='Markdown'
        )
    else:
        # Si el comando /start es enviado en el grupo, no responder
        logger.info(f"Ignorando /start en el grupo {update.message.chat.title}")

async def synap(update: Update, context: ContextTypes.DEFAULT_TYPE, model: NumerologyModel):
    logger.info(f"Comando /synap recibido de {update.message.from_user.username}")

    # Verificar si el mensaje proviene del grupo VIP
    if VIP_GROUP_ID and str(update.effective_chat.id) != VIP_GROUP_ID:
        logger.warning("Acceso denegado: El comando /synap fue llamado fuera del grupo VIP")
        await update.message.reply_text('Este comando solo está disponible en el grupo VIP.', parse_mode='HTML')
        return

    user_input = ' '.join(context.args)
    if not user_input:
        logger.info("No se proporcionó entrada de usuario en el comando /synap")
        await update.message.reply_text('Por favor, proporciona un número en tu consulta.', parse_mode='HTML')
        return

    # Extraer el número de la entrada del usuario
    match = re.search(r'\b\d{1,2}\b', user_input)
    if not match:
        logger.warning("No se encontró un número válido en la entrada del usuario")
        await update.message.reply_text('No se encontró un número válido en tu consulta.', parse_mode='HTML')
        return
    input_number = int(match.group())
    logger.info(f"Número extraído: {input_number}")

    # Verificar si el modelo está entrenado antes de hacer predicciones
    if not model.is_trained:
        logger.warning("El modelo no está entrenado o no se cargó correctamente.")
        await update.message.reply_text('El modelo de predicciones no está disponible en este momento.', parse_mode='HTML')
        return

    # Obtener recomendaciones del modelo
    recommended_numbers = model.predict(input_number)
    if not recommended_numbers:
        logger.warning("No se obtuvieron recomendaciones del modelo.")
        await update.message.reply_text('No se pudieron generar recomendaciones en este momento.', parse_mode='HTML')
        return
    limited_recommendations = recommended_numbers[:10]
    logger.info(f"Recomendaciones generadas: {limited_recommendations}")

    # Obtener el nombre del bot de manera dinámica
    bot_info = await context.bot.get_me()
    bot_name = bot_info.first_name

    # Obtener el nombre del grupo de manera dinámica
    chat_title = update.effective_chat.title

    # Generar el mensaje VIP personalizado
    vip_message = model.create_vip_message(input_number)

    # Enviar el mensaje VIP al usuario, incluyendo el nombre del bot y el grupo en HTML
    try:
        response_message = (
            f"🎉✨ <b>Predicciones para usuarios VIP del bot {bot_name}</b> ✨🎉\n"
            f"{chat_title}\n\n"
            f"{vip_message}"
        )
        await update.message.reply_text(response_message, parse_mode='HTML')
        logger.info(f"Respuesta generada para el usuario: {response_message}")
    except Exception as e:
        logger.error(f"Error al generar la respuesta: {e}")
        await update.message.reply_text('Ocurrió un error al procesar tu consulta. Por favor, intenta nuevamente más tarde.')

async def conversar(update: Update, context: ContextTypes.DEFAULT_TYPE, conversar_model: Conversar):
    """Función para interactuar con el modelo conversacional"""
    user_message = update.message.text
    response = conversar_model.generate_response(user_message)
    
    await update.message.reply_text(f"🤖 Respuesta: {response}")

# En tu función main(), ajustada para que siempre entrene el modelo
def main():
    logger.info("Iniciando el bot...")

    # Inicializar la base de datos
    logger.info("Inicializando la base de datos...")
    db = Database()
    logger.info("Base de datos inicializada.")

    # Inicializar el modelo de numerología
    logger.info("Inicializando el modelo de numerología...")
    numerology_model = NumerologyModel(db)
    logger.info("Modelo de numerología inicializado.")
    
    # Inicializar y cargar el modelo de conversación
    logger.info("Cargando el modelo conversacional...")
    conversar_model = Conversar(db)
    try:
        conversar_model.model = tf.keras.models.load_model('conversational_model.keras')
        with open('tokenizer.pkl', 'rb') as f:
            conversar_model.tokenizer = pickle.load(f)
        conversar_model.is_trained = True
        logger.info("Modelo conversacional cargado exitosamente.")
    except Exception as e:
        logger.error(f"No se pudo cargar el modelo conversacional: {e}")
        return

    # Entrenar el modelo de numerología al iniciar
    logger.info("Entrenando el modelo de numerología...")
    numerology_model.train()
    
    if not numerology_model.is_trained:
        logger.error("No se pudo entrenar el modelo de numerología. El bot no puede iniciarse.")
        return

    logger.info("El modelo ha sido entrenado correctamente.")

    # Iniciar el scheduler para reentrenamiento periódico
    logger.info("Iniciando el scheduler para reentrenamiento periódico...")
    start_scheduler(numerology_model, conversar_model)
    logger.info("Scheduler iniciado.")

    # Crear la aplicación de Telegram
    logger.info("Iniciando la aplicación de Telegram...")
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    logger.info("Aplicación de Telegram creada.")

    # Agregar el MessageHandler para responder a cualquier mensaje de texto
    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_message, conversar_model=conversar_model))
    application.add_handler(message_handler)

    # Agregar comandos al bot
    application.add_handler(CommandHandler("start", partial(start, model=numerology_model)))
    application.add_handler(CommandHandler("synap", partial(synap, model=numerology_model)))
    
    # Iniciar el bot
    logger.info("Iniciando el bot...")
    application.run_polling()
    logger.info("Bot en funcionamiento.")


if __name__ == '__main__':
    main()

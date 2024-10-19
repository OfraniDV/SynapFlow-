#bot

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import logging
import re
import asyncio
from functools import partial
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from model import NumerologyModel
from database import Database
from scheduler import start_scheduler

# Establecer la política del event loop para Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configurar el logger con más detalles (DEBUG)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Cambiar a DEBUG para más detalles
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
VIP_GROUP_ID = os.getenv('VIP_GROUP_ID')

# Inicializar la base de datos
db = Database()

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

# En tu función main(), agrega el manejador
def main():
    logger.info("Iniciando el bot...")

    # Inicializar la base de datos
    db = Database()

    # Inicializar o cargar el modelo de numerología
    numerology_model = NumerologyModel(db)
    if not os.path.exists('numerology_model.keras'):
        logger.info("Modelo de numerología no encontrado. Entrenando el modelo...")
        numerology_model.train()
    else:
        logger.info("Cargando el modelo de numerología preentrenado...")
        numerology_model.load_model('numerology_model.keras')

    if not numerology_model.is_trained:
        logger.error("No se pudo entrenar ni cargar el modelo de numerología. El bot no puede iniciarse.")
        return

    # Iniciar el scheduler para reentrenamiento periódico
    start_scheduler(numerology_model)

    # Crear la aplicación de Telegram
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Agregar comandos y pasar el modelo como argumento parcial
    application.add_handler(CommandHandler("start", partial(start, model=numerology_model)))
    application.add_handler(CommandHandler("synap", partial(synap, model=numerology_model)))

    # Iniciar el bot
    application.run_polling()

    logger.info("Bot en funcionamiento...")

if __name__ == '__main__':
    main()

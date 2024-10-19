# bot.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)
from telegram.ext import MessageHandler, filters
from model import NumerologyModel
from database import Database
from scheduler import start_scheduler
from dotenv import load_dotenv
import re
import asyncio

# Configurar el logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Cambia a DEBUG para más detalles
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
VIP_GROUP_ID = os.getenv('VIP_GROUP_ID')

# Inicializar la base de datos y el modelo
db = Database()
model = NumerologyModel(db)  # Pasamos la instancia de la base de datos al modelo
model.train()  # Entrenar el modelo al iniciar


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Comando /start recibido de {update.message.from_user.username}")
    await update.message.reply_text('¡Hola! Soy synapflow, tu asistente de numerología.')

async def synap(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    # Almacenar la interacción en la base de datos
    user_id = update.message.from_user.id
    db.save_interaction(user_id, user_input, limited_recommendations)
    logger.info(f"Interacción guardada en la base de datos para el usuario {user_id}")

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
        await update.message.reply_text('Ocurrió un error al procesar tu consulta. Por favor, intenta nuevamente más tarde.', parse_mode='HTML')

async def insertnum(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Comando /insertnum recibido de {update.message.from_user.username}")

    # Verificar si el mensaje proviene del grupo VIP
    if VIP_GROUP_ID and str(update.effective_chat.id) != VIP_GROUP_ID:
        logger.warning("Acceso denegado: El comando /insertnum fue llamado fuera del grupo VIP")
        await update.message.reply_text('Este comando solo está disponible en el grupo VIP.')
        return

    user_input = ' '.join(context.args)
    if not user_input:
        logger.info("No se proporcionó fórmula para insertar en /insertnum")
        await update.message.reply_text('Por favor, proporciona la fórmula que deseas insertar.')
        return

    # Insertar la fórmula en la base de datos
    db.insert_formula(user_input)
    logger.info(f"Fórmula insertada en la base de datos: {user_input}")
    await update.message.reply_text('La fórmula ha sido insertada correctamente.')

    # Reentrenar el modelo con las nuevas fórmulas
    logger.info("Reentrenando el modelo con las nuevas fórmulas")
    model.train()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    user_id = update.message.from_user.id

    # Opcional: Extraer números del mensaje
    numbers = re.findall(r'\b\d{1,2}\b', user_input)
    if numbers:
        input_number = int(numbers[0])
        logger.info(f"Número extraído del mensaje: {input_number}")

        # Obtener recomendaciones del modelo
        recommended_numbers = model.predict(input_number)
        limited_recommendations = recommended_numbers[:10]
        logger.info(f"Recomendaciones generadas: {limited_recommendations}")

        # Guardar la interacción
        db.save_interaction(user_id, user_input, limited_recommendations)
        logger.info(f"Interacción guardada en la base de datos para el usuario {user_id}")

        # Enviar respuesta al usuario
        response_message = f"Tus recomendaciones son: {', '.join(map(str, limited_recommendations))}"
        await update.message.reply_text(response_message)
    else:
        # Si no hay números, puedes decidir si guardar la interacción o no
        logger.info("No se encontraron números en el mensaje.")
        # Opcionalmente, guardar la interacción sin recomendaciones
        db.save_interaction(user_id, user_input, recommendations=[])
        await update.message.reply_text("Por favor, envía un número para obtener recomendaciones.")

# En tu función main(), agrega el manejador
def main():
    logger.info("Iniciando el bot...")
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Inicializar la base de datos
    db = Database()

    # Inicializar el modelo de numerología
    numerology_model = NumerologyModel(db)

    # Iniciar el scheduler para reentrenamiento periódico
    start_scheduler(numerology_model)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("synap", synap))
    application.add_handler(CommandHandler("insertnum", insertnum))

    # Agrega el manejador de mensajes
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
    logger.info("Bot en funcionamiento...")

if __name__ == '__main__':
    main()

# commands/synap.py

import logging
import re
from telegram import Update
from telegram.ext import ContextTypes
from datetime import datetime
import os

logger = logging.getLogger(__name__)

async def synap(update: Update, context: ContextTypes.DEFAULT_TYPE, db, numerology_model):
    logger.info(f"Comando /synap recibido de {update.message.from_user.username}")

    # Obtener el ID del grupo VIP desde las variables de entorno
    VIP_GROUP_ID = os.getenv('VIP_GROUP_ID')

    if update.message.chat.type == 'private':
        logger.warning(f"El usuario {update.message.from_user.username} intentó usar el comando en un chat privado.")
        await update.message.reply_text(
            "⚠️ Este comando no está disponible en chats privados. "
            "Para recibir predicciones, debes estar en el **grupo VIP**.", 
            parse_mode='HTML'
        )
        return

    if VIP_GROUP_ID is None:
        logger.error("VIP_GROUP_ID no está configurado. Asegúrate de que la variable de entorno está correctamente configurada.")
        await update.message.reply_text("⚠️ Error: El ID del grupo VIP no está configurado.", parse_mode='HTML')
        return

    if str(update.message.chat_id) != VIP_GROUP_ID:
        logger.warning(f"El grupo con ID {update.message.chat_id} no está autorizado para usar el comando /synap.")
        await update.message.reply_text(
            "⚠️ Este comando solo está disponible en el **grupo VIP**. Para poder recibir predicciones, "
            "asegúrate de estar en el grupo correcto.", parse_mode='HTML'
        )
        return

    # Aquí puedes verificar si el grupo está desactivado en la base de datos
    grupo_activado = db.verificar_grupo_activado(update.message.chat_id)  # Esta función es un ejemplo
    if not grupo_activado:
        logger.info(f"El grupo {update.message.chat_id} está desactivado, usando el modelo local.")
        await update.message.reply_text("Este grupo está desactivado. Usando predicciones del modelo local.", parse_mode='HTML')
        return

    # Continuar con el proceso habitual si el grupo está activado
    # Verificar si el modelo de numerología está entrenado
    if not numerology_model.is_trained:
        await update.message.reply_text('⚠️ El modelo de predicciones no está disponible en este momento.', parse_mode='HTML')
        return

    user_input = ' '.join(context.args)
    if not user_input:
        await update.message.reply_text('Por favor, proporciona un número en tu consulta.', parse_mode='HTML')
        return

    match = re.search(r'\b\d{1,2}\b', user_input)
    if not match:
        await update.message.reply_text('No se encontró un número válido en tu consulta.', parse_mode='HTML')
        return

    input_number = int(match.group())

    try:
        vip_message = numerology_model.create_vip_message(input_number)
        if not vip_message:
            await update.message.reply_text('No se pudieron generar recomendaciones en este momento.', parse_mode='HTML')
            return
    except Exception as e:
        logger.error(f"Error al generar predicciones VIP: {e}")
        await update.message.reply_text('Ocurrió un error al generar las predicciones. Por favor, intenta más tarde.', parse_mode='HTML')
        return

    user_first_name = update.message.from_user.first_name
    group_name = update.message.chat.title if update.message.chat.type != 'private' else "VIP Group"
    bot_info = await context.bot.get_me()
    bot_name = bot_info.first_name
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    response_message = (
        f"<b>{group_name}</b>\n\n"
        f"{vip_message}\n"
        f"📅 <i>Fecha y hora de consulta: {current_time}</i>\n"
        f"🔮 <b>Desarrollado por @Odulami usando {bot_name}</b>"
    )

    await update.message.reply_text(response_message, parse_mode='HTML')
    logger.info(f"Predicciones VIP enviadas para el número {input_number}.")

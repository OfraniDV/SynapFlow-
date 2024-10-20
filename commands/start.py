# commands/start.py

import logging
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE, db, model, conversar_model):
    user_id = update.message.from_user.id
    user_first_name = update.message.from_user.first_name
    chat_type = update.message.chat.type

    logger.info(f"Comando /start recibido de {update.message.from_user.username} en {chat_type}")

    if chat_type == 'private':
        bot_info = await context.bot.get_me()
        bot_name = bot_info.first_name

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

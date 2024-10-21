# commands/feedback.py

from telegram import Update
from telegram.ext import ContextTypes
import logging

async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE, db):
    """Maneja la retroalimentación del usuario sobre las respuestas del bot."""
    logger = logging.getLogger(__name__)

    user_feedback = update.message.text.split(' ', 1)[1].lower() if len(update.message.text.split()) > 1 else ''

    if user_feedback in ['si', 'sí', 'positivo', 'buena']:
        feedback_type = 'positivo'
    elif user_feedback in ['no', 'negativo', 'malo']:
        feedback_type = 'negativo'
    else:
        await update.message.reply_text("Por favor, responde con 'si' o 'no' para indicar si la respuesta fue útil.")
        return

    # Guardar la retroalimentación en la base de datos
    try:
        db.save_feedback(update.message.from_user.id, feedback_type)
        logger.info(f"Retroalimentación guardada: Usuario {update.message.from_user.id} - {feedback_type}")
        await update.message.reply_text("¡Gracias por tu retroalimentación!")
    except Exception as e:
        logger.error(f"Error al guardar la retroalimentación: {e}")
        await update.message.reply_text("Hubo un error al guardar tu retroalimentación. Por favor, intenta nuevamente.")

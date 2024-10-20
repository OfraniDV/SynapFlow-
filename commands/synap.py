# commands/synap.py

import logging
import re
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

async def synap(update: Update, context: ContextTypes.DEFAULT_TYPE, db, model, conversar_model):
    logger.info(f"Comando /synap recibido de {update.message.from_user.username}")
    
    # Verificar si el modelo de numerología está entrenado
    if not model.is_trained:
        await update.message.reply_text('El modelo de predicciones no está disponible en este momento.', parse_mode='HTML')
        return

    # Obtener la entrada del usuario
    user_input = ' '.join(context.args)
    if not user_input:
        await update.message.reply_text('Por favor, proporciona un número en tu consulta.', parse_mode='HTML')
        return

    # Extraer el número de la entrada
    match = re.search(r'\b\d{1,2}\b', user_input)
    if not match:
        await update.message.reply_text('No se encontró un número válido en tu consulta.', parse_mode='HTML')
        return

    input_number = int(match.group())
    
    # Obtener las recomendaciones del modelo de numerología
    recommended_numbers = model.predict(input_number)
    if not recommended_numbers:
        await update.message.reply_text('No se pudieron generar recomendaciones en este momento.', parse_mode='HTML')
        return

    # Limitar las recomendaciones a los primeros 10
    limited_recommendations = recommended_numbers[:10]
    
    # Obtener el nombre del usuario y generar un mensaje más personalizado
    user_first_name = update.message.from_user.first_name
    
    # Verificar si el grupo está registrado en la base de datos (si es un grupo)
    if update.message.chat.type != 'private':
        group_id = update.message.chat_id
        if not db.is_group_registered(group_id):
            logger.warning(f"El grupo con ID {group_id} no está autorizado para usar el comando /synap.")
            await update.message.reply_text("Este grupo no está autorizado para usar este comando.", parse_mode='HTML')
            return
    
    # Generar el mensaje final con las recomendaciones
    response_message = (
        f"📊 ¡Hola, {user_first_name}! Aquí tienes tus recomendaciones basadas en el número {input_number}: "
        f"{', '.join(map(str, limited_recommendations))}\n\n"
        "Estas predicciones son generadas utilizando numerología avanzada. 🔮"
    )
    
    # Enviar las recomendaciones al usuario
    await update.message.reply_text(response_message, parse_mode='HTML')

    # Opcional: Agregar alguna lógica para almacenar la interacción en la base de datos si se desea
    logger.info(f"Recomendaciones enviadas: {limited_recommendations}")

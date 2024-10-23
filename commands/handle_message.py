# commands/handle_message.py

import logging
import os
import re
from telegram import Update
from telegram.ext import ContextTypes
from model import Conversar, NumerologyModel  # Importar las clases necesarias
from database import Database  # Importar la clase de base de datos

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE, numerology_model, conversar_model, db):
    """Función para manejar cualquier mensaje de texto y generar una respuesta solo si el grupo está autorizado"""
    logger = logging.getLogger(__name__)

    # Verificar si el update contiene un mensaje válido
    if update.message is None or update.message.text is None:
        logger.warning("El update no contiene un mensaje de texto válido.")
        return

    # Capturar el mensaje del usuario
    user_message = update.message.text
    chat_type = update.message.chat.type  # Verificar si es un chat privado o grupal
    group_id = update.message.chat_id  # Obtener el ID del grupo o chat
    user_first_name = update.message.from_user.first_name  # Obtener el nombre del usuario

    logger.info(f"Mensaje recibido: {user_message} en el grupo/chat con ID: {group_id}")

    # Obtener el alias del bot dinámicamente
    bot_me = await context.bot.get_me()  # Obtiene la información del bot
    bot_alias = f"@{bot_me.username}"  # Extrae el nombre de usuario

    logger.info(f"Alias del bot: {bot_alias}")

    # Verificar si el mensaje contiene el alias del bot
    if bot_alias not in user_message:
        logger.info(f"El mensaje no contiene el alias del bot {bot_alias}, no se responderá.")
        return  # Si no se menciona el alias, el bot no responde.

    # Verificar si es un chat privado
    if chat_type == 'private':
        logger.info("El mensaje proviene de un chat privado.")
        owner_id = os.getenv('OWNER_ID')
        owner_name = 'el desarrollador del bot'
        owner_profile_link = f"https://t.me/{owner_id}"

        response_message = (
            f"👋 <b>¡Hola {user_first_name}!</b>\n\n"
            "Soy un <b>asistente de inteligencia artificial</b> que está en constante aprendizaje. 🤖\n\n"
            "📌 Solo puedo responder dentro de <b>grupos autorizados</b>.\n"
            "🔒 Contacta con mi desarrollador si necesitas más ayuda.\n\n"
            f"<a href='{owner_profile_link}'>Perfil del desarrollador</a>."
        )

        await update.message.reply_text(response_message, parse_mode='HTML')
        return

    # Verificar si es un grupo autorizado
    if chat_type != 'private':
        if not db.is_group_registered(group_id):
            await update.message.reply_text("Este bot no está autorizado para responder en este grupo.")
            return

    def contains_number(text):
        return bool(re.search(r'\b(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|\d+)\b', text.lower()))

    if contains_number(user_message):
        # Extraer el primer número encontrado
        numbers = re.findall(r'\b\d+\b', user_message)
        if not numbers:
            number_texts = {"cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
            word_numbers = re.findall(r'\b(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b', user_message.lower())
            numbers = [str(number_texts[num]) for num in word_numbers]
        number = int(numbers[0]) if numbers else None

        # Generar respuesta del modelo de numerología
        try:
            predicted_numbers = numerology_model.predict(number)
            numerology_response = f"#Numerologiaenentrenamiento\nLos números derivados son: {', '.join(map(str, predicted_numbers))}."
        except Exception as e:
            logger.error(f"Error en predicción numerológica: {e}")
            numerology_response = "#Numerologiaenentrenamiento\nError al procesar el número proporcionado."

        # Respuesta del modelo local
        try:
            local_response = conversar_model.model_generate_response(user_message)
            final_local_response = f"#ModeloConversacionenentrenamiento\n{local_response}"
        except Exception as e:
            logger.error(f"Error en modelo local: {e}")
            final_local_response = "#ModeloConversacionenentrenamiento\nError al generar respuesta local."

        # Respuesta de GPT-4
        try:
            gpt_response = conversar_model.gpt4o_generate_response(user_message)
            final_gpt_response = f"#GPT4Entrenandomismodelos\n{gpt_response}"

            # Almacenar la respuesta de GPT-4 para el ajuste fino de ambos modelos
            conversar_model.almacenar_para_ajuste_fino(user_message, gpt_response)  # Guardar para ajustes finos
        except Exception as e:
            logger.error(f"Error en GPT-4: {e}")
            final_gpt_response = "#GPT4Entrenandomismodelos\nError al generar respuesta GPT-4."

        # Enviar todas las respuestas en mensajes separados
        await update.message.reply_text(numerology_response, parse_mode='HTML')
        await update.message.reply_text(final_local_response, parse_mode='HTML')
        await update.message.reply_text(final_gpt_response, parse_mode='HTML')

        return

    # Si no contiene números, proceder con el modelo conversacional
    if not conversar_model.is_trained:
        await update.message.reply_text('El modelo conversacional no está disponible en este momento.')
        return

    response = conversar_model.generate_response(user_message)
    await update.message.reply_text(f"🤖 <b>Respuesta:</b> {response}", parse_mode='HTML')

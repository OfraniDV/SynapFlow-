# commands/handle_message.py

import logging
import os
import re
from telegram import Update
from telegram.ext import ContextTypes
from model import Conversar, NumerologyModel  # Importar las clases necesarias
from database import Database  # Importar la clase de base de datos

# Inicializar la base de datos
db = Database()  # Esto puede cambiar según cómo estés configurando tu DB
conversar_model = Conversar(db)  # Tu modelo de conversación
numerology_model = NumerologyModel(db)  # Tu modelo de numerología

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE, numerology_model, conversar_model, db):
    """Función para manejar cualquier mensaje de texto y generar una respuesta solo si el grupo está autorizado"""
    logger = logging.getLogger(__name__)

    # Capturar el mensaje del usuario
    user_message = update.message.text
    chat_type = update.message.chat.type  # Verificar si es un chat privado o grupal
    group_id = update.message.chat_id  # Obtener el ID del grupo o chat
    user_first_name = update.message.from_user.first_name  # Obtener el nombre del usuario

    logger.info(f"Mensaje recibido: {user_message} en el grupo/chat con ID: {group_id}")

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
        logger.info(f"Verificando si el grupo con ID {group_id} está registrado en la base de datos.")
        if not db.is_group_registered(group_id):
            logger.warning(f"El grupo {group_id} no está registrado. Enviando mensaje de no autorizado.")
            await update.message.reply_text("Este bot no está autorizado para responder en este grupo.")
            return
        logger.info(f"El grupo {group_id} está registrado y autorizado para recibir respuestas.")


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
            logger.info(f"Procesando número: {number} para generar predicciones de numerología.")
            predicted_numbers = numerology_model.predict(number)
            logger.info(f"Números predichos por el modelo de numerología: {predicted_numbers}")
            numerology_response = f"#Numerologiaenentrenamiento\nLos números derivados son: {', '.join(map(str, predicted_numbers))}."
        except Exception as e:
            logger.error(f"Error en predicción numerológica para el número {number}: {e}")
            numerology_response = "#Numerologiaenentrenamiento\nError al procesar el número proporcionado."


        # Respuesta del modelo local
        try:
            logger.info(f"Generando respuesta local para el mensaje: {user_message}")
            local_response = conversar_model.model_generate_response(user_message)
            logger.info(f"Respuesta generada por el modelo local: {local_response}")
            final_local_response = f"#ModeloConversacionenentrenamiento\n{local_response}"
        except Exception as e:
            logger.error(f"Error al generar respuesta local para el mensaje {user_message}: {e}")
            final_local_response = "#ModeloConversacionenentrenamiento\nError al generar respuesta local."


        # Respuesta de GPT-4
        try:
            logger.info(f"Generando respuesta GPT-4 para el mensaje: {user_message}")
            gpt_response = conversar_model.gpt4o_generate_response(user_message)
            logger.info(f"Respuesta generada por GPT-4: {gpt_response}")
            final_gpt_response = f"#GPT4Entrenandomismodelos\n{gpt_response}"

            # Almacenar la respuesta de GPT-4 para el ajuste fino de ambos modelos
            logger.info(f"Almacenando respuesta de GPT-4 para ajuste fino.")
            conversar_model.almacenar_para_ajuste_fino(user_message, gpt_response)
        except Exception as e:
            logger.error(f"Error en GPT-4 al procesar el mensaje {user_message}: {e}")
            final_gpt_response = "#GPT4Entrenandomismodelos\nError al generar respuesta GPT-4."


        # Enviar todas las respuestas en mensajes separados
        await update.message.reply_text(numerology_response, parse_mode='HTML')
        await update.message.reply_text(final_local_response, parse_mode='HTML')
        await update.message.reply_text(final_gpt_response, parse_mode='HTML')

        return

    # Si no contiene números, proceder con el modelo conversacional
    if not conversar_model.is_trained:
        logger.warning(f"El modelo conversacional no está entrenado. No se puede generar respuesta para el mensaje: {user_message}")
        await update.message.reply_text('El modelo conversacional no está disponible en este momento.')
        return

    logger.info(f"Generando respuesta conversacional para el mensaje: {user_message}")
    response = conversar_model.generate_response(user_message)
    logger.info(f"Respuesta generada por el modelo conversacional: {response}")
    await update.message.reply_text(f"🤖 <b>Respuesta:</b> {response}", parse_mode='HTML')


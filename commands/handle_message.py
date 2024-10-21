# commands/handle_message.py

import logging
import os
import re
from telegram import Update
from telegram.ext import ContextTypes
from model import Conversar, NumerologyModel  # Importar las clases necesarias
from database import Database  # Importar la clase de base de datos

# Inicializar la base de datos (asegúrate de tenerla configurada)
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

        # Obtener el OWNER_ID desde el archivo .env
        owner_id = os.getenv('OWNER_ID')

        # Intentar obtener el firstname del owner desde logsfirewallids
        owner_name = None
        owner_alias = None
        try:
            owner_info = db.get_owner_info(owner_id)  # Suponiendo que tenemos esta función en database.py
            if owner_info:
                owner_name = owner_info.get('nombre')
                owner_alias = owner_info.get('alias')
        except Exception as e:
            logger.error(f"Error al obtener la información del propietario: {e}")

        # Si no se obtiene el nombre desde la tabla, usar una fallback
        owner_name = owner_name or 'el desarrollador del bot'
        owner_profile_link = f"https://t.me/{owner_alias}" if owner_alias else f"https://t.me/{owner_id}"

        # Crear el mensaje de respuesta para el chat privado utilizando HTML
        response_message = (
            f"👋 <b>¡Hola {user_first_name}!</b>\n\n"
            "Soy un <b>asistente de inteligencia artificial</b> que está en constante aprendizaje. 🤖\n\n"
            "📌 Lamentablemente, por motivos de <b>seguridad</b>, solo puedo responder dentro de <b>grupos autorizados</b>.\n"
            "🔒 Si necesitas mi ayuda, por favor contacta con mi desarrollador.\n\n"
            f"Puedes ponerte en contacto con <b>{owner_name}</b> haciendo clic aquí: "
            f"<a href='{owner_profile_link}'>Perfil del desarrollador</a>."
        )

        # Responder con el mensaje personalizado
        await update.message.reply_text(response_message, parse_mode='HTML')
        return

    # Verificar si es un grupo y si está registrado
    if chat_type != 'private':
        logger.info("El mensaje proviene de un grupo, verificando autorización...")

        # Verificar si el grupo está registrado en la base de datos
        if not db.is_group_registered(group_id):
            logger.warning(f"El grupo con ID {group_id} no está autorizado.")
            await update.message.reply_text("Este bot no está autorizado para responder en este grupo.")
            return
        else:
            logger.info(f"El grupo con ID {group_id} está autorizado.")

    # Función para detectar números en el mensaje
    def contains_number(text):
        # Buscar números en formato numérico y textual
        return bool(re.search(r'\b(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|\d+)\b', text.lower()))

    # Si el mensaje contiene números
    if contains_number(user_message):
        logger.info("El mensaje contiene números, procesando con el modelo de numerología.")

        # Extraer números del mensaje
        numbers = re.findall(r'\b\d+\b', user_message)
        if not numbers:
            # Si no se encuentran números en formato digitado, intenta extraer de palabras
            number_texts = {
                "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4,
                "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
                "diez": 10
            }
            # Encuentra todas las palabras numéricas
            word_numbers = re.findall(r'\b(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b', user_message.lower())
            # Convertir palabras numéricas a dígitos
            numbers = [str(number_texts[num]) for num in word_numbers]

        if not numbers:
            # Si aún no hay números, responder al usuario
            await update.message.reply_text("No se encontró un número válido en tu mensaje.")
            return

        # Convertir las cadenas de números a enteros
        try:
            # Aquí puedes decidir si procesar todos los números o solo el primero
            # Por simplicidad, procesaremos el primero
            number = int(numbers[0])
            predicted_numbers = numerology_model.predict(number)
        except ValueError as ve:
            logger.error(f"Error durante la predicción: {ve}")
            await update.message.reply_text("Hubo un error al procesar el número proporcionado.")
            return
        except Exception as e:
            logger.error(f"Error durante la predicción: {e}")
            await update.message.reply_text("Ocurrió un error inesperado al procesar tu solicitud.")
            return

        # Generar respuesta basada en la predicción
        response = f"Los números que se derivan de tu consulta son: {', '.join(map(str, predicted_numbers))}."

        # Aquí puedes añadir lógica adicional para más contexto o análisis
        # Luego, se usa el modelo conversacional para afinar la respuesta
        final_response = conversar_model.generate_response(response)

        # Enviar la respuesta al usuario
        await update.message.reply_text(final_response, parse_mode='HTML')
        return

    # Si no contiene números, proceder como antes
    # Verificar si el modelo conversacional está entrenado
    if not conversar_model.is_trained:
        logger.warning("El modelo conversacional no está entrenado o no se cargó correctamente.")
        await update.message.reply_text('El modelo conversacional no está disponible en este momento.')
        return

    # Generar una respuesta utilizando el modelo conversacional
    response = conversar_model.generate_response(user_message)
    
    # Enviar la respuesta al usuario
    await update.message.reply_text(f"🤖 <b>Respuesta:</b> {response}", parse_mode='HTML')


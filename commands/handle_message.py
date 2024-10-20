import logging
import os

async def handle_message(update, context, conversar_model, db):
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

    # Verificar si el modelo conversacional está entrenado
    if not conversar_model.is_trained:
        logger.warning("El modelo conversacional no está entrenado o no se cargó correctamente.")
        await update.message.reply_text('El modelo conversacional no está disponible en este momento.')
        return

    # Generar una respuesta utilizando el modelo conversacional
    response = conversar_model.generate_response(user_message)
    
    # Enviar la respuesta al usuario
    await update.message.reply_text(f"🤖 <b>Respuesta:</b> {response}", parse_mode='HTML')
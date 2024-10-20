# commands/addgroup.py

import logging
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

async def addgroup(update: Update, context: ContextTypes.DEFAULT_TYPE, db):
    """Comando para agregar un grupo a la base de datos."""
    group_id = update.effective_chat.id  # ID del grupo
    group_name = update.effective_chat.title  # Nombre del grupo
    group_type = update.effective_chat.type  # Tipo de chat (grupo o supergrupo)
    
    # Obtener el serial del grupo si es necesario (puedes usar cualquier lógica que quieras para generar un "serial")
    group_serial = f"GRP-{group_id}"  # Ejemplo de serial simple

    # Verificar si ya existe el grupo en la base de datos
    if db.is_group_registered(group_id):
        await update.message.reply_text(f"⚠️ El grupo {group_name} ya está registrado en la base de datos.")
        return

    try:
        # Agregar el grupo a la base de datos
        db.add_group(group_id, group_type, group_serial)
        
        # Enviar mensaje de éxito
        await update.message.reply_text(f"✅ El grupo {group_name} ha sido agregado exitosamente a la base de datos.")
        logger.info(f"Grupo {group_name} (ID: {group_id}) agregado exitosamente a la base de datos.")
    
    except Exception as e:
        # Enviar mensaje de error y loguear
        logger.error(f"Error al agregar grupo {group_name} (ID: {group_id}): {e}")
        await update.message.reply_text(f"❌ Error al agregar el grupo: {e}")

# commands/delgroup.py

import logging
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

async def delgroup(update: Update, context: ContextTypes.DEFAULT_TYPE, db):
    """Comando para eliminar un grupo de la base de datos."""
    group_id = update.effective_chat.id  # ID del grupo
    group_name = update.effective_chat.title  # Nombre del grupo
    
    # Verificar si el grupo está registrado en la base de datos
    if not db.is_group_registered(group_id):
        await update.message.reply_text(f"⚠️ El grupo {group_name} no está registrado en la base de datos.")
        return

    try:
        # Intentar eliminar el grupo de la base de datos
        db.delete_group(group_id)
        
        # Enviar mensaje de éxito
        await update.message.reply_text(f"✅ El grupo {group_name} ha sido eliminado exitosamente de la base de datos.")
        logger.info(f"Grupo {group_name} (ID: {group_id}) eliminado exitosamente de la base de datos.")
    
    except Exception as e:
        # Enviar mensaje de error y loguear
        logger.error(f"Error al eliminar el grupo {group_name} (ID: {group_id}): {e}")
        await update.message.reply_text(f"❌ Error al eliminar el grupo: {e}")

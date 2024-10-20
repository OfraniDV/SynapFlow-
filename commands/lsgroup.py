# commands/lsgroup.py

import logging
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

async def lsgroup(update: Update, context: ContextTypes.DEFAULT_TYPE, db):
    logger.info("Ejecutando el comando /lsgroup para listar grupos...")
    try:
        groups = db.get_groups()
        logger.debug(f"Grupos obtenidos: {groups}")
        
        if groups:
            group_list = "\n".join([f"ID: {group[0]}, Nombre: {group[1]}" for group in groups])
            await update.message.reply_text(f"📋 <b>Grupos registrados en la base de datos:</b>\n\n{group_list}", parse_mode='HTML')
            logger.info(f"{len(groups)} grupos listados exitosamente.")
        else:
            await update.message.reply_text("⚠️ No hay grupos registrados en la base de datos.")
            logger.info("No hay grupos registrados.")
    
    except Exception as e:
        logger.error(f"Error al listar grupos: {e}")
        await update.message.reply_text(f"❌ Error al listar grupos: {e}")


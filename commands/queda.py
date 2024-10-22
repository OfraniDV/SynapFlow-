import logging
from telegram import Update
from telegram.ext import ContextTypes

async def queda(update: Update, context: ContextTypes.DEFAULT_TYPE, db):
    """Comando para buscar significado de un número o una palabra clave en la charada."""
    logger = logging.getLogger(__name__)

    try:
        # Verificar que el usuario proporcionó un argumento
        if len(context.args) < 1:
            await update.message.reply_text("Uso incorrecto. Formato: /queda <número o palabra>")
            return

        # Obtener el argumento de búsqueda (número o palabra)
        busqueda = " ".join(context.args).strip()

        # Verificar si es un número
        if busqueda.isdigit():
            numero = int(busqueda)
            # Buscar los significado para el número en la tabla
            significado = db.get_significado_por_numero(numero)
            if significado:
                await update.message.reply_text(f"El número {numero} representa: {', '.join(significado)}.")
            else:
                await update.message.reply_text(f"No se encontraron significado para el número {numero}.")
        else:
            palabra_clave = busqueda.lower()
            # Buscar los números que contienen esa palabra en los significado
            coincidencias = db.buscar_numeros_por_significado(palabra_clave)
            if coincidencias:
                resultados = [f"{numero}: {', '.join(significado)}" for numero, significado in coincidencias]
                await update.message.reply_text(f"La palabra '{palabra_clave}' aparece en los siguientes números:\n" + "\n".join(resultados))
            else:
                await update.message.reply_text(f"No se encontraron coincidencias para la palabra '{palabra_clave}'.")
    
    except Exception as e:
        logger.error(f"Error en el comando /queda: {e}")
        await update.message.reply_text("Ocurrió un error al procesar la solicitud.")

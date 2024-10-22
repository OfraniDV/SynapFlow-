import logging
from telegram import Update
from telegram.ext import ContextTypes

async def charada(update: Update, context: ContextTypes.DEFAULT_TYPE, db):
    """Comando para agregar o actualizar números con sus significados en la charada."""
    logger = logging.getLogger(__name__)

    try:
        # Verificar que se pasaron argumentos suficientes
        if len(context.args) < 2:
            await update.message.reply_text("Uso incorrecto. Formato: /charada <número o palabra> <significados o números>")
            return

        # Separar números y palabras
        numeros = []
        palabras = []

        for arg in context.args:
            if arg.isdigit():
                numeros.append(int(arg))
            else:
                palabras.append(arg.strip(",. "))

        # Verificar que tenemos al menos un número y al menos una palabra
        if not numeros or not palabras:
            await update.message.reply_text("Por favor, proporciona al menos un número y una palabra.")
            return

        logger.info(f"Procesando números: {numeros} con significados: {palabras}")

        # Recorrer todos los números e intentar agregarles los significados
        for numero in numeros:
            # Obtener los significados existentes de la base de datos
            significados_existentes = db.get_significados_por_numero(numero)

            # Identificar significados que ya están presentes
            significados_repetidos = set(significados_existentes) & set(palabras)
            significados_a_agregar = set(palabras) - set(significados_existentes)

            # Si no hay nada nuevo que agregar, indicarlo
            if not significados_a_agregar:
                await update.message.reply_text(f"Todos los significados ya están presentes para el número {numero}.")
                logger.info(f"Todos los significados ya presentes para el número {numero}.")
                continue

            # Agregar los nuevos significados a la base de datos
            db.actualizar_charada(numero, list(significados_a_agregar))
            mensaje_confirmacion = f"Los siguientes significados se han añadido para el número {numero}: {' '.join(significados_a_agregar)}."

            # Si hubo significados repetidos, informarlo también
            if significados_repetidos:
                mensaje_confirmacion += f"\nLos siguientes significados ya existían: {' '.join(significados_repetidos)}."

            await update.message.reply_text(mensaje_confirmacion)

            # Log del éxito de la operación
            logger.info(f"Significados añadidos para el número {numero}: {', '.join(significados_a_agregar)}")
    
    except Exception as e:
        logger.error(f"Error en el comando /charada: {e}")
        await update.message.reply_text("Ocurrió un error al procesar la solicitud.")

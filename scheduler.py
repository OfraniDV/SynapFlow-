import schedule
import time
import threading
import logging

def start_scheduler(numerology_model):
    def realizar_ajuste_fino():
        try:
            logging.info("===== Iniciando ajuste fino del modelo de numerología =====")
            numerology_model.ajuste_fino()
            logging.info("✔️ Ajuste fino del modelo de numerología completado exitosamente.")
        except Exception as e:
            logging.error(f"❌ Error durante el ajuste fino del modelo de numerología: {e}")

    def run_scheduler():
        logging.info("✔️ El programador de tareas (scheduler) está en ejecución.")
        while True:
            schedule.run_pending()
            time.sleep(1)

    # Realizar ajuste fino inmediatamente cuando el bot se enciende
    logging.info("⚡ El bot ha iniciado, realizando ajuste fino del modelo de numerología...")
    realizar_ajuste_fino()

    # Programar el ajuste fino todos los días a las 2 AM
    logging.info("⏰ Programando el ajuste fino del modelo diariamente a las 2:00 AM.")
    schedule.every().day.at("02:00").do(realizar_ajuste_fino)

    # Iniciar el scheduler en un hilo separado
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    logging.info("✔️ Hilo del scheduler iniciado correctamente.")

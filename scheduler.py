# scheduler.py
import schedule
import time
import threading
import logging

def start_scheduler(numerology_model, conversar_model):
    def retrain_models():
        try:
            logging.info("===== Iniciando proceso de reentrenamiento de modelos =====")

            # Proceso de reentrenamiento del modelo de numerología
            logging.info("----- Iniciando reentrenamiento del modelo de numerología -----")
            numerology_model.train()
            logging.info("✔️ Modelo de numerología reentrenado exitosamente.")

            # Realizar ajuste fino en el modelo conversacional utilizando los datos almacenados
            logging.info("----- Iniciando proceso de ajuste fino del modelo conversacional -----")
            conversar_model.realizar_ajuste_fino()
            logging.info("✔️ Ajuste fino del modelo conversacional completado exitosamente.")

        except Exception as e:
            logging.error(f"❌ Error durante el reentrenamiento de los modelos: {e}")

    # Programar el reentrenamiento y ajuste fino todos los días a las 2 AM
    logging.info("⏰ Programando el reentrenamiento de los modelos diariamente a las 2:00 AM.")
    schedule.every().day.at("02:00").do(retrain_models)

    def run_scheduler():
        logging.info("✔️ El programador de tareas (scheduler) está en ejecución.")
        while True:
            schedule.run_pending()
            time.sleep(1)

    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # Para que se detenga cuando el programa principal termine
    scheduler_thread.start()
    logging.info("✔️ Hilo del scheduler iniciado correctamente.")

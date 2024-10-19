# scheduler.py

import schedule
import time
import threading
import logging

def start_scheduler(numerology_model):
    # Definir la función que se ejecutará periódicamente
    def retrain_model():
        try:
            numerology_model.train()
            logging.info("Modelo reentrenado exitosamente.")
        except Exception as e:
            logging.error(f"Error durante el reentrenamiento del modelo: {e}")

    # Programar el reentrenamiento cada 1h
    schedule.every(60).minutes.do(retrain_model)

    # Función que ejecuta el scheduler en un hilo separado
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    # Iniciar el scheduler en un hilo
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # Para que se detenga cuando el programa principal termine
    scheduler_thread.start()

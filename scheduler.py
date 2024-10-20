import schedule
import time
import threading
import logging

def start_scheduler(numerology_model, conversar_model):
    def retrain_models():
        try:
            # Reentrenar el modelo de numerología
            numerology_model.train()
            logging.info("Modelo de numerología reentrenado exitosamente.")
            
            # Realizar ajuste fino en el modelo conversacional con nuevos mensajes
            nuevos_datos = conversar_model.db.get_new_messages()
            if nuevos_datos:
                conversar_model.ajuste_fino(nuevos_datos)
                logging.info("Ajuste fino del modelo conversacional completado exitosamente.")
            else:
                logging.info("No se encontraron nuevos datos para ajuste fino.")
            
        except Exception as e:
            logging.error(f"Error durante el reentrenamiento de los modelos: {e}")

    # Programar el reentrenamiento y ajuste fino cada 1 hora
    schedule.every(60).minutes.do(retrain_models)

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # Para que se detenga cuando el programa principal termine
    scheduler_thread.start()


import schedule
import time
import threading
import logging

def start_scheduler(numerology_model, conversar_model):
    def retrain_models():
        try:
            # Reentrenar el modelo de numerología (esto puede seguir a mayor intervalo si no es crítico)
            numerology_model.train()
            logging.info("Modelo de numerología reentrenado exitosamente.")
            
            # Realizar ajuste fino en el modelo conversacional con nuevos mensajes
            nuevos_datos = conversar_model.db.get_new_messages()

            if nuevos_datos:
                # Procesar los datos en lotes de 10 mensajes
                for i in range(0, len(nuevos_datos), 10):
                    lote = nuevos_datos[i:i + 10]
                    conversar_model.ajuste_fino(lote)  # Ajuste fino con un lote de 10 interacciones
                logging.info(f"Ajuste fino del modelo conversacional completado exitosamente con {len(nuevos_datos)} mensajes.")
            else:
                logging.info("No se encontraron nuevos datos para ajuste fino.")
            
        except Exception as e:
            logging.error(f"Error durante el reentrenamiento de los modelos: {e}")

    # Programar el reentrenamiento y ajuste fino cada 5 minutos
    schedule.every(5).minutes.do(retrain_models)

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # Para que se detenga cuando el programa principal termine
    scheduler_thread.start()

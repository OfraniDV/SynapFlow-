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
            
            # Realizar ajuste fino en el modelo conversacional con nuevos mensajes
            logging.info("----- Iniciando proceso de ajuste fino del modelo conversacional -----")
            nuevos_datos = conversar_model.db.get_new_messages()

            if nuevos_datos:
                logging.info(f"Se encontraron {len(nuevos_datos)} nuevos mensajes para el ajuste fino del modelo conversacional.")
                # Procesar los datos en lotes de 10 mensajes
                for i in range(0, len(nuevos_datos), 10):
                    lote = nuevos_datos[i:i + 10]
                    logging.debug(f"Procesando lote de {len(lote)} mensajes: {lote}")
                    conversar_model.realizar_ajuste_fino(lote)  # Ajuste fino con un lote de 10 interacciones
                    logging.info(f"✔️ Lote de {len(lote)} mensajes procesado correctamente.")
                logging.info(f"✔️ Ajuste fino del modelo conversacional completado exitosamente con un total de {len(nuevos_datos)} mensajes.")
            else:
                logging.info("No se encontraron nuevos datos para ajuste fino del modelo conversacional.")
            
        except Exception as e:
            logging.error(f"❌ Error durante el reentrenamiento de los modelos: {e}")

    # Programar el reentrenamiento y ajuste fino cada 5 minutos
    logging.info("⏰ Programando el reentrenamiento de los modelos cada 5 minutos.")
    schedule.every(5).minutes.do(retrain_models)

    def run_scheduler():
        logging.info("✔️ El programador de tareas (scheduler) está en ejecución.")
        while True:
            schedule.run_pending()
            time.sleep(1)

    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # Para que se detenga cuando el programa principal termine
    scheduler_thread.start()
    logging.info("✔️ Hilo del scheduler iniciado correctamente.")


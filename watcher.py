import os
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class MyHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.process = None
        self.start_bot()

    def on_modified(self, event):
        # Ignorar los archivos que no sean .py o que estén fuera de ./src
        if not event.src_path.endswith(".py") or not event.src_path.startswith("./src"):
            return

        print(f"Archivo modificado: {event.src_path}. Reiniciando el bot...")
        self.restart_bot()

    def start_bot(self):
        # Ejecutar el archivo main.py
        self.process = subprocess.Popen([sys.executable, "main.py"])
        print("Bot iniciado automáticamente.")

    def restart_bot(self):
        # Detener el proceso anterior si está corriendo
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Bot detenido.")
        
        # Reiniciar el bot
        self.start_bot()

if __name__ == "__main__":
    path = "./src"  # Solo observar archivos dentro de ./src
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

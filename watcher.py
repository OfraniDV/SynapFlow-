import os
import time
import subprocess
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MyHandler(FileSystemEventHandler):
    def __init__(self, script_name):
        self.script_name = script_name
        self.python_interpreter = sys.executable  # Obtiene la ruta del Python activo en el entorno virtual
        self.process = None
        self.restart_script()

    def restart_script(self):
        try:
            if self.process:
                print(f"Terminando el proceso anterior de {self.script_name}")
                self.process.terminate()
                self.process.wait(timeout=5)  # Timeout para forzar el cierre
            self.process = subprocess.Popen([self.python_interpreter, self.script_name])
            print(f'{self.script_name} iniciado.')
        except subprocess.TimeoutExpired:
            print(f"El proceso anterior no se cerró en el tiempo esperado. Forzando cierre.")
            self.process.kill()
            self.process = subprocess.Popen([self.python_interpreter, self.script_name])
        except Exception as e:
            print(f"Error al reiniciar {self.script_name}: {e}")


    def on_modified(self, event):
        if event.src_path.endswith(self.script_name):  # Verifica si se modificó bot.py
            print(f'{event.src_path} modificado. Reiniciando {self.script_name}...')
            self.restart_script()


if __name__ == "__main__":
    script_name = 'bot.py'  # Cambia esto si tu script tiene otro nombre
    path = '.'  # Directorio a observar (directorio actual)

    event_handler = MyHandler(script_name)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    print(f'Observando cambios en {path}...')
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

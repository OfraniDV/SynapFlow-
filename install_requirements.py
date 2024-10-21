import subprocess

# Abre y lee el archivo requirements.txt
with open('requirements.txt') as f:
    packages = f.read().splitlines()

# Intenta instalar cada paquete
for package in packages:
    try:
        print(f"Instalando {package}...")
        subprocess.check_call(['pip', 'install', package])
    except subprocess.CalledProcessError:
        print(f"Error al instalar {package}. Saltando al siguiente paquete.")

# src/ai/data_preprocessor.py

import re

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ\s]', '', texto)
    return texto

def preparar_datos(textos):
    textos_limpios = [limpiar_texto(texto) for texto in textos]
    return textos_limpios

import re

def preprocess_interaction(text):
    # Eliminar caracteres especiales, convertir a minúsculas, etc.
    text = re.sub(r'\W+', ' ', text.lower())
    # Aquí podrías agregar más técnicas de preprocesamiento como tokenización
    vectorized = [ord(char) for char in text[:100]]  # Un ejemplo simple
    return vectorized

from ai.model import load_model
from ai.data_preprocessor import preprocess_interaction

def predict_response(text):
    model = load_model()
    if model:
        preprocessed_data = preprocess_interaction(text)
        response = model.predict([preprocessed_data])
        return response
    else:
        return "El modelo no está entrenado."

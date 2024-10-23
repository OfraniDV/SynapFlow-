#model.py

# Importaciones de librerías estándar
import os
import json
import pickle
import logging
import time
from datetime import datetime
import requests
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from dotenv import load_dotenv  # Para cargar variables de entorno

# Importaciones de spaCy para procesamiento de lenguaje natural
import spacy
nlp = spacy.load("es_core_news_md")  # Modelo de lenguaje en español

# Importaciones de TensorFlow y Keras para construcción del modelo
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Activation, Dot, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer


# Importaciones de Scikit-learn para procesamiento de datos
from sklearn.preprocessing import MultiLabelBinarizer

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()



# Crear un formateador personalizado
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# Configurar el logger
logger = logging.getLogger(__name__)

# Configurar el nivel de logging a través de una variable de entorno (si existe) o por defecto a INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(log_level)

# Crear un manejador para la consola
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_formatter = logging.Formatter(log_format, datefmt=date_format)
console_handler.setFormatter(console_formatter)

# Crear un manejador para un archivo de log
file_handler = logging.FileHandler('app.log', mode='a')  # Guarda en un archivo llamado 'app.log'
file_handler.setLevel(log_level)
file_formatter = logging.Formatter(log_format, datefmt=date_format)
file_handler.setFormatter(file_formatter)

# Agregar los manejadores al logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info('Logger inicializado correctamente.')

# Definir la capa de atención personalizada
class AttentionLayer(Layer):
    def call(self, inputs):
        decoder_outputs, encoder_outputs = inputs
        attn_layer = tf.keras.layers.Attention(name='attention_layer')
        return attn_layer([decoder_outputs, encoder_outputs])

def build_model(self):
    """Construye el modelo Encoder-Decoder mejorado con mecanismo de atención."""
    try:
        logging.info("[Conversar] Construyendo el modelo Encoder-Decoder mejorado con atención...")

        # Tamaño del vocabulario y dimensiones
        vocab_size = len(self.tokenizer.word_index) + 1
        embedding_dim = 256  # Puedes ajustar este valor
        latent_dim = 512     # Aumentamos el tamaño de la capa LSTM

        logging.info(f"[Conversar] Tamaño del vocabulario: {vocab_size}")
        logging.info(f"[Conversar] Dimensión de embedding: {embedding_dim}")
        logging.info(f"[Conversar] Dimensión latente (LSTM units): {latent_dim}")

        # ENCODER
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
        encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_lstm'))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # DECODER
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
        decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        # MECANISMO DE ATENCIÓN
        attn_out = AttentionLayer()([decoder_outputs, encoder_outputs])

        # Concatenar la salida del decoder y el contexto de atención
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        # Capa densa final para generar la predicción de la siguiente palabra
        decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Modelo de entrenamiento
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        logging.info("[Conversar] Modelo de entrenamiento creado.")

        # Compilar el modelo
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logging.info("[Conversar] Modelo compilado exitosamente.")

        # Configurar los modelos de inferencia
        self.setup_inference_models()
        logging.info("[Conversar] Modelos de inferencia configurados.")

        logging.info("[Conversar] Modelo Encoder-Decoder mejorado construido y compilado exitosamente.")

    except Exception as e:
        logging.error(f"[Conversar] Error al construir el modelo: {e}")
        self.model = None

class NumerologyModel:
    def __init__(self, db):
        self.db = db
        self.model = None  # Modelo de red neuronal para numerología
        self.mlb = None  # MultiLabelBinarizer para etiquetas de entrenamiento
        self.is_trained = False  # Indicador de si el modelo ha sido entrenado
        self.mapping = {}  # Mapeo de número de entrada a recomendaciones de salida
        self.vibrations_by_day = {}  # Vibraciones numerológicas por día de la semana
        self.most_delayed_numbers = {}  # Números con más días de atraso por categoría
        self.delayed_numbers = {}  # Números atrasados agrupados por categoría
        self.root_numbers = {}  # Raíces de números detectadas en fórmulas
        self.inseparable_numbers = {}  # Números inseparables detectados en fórmulas
        self.lottery_results = []  # Resultados de loterías extraídos de las fórmulas
        self.max_sequence_length = None  # Longitud máxima de las secuencias para el modelo
        self.current_date = None  # Fecha actual (se actualiza al procesar fórmulas)
        self.charadas = {}  # Diccionario para almacenar las charadas
        
        # Inicialización de otras variables utilizadas en patrones y predicciones
        self.most_probable_numbers = []  # Números más probables basados en coincidencias de patrones
        self.pattern_matches = {}  # Diccionario para rastrear coincidencias de números en patrones
        

        
        logging.info("Clase NumerologyModel inicializada con todos los atributos necesarios.")

    def generate_features(self, input_number):
        """Genera características para un número dado basado en vibraciones, atrasos y otros patrones."""
        features = [input_number]
        
        # Agregar vibraciones del día
        current_date = datetime.now()
        day_of_week_es = self.get_day_in_spanish(current_date.strftime("%A"))
        day_vibrations = self.vibrations_by_day.get(day_of_week_es, {})
        digits = day_vibrations.get('digits', [])
        features.extend([int(digit) for digit in digits if digit.isdigit()])

        # Agregar vibraciones del número
        number_vibrations = self.mapping.get(input_number, [])
        features.extend([int(num) for num in number_vibrations if num.isdigit()])

        # Agregar indicadores de números más atrasados
        for category in ['CENTENAS', 'DECENAS', 'TERMINALES', 'PAREJAS']:
            most_delayed = self.most_delayed_numbers.get(category)
            is_most_delayed = 1 if most_delayed and input_number == most_delayed['number'] else 0
            features.append(is_most_delayed)

        return features
   
    def load(self, model_file):
        """Carga el modelo preentrenado desde el archivo y aplica ajustes finos si existen."""
        try:
            logging.info("[NumerologyModel] Cargando el modelo de numerología...")

            # Cargar el modelo
            self.model = tf.keras.models.load_model(model_file)
            logging.info("[NumerologyModel] Modelo de numerología cargado exitosamente.")

            # Cargar el MultiLabelBinarizer
            mlb_path = 'mlb_numerologia.pkl'
            if os.path.exists(mlb_path):
                with open(mlb_path, 'rb') as mlb_file:
                    self.mlb = pickle.load(mlb_file)
                logging.info("[NumerologyModel] MultiLabelBinarizer cargado exitosamente.")
            else:
                logging.error(f"[NumerologyModel] Archivo {mlb_path} no encontrado.")
                self.is_trained = False
                return

            # Cargar la longitud máxima de secuencias
            seq_length_path = 'max_sequence_length_numerologia.pkl'
            if os.path.exists(seq_length_path):
                with open(seq_length_path, 'rb') as seq_file:
                    self.max_sequence_length = pickle.load(seq_file)
                logging.info("[NumerologyModel] Longitud máxima de secuencias cargada exitosamente.")
            else:
                logging.error(f"[NumerologyModel] Archivo {seq_length_path} no encontrado.")
                self.is_trained = False
                return

            # Extraer reglas de las fórmulas inmediatamente después de cargar el modelo
            formulas = self.db.get_all_formulas()
            if formulas:
                logging.info("[NumerologyModel] Extrayendo reglas de las fórmulas después de cargar el modelo...")
                self.extract_rules_from_formulas(formulas)
            else:
                logging.error("[NumerologyModel] No se encontraron fórmulas al cargar el modelo.")
                self.is_trained = False
                return

            # Marcar el modelo como cargado y entrenado
            self.is_trained = True
            logging.info("[NumerologyModel] Modelo de numerología cargado y listo para su uso.")

            # Aplicar ajustes finos si existen suficientes datos
            logging.info("[NumerologyModel] Aplicando ajustes finos guardados.")
            self.aplicar_ajustes_finos()

        except Exception as e:
            logging.error(f"[NumerologyModel] Error al cargar el modelo de numerología: {e}")
            self.is_trained = False

    def aplicar_ajustes_finos(self):
        """Aplica los ajustes finos guardados al modelo basado en las fórmulas extraídas."""
        try:
            ajuste_fino_path = 'ajuste_fino_datos_numerologia.pkl'
            # Cargar los datos para el ajuste fino
            datos_para_ajuste = []
            if os.path.exists(ajuste_fino_path):
                with open(ajuste_fino_path, 'rb') as f:
                    while True:
                        try:
                            data = pickle.load(f)
                            datos_para_ajuste.append(data)
                        except EOFError:
                            break

            # Asegurarse de que los datos para ajuste fino están basados en las fórmulas extraídas
            if len(self.mapping) == 0 or len(self.vibrations_by_day) == 0:
                logging.warning("No se han extraído suficientes reglas para realizar el ajuste fino.")
                return

            # Asegurarse de que hay suficientes datos para el ajuste fino
            if len(datos_para_ajuste) < 10:
                logging.info("No hay suficientes datos para realizar un ajuste fino.")
                return

            # Extraer las entradas y las etiquetas
            inputs = []
            respuestas = []
            for data in datos_para_ajuste:
                try:
                    # Verificar que el input sea un número válido
                    input_number = int(data["input"])
                    inputs.append(input_number)
                    respuestas.append(data["response"])
                except ValueError:
                    logging.warning(f"Entrada inválida encontrada y omitida: {data['input']}")

            # Si no hay entradas válidas, salir
            if not inputs:
                logging.warning("No se encontraron entradas válidas para el ajuste fino.")
                return

            # Generar características y etiquetas a partir de las fórmulas
            nuevas_X = [self.generate_features(input_number) for input_number in inputs]
            nuevas_X = pad_sequences(nuevas_X, maxlen=self.max_sequence_length)
            nuevas_y = self.mlb.fit_transform(respuestas)

            # Revisar dimensiones de las características y etiquetas
            logging.info(f"Forma de nuevas_X: {nuevas_X.shape}")
            logging.info(f"Forma de nuevas_y: {nuevas_y.shape}")
            
            # Verificar si las dimensiones de nuevas_X y nuevas_y coinciden
            if nuevas_X.shape[0] != nuevas_y.shape[0]:
                logging.error(f"Las dimensiones de nuevas_X ({nuevas_X.shape}) y nuevas_y ({nuevas_y.shape}) no coinciden.")
                return

            # Verificar la cantidad de clases (columnas de nuevas_y) y ajustar el modelo si es necesario
            if nuevas_y.shape[1] != self.model.output_shape[1]:
                logging.warning(f"El número de clases en nuevas_y ({nuevas_y.shape[1]}) no coincide con la salida del modelo ({self.model.output_shape[1]}). Ajustando la capa de salida.")
                from tensorflow.keras.layers import Dense
                self.model.pop()  # Eliminar la última capa
                self.model.add(Dense(nuevas_y.shape[1], activation='sigmoid'))
                self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                logging.info(f"Capa de salida del modelo ajustada a {nuevas_y.shape[1]} clases.")

            # Realizar el ajuste fino del modelo basado en las fórmulas
            logging.info("Comenzando el ajuste fino del modelo.")
            self.model.fit(nuevas_X, nuevas_y, epochs=2, batch_size=10)
            logging.info("Ajustes finos aplicados exitosamente basados en las fórmulas.")

        except Exception as e:
            logging.error(f"Error al aplicar ajustes finos: {e}")


    def ajuste_fino(self):
        """Realiza un ajuste fino en el modelo de numerología utilizando solo fórmulas extraídas."""
        try:
            # Obtener todas las fórmulas desde la tabla logsfirewallids
            formulas = self.db.get_all_formulas()
            if not formulas:
                logging.error("No se encontraron fórmulas en la tabla logsfirewallids.")
                return

            # Extraer reglas de las fórmulas
            logging.info("Extrayendo reglas de las fórmulas...")
            self.extract_rules_from_formulas(formulas)

            # Si no se han extraído suficientes reglas, no realizar el ajuste fino
            if len(self.mapping) == 0:
                logging.error("No se encontraron suficientes reglas para el ajuste fino.")
                return

            # Preparar los datos actuales (fórmulas e interacciones)
            logging.info("Iniciando la preparación de los datos para ajuste fino...")
            self.prepare_data()

            if not hasattr(self, 'X') or not hasattr(self, 'y') or self.X.size == 0 or len(self.y) == 0:
                logging.error("No hay suficientes datos para realizar el ajuste fino.")
                return

            # Realizar ajuste fino basado en las reglas extraídas de las fórmulas
            logging.info("Realizando ajuste fino del modelo basado en las fórmulas...")
            X_train = self.generar_caracteristicas(self.X)
            y_binarized = self.preprocesar_etiquetas(self.y)

            self.model.fit(X_train, y_binarized, epochs=2, batch_size=10, verbose=1)
            logging.info("Ajuste fino completado exitosamente.")

            # Guardar el modelo ajustado
            self.model.save('numerology_model_finetuned.keras')
            logging.info("Modelo ajustado guardado como 'numerology_model_finetuned.keras'.")

        except Exception as e:
            logging.error(f"Error durante el ajuste fino NumerologyModel: {e}")

    def generar_caracteristicas(self, X_data):
        """Genera las características para el ajuste fino."""
        try:
            X_features = []
            for input_number in X_data.flatten():
                input_number = int(input_number)
                features = self.generate_features(input_number)
                X_features.append(features)

            # Realizar padding en las secuencias de características
            X_train = pad_sequences(X_features, padding='post', dtype='int32', maxlen=self.max_sequence_length)
            logging.info(f"Forma final de X_train después del padding: {X_train.shape}")
            return X_train

        except Exception as e:
            logging.error(f"Error al generar características para ajuste fino: {e}")
            return None

    def preprocesar_etiquetas(self, y_data):
        """Preprocesa las etiquetas de los datos para el ajuste fino."""
        try:
            y_binarized = self.mlb.fit_transform(y_data)
            logging.info(f"Forma de y_binarized después de aplicar MultiLabelBinarizer: {y_binarized.shape}")
            return y_binarized

        except Exception as e:
            logging.error(f"Error al preprocesar etiquetas para ajuste fino: {e}")
            return None

    def prepare_data(self):
        # Obtener todas las fórmulas desde la tabla logsfirewallids
        formulas = self.db.get_all_formulas()
        if not formulas:
            logging.error("No se encontraron fórmulas en la tabla logsfirewallids.")
            return

        # Extraer reglas de las fórmulas
        data = self.extract_rules_from_formulas(formulas)
        if not data:
            logging.error("No se pudieron extraer reglas de las fórmulas.")
            return

        # Obtener interacciones desde la tabla logsfirewallids
        interactions = self.db.get_all_interactions()
        if interactions:
            interaction_data = []
            for user_input, recommendations in interactions:
                # Convertir user_input a número entero
                try:
                    input_number = int(user_input.strip())
                except ValueError:
                    #logging.warning(f"Entrada de usuario inválida: {user_input}")
                    continue  # Ignorar si no es un número válido

                # Procesar recomendaciones (asumiendo que están almacenadas como una cadena separada por comas)
                recommended_numbers = [num.strip() for num in recommendations.split(',') if num.strip().isdigit()]
                if not recommended_numbers:
                    #logging.warning(f"Recomendaciones inválidas: {recommendations}")
                    continue

                interaction_data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})

            # Agregar los datos de interacciones al conjunto de datos principal
            data.extend(interaction_data)
            #logging.info(f"Datos combinados de fórmulas e interacciones: {len(data)} entradas.")

        # Crear DataFrame
        self.data = pd.DataFrame(data, columns=['input_number', 'recommended_numbers'])

        # Preprocesar los datos
        self.X = self.data['input_number'].values.reshape(-1, 1)
        self.y = self.data['recommended_numbers'].apply(lambda x: [int(num) for num in x])
        #logging.info(f"Datos de entrenamiento X: {self.X}")
        #logging.info(f"Etiquetas de entrenamiento y: {self.y}")

        # Preparar los números atrasados más significativos
        # Asumimos que self.delayed_numbers se ha llenado en extract_rules_from_formulas
        self.most_delayed_numbers = {}
        for category in self.delayed_numbers:
            if self.delayed_numbers[category]:
                # Obtener el número con más días de atraso en la categoría
                max_delay = max(self.delayed_numbers[category], key=lambda x: x['days'])
                self.most_delayed_numbers[category] = max_delay
        #logging.info(f"Números más atrasados por categoría: {self.most_delayed_numbers}")

    def extract_rules_from_formulas(self, formulas):
        data = []
        self.mapping = {}
        self.vibrations_by_day = {}  # Diccionario para almacenar vibraciones por día
        self.root_numbers = {}  # Diccionario para raíces de números
        self.inseparable_numbers = {}  # Diccionario para números inseparables
        self.delayed_numbers = {}  # Diccionario para números atrasados
        self.lottery_results = []  # Lista para resultados de lotería
        self.charadas = {}  # Nuevo diccionario para almacenar las charadas
        self.current_date = None  # Variable para la fecha actual

        # Función auxiliar para procesar números con 'v'
        def process_numbers_with_v(numbers_list):
            processed_numbers = []
            for num in numbers_list:
                if 'v' in num:
                    base_num = re.findall(r'\d{1,2}', num)[0]
                    reversed_num = base_num[::-1].zfill(2)  # Invertir y rellenar con ceros si es necesario
                    processed_numbers.extend([base_num, reversed_num])
                else:
                    processed_numbers.append(num)
            return processed_numbers

        for formula in formulas:
            lines = formula.split('\n')
            current_category = None  # Variable para la categoría actual
            for line in lines:
                line = line.strip()

                # Detectar la fecha
                date_match = re.match(r'🗓(\d{2}/\d{2}/\d{4})🗓', line)
                if date_match:
                    self.current_date = date_match.group(1)
                    logging.debug(f"Fecha detectada: {self.current_date}")
                    continue

                # Detectar el periodo del día
                if re.match(r'D[ií]a', line, re.IGNORECASE):
                    current_category = 'Día'
                    continue
                elif re.match(r'Noche', line, re.IGNORECASE):
                    current_category = 'Noche'
                    continue

                # Variable para verificar si se encontró un patrón en la línea
                matches_found = False

                # Patrón 1: --Si sale XX tambien sale YY.ZZ.WW
                match = re.match(r'--Si sale (\d{1,2}) tambien sale ([\d\.\,v ]+)', line)
                if match:
                    input_number = int(match.group(1))
                    raw_numbers = re.findall(r'\d{1,2}v?', match.group(2))
                    recommended_numbers = process_numbers_with_v(raw_numbers)
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    logging.debug(f"Patrón 1 encontrado: {line}")
                    matches_found = True
                    continue  # Pasar a la siguiente línea

                # Patrón 2: Símbolos y formato 👉XX=YY.ZZ.AA.BB
                match = re.match(r'.*👉(\d{1,2})[=:-]([\d\.\,\sv]+)', line)
                if match:
                    input_number = int(match.group(1))
                    raw_numbers = re.findall(r'\d{1,2}v?', match.group(2))
                    recommended_numbers = process_numbers_with_v(raw_numbers)
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    logging.debug(f"Patrón 2 encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 3: Formato XX--YY, ZZ, AA, BB (Inseparables)
                match = re.match(r'.*(\d{1,2})--([\d\.\,\sv]+)', line)
                if match:
                    input_number = int(match.group(1))
                    raw_numbers = re.findall(r'\d{1,2}v?', match.group(2))
                    inseparable_numbers = process_numbers_with_v(raw_numbers)
                    # Almacenar en inseparable_numbers
                    self.inseparable_numbers.setdefault(input_number, []).extend(inseparable_numbers)
                    logging.debug(f"Patrón 3 (Inseparables) encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 4: Formato 🪸XX=(YYv)=ZZ🪸 (Raíces)
                match = re.match(r'^.*?(\d{1,2})=\((\d{1,2}v?)\)=([\d]{1,2}).*?$', line)
                if match:
                    input_number = int(match.group(1))
                    v_number = match.group(2)
                    output_number = match.group(3)
                    v_numbers = process_numbers_with_v([v_number])
                    recommended_numbers = v_numbers + [output_number]
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    # Almacenar en root_numbers
                    self.root_numbers.setdefault(input_number, []).extend(recommended_numbers)
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    logging.debug(f"Patrón 4 (Raíz) encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 5: Formato XX👉YY.ZZ.AA.BB
                match = re.match(r'.*(\d{1,2})👉([\d\.\,\sv]+)', line)
                if match:
                    input_number = int(match.group(1))
                    raw_numbers = re.findall(r'\d{1,2}v?', match.group(2))
                    recommended_numbers = process_numbers_with_v(raw_numbers)
                    # Almacenar en root_numbers
                    self.root_numbers.setdefault(input_number, []).extend(recommended_numbers)
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    logging.debug(f"Patrón 5 encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 6: Formato con puntos y comas XX=YY, ZZ
                match = re.match(r'.*(\d{1,2})[:=] ?([\d\.\,\sv]+)', line)
                if match:
                    input_number = int(match.group(1))
                    raw_numbers = re.findall(r'\d{1,2}v?', match.group(2))
                    recommended_numbers = process_numbers_with_v(raw_numbers)
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    logging.debug(f"Patrón 6 encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 7: Formato especial para parejas XX=YY
                match = re.match(r'.*(\d{1,2})=(\d{1,2}v?)', line)
                if match:
                    input_number = int(match.group(1))
                    raw_numbers = [match.group(2)]
                    recommended_numbers = process_numbers_with_v(raw_numbers)
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    logging.debug(f"Patrón 7 encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 8: Tabla de Raíces XX👉YY.ZZ... (también puede ser usado para raíces)
                match = re.match(r'(\d{1,2})👉([\d\.\,\sv]+)', line)
                if match:
                    input_number = int(match.group(1))
                    raw_numbers = re.findall(r'\d{1,2}v?', match.group(2))
                    recommended_numbers = process_numbers_with_v(raw_numbers)
                    # Almacenar en root_numbers
                    self.root_numbers.setdefault(input_number, []).extend(recommended_numbers)
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    logging.debug(f"Patrón 8 (Raíz) encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 9: Formato XX-YY=(ZZ)
                match = re.match(r'^.*?(\d{1,2})-(\d{1,2})=\((\d{1,2})\).*?$', line)
                if match:
                    input_numbers = [int(match.group(1)), int(match.group(2))]
                    recommended_number = match.group(3)
                    for input_number in input_numbers:
                        recommended_numbers = [recommended_number]
                        data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                        self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    logging.debug(f"Patrón 9 encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 10: Formato DíaAbbr=(digits)=numbers
                match = re.match(r'^.*?([LMXJVSD])=\(([\d\.y]+)\)=([\d\.\,\sv]+).*?$', line)
                if match:
                    day_abbr = match.group(1)
                    digits = match.group(2)
                    numbers_str = match.group(3)
                    digits_list = re.findall(r'\d+', digits)
                    raw_numbers = re.findall(r'\d{1,2}v?', numbers_str)
                    recommended_numbers = process_numbers_with_v(raw_numbers)
                    day_full = self.day_abbr_to_full_name(day_abbr)
                    # Almacenar las vibraciones por día
                    self.vibrations_by_day.setdefault(day_full, {}).update({
                        'digits': digits_list,
                        'numbers': recommended_numbers
                    })
                    logging.debug(f"Patrón 10 encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 11: Formato Parejas DíaAbbr=Numbers
                matches = re.findall(r'([LMXJVSD])=([\d\.]+)', line)
                if matches:
                    for day_abbr, numbers_str in matches:
                        raw_numbers = re.findall(r'\d{1,2}v?', numbers_str)
                        numbers = process_numbers_with_v(raw_numbers)
                        day_full = self.day_abbr_to_full_name(day_abbr)
                        self.vibrations_by_day.setdefault(day_full, {}).setdefault('parejas', []).extend(numbers)
                    logging.debug(f"Patrón 11 encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 12: Formato Especial (Texto genérico con números)
                # Si no se ha encontrado ningún patrón, buscar números en la línea
                numbers_in_line = re.findall(r'\b\d{1,2}v?\b', line)
                if numbers_in_line:
                    logging.debug(f"Números encontrados sin patrón específico: {numbers_in_line}")
                    # Puedes decidir cómo manejar estos números o simplemente ignorarlos
                    continue

                # Patrón 13: Formato de números atrasados (Centenas, Decenas, Terminales, Parejas)
                match = re.match(r'^.*?([0-9]{1,2})[-–](\d+)\s*[dD][ií]as?.*$', line)
                if match and current_category:
                    number = int(match.group(1))  # Número atrasado
                    days = int(match.group(2))  # Días de atraso
                    category = current_category  # Categoría actual (Centena, Decena, Terminal, etc.)

                    # Almacenar en delayed_numbers dentro de su categoría
                    self.delayed_numbers.setdefault(category, []).append({'number': number, 'days': days})
                    logging.debug(f"Patrón 13 (Números atrasados) encontrado: {number} en {category} con {days} días")
                    matches_found = True
                    continue

                # Patrón 14: Formato de resultados de lotería (resumen)
                match = re.match(r'^✅([A-Z]{2})[^0-9]*(\d{1,2}:\d{2}\s*[APM]{2})\s+([\d-]+)', line)
                if match and self.current_date and current_category:
                    location = match.group(1)
                    time = match.group(2)
                    numbers_str = match.group(3)
                    winning_numbers = numbers_str.split('-')
                    winning_numbers = [num.strip() for num in winning_numbers if num.strip().isdigit()]

                    # Almacenar en lottery_results
                    self.lottery_results.append({
                        'date': self.current_date,
                        'period': current_category,
                        'location': location,
                        'time': time,
                        'winning_numbers': winning_numbers
                    })
                    logging.debug(f"Patrón 14 (Resultados de lotería) encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 15: Formato detallado de resultados de lotería con centena, fijo y corridos
                match = re.match(r'^✅([A-Z]{2})[^0-9]*(\d{1,2}:\d{2}\s*[APM]{2})\s+(\d{3})-([\d-]+)', line)
                if match and self.current_date and current_category:
                    location = match.group(1)
                    time = match.group(2)
                    centena = match.group(3)
                    corridos_str = match.group(4)

                    fijo = centena[-2:]  # Los dos últimos dígitos de la centena
                    corridos = corridos_str.split('-')
                    corridos = [num.strip() for num in corridos if num.strip().isdigit()]

                    # Generar los posibles parles
                    parles = []
                    # Parles entre fijo y corridos
                    for corrido in corridos:
                        parles.append((fijo, corrido))
                    # Parles entre corridos
                    if len(corridos) >= 2:
                        for i in range(len(corridos)):
                            for j in range(i+1, len(corridos)):
                                parles.append((corridos[i], corridos[j]))

                    # Almacenar en lottery_results
                    self.lottery_results.append({
                        'date': self.current_date,
                        'period': current_category,
                        'location': location,
                        'time': time,
                        'centena': centena,
                        'fijo': fijo,
                        'corridos': corridos,
                        'parles': parles
                    })
                    logging.debug(f"Patrón 15 (Resultados detallados de lotería) encontrado: {line}")
                    matches_found = True
                    continue

                # Regla 16: Formato de Raíz de Número
                match = re.match(r'^.*?(\d{1,2}v?)\s*\(Raíz del\s*(\d{1,2})\).*?$', line)
                if match:
                    root_number = match.group(1)
                    base_number = match.group(2)
                    recommended_numbers = [root_number, base_number]

                    # Almacenar en root_numbers y mapping
                    self.root_numbers.setdefault(base_number, []).extend(recommended_numbers)
                    self.mapping.setdefault(base_number, []).extend(recommended_numbers)
                    logging.debug(f"Raíz de número encontrada: {line}")
                    matches_found = True
                    continue

                # Regla 17: Formato Charada
                match = re.match(r'^([A-Z]+)👉([\d\s]+)', line)
                if match:
                    charada_name = match.group(1).strip()
                    numbers_str = match.group(2).strip()
                    charada_numbers = re.findall(r'\d{1,2}', numbers_str)

                    # Almacenar en el diccionario de charadas
                    self.charadas.setdefault(charada_name, []).extend(charada_numbers)
                    logging.debug(f"Charada encontrada: {charada_name} con números: {charada_numbers}")
                    matches_found = True
                    continue

                # Regla 18: Charada Amplia
                match = re.match(r'👉Charada Amplia\s*([A-Z]+)', line)
                if match:
                    charada_name = match.group(1).strip()
                    next_line = next((l for l in lines if l.strip()), "")
                    charada_amplia_numbers = re.findall(r'\d{1,2}', next_line)

                    # Almacenar en el diccionario de charadas
                    self.charadas.setdefault(charada_name, []).extend(charada_amplia_numbers)
                    logging.debug(f"Charada Amplia encontrada: {charada_name} con números: {charada_amplia_numbers}")
                    matches_found = True
                    continue


                # Si no se encontró ningún patrón y no hay números, pasar a la siguiente línea
                if not matches_found:
                    logging.debug(f"Ningún patrón encontrado para la línea: {line}")
                    continue

        # Eliminar duplicados en los diccionarios
        for key in self.mapping:
            self.mapping[key] = list(set(self.mapping[key]))
        for key in self.root_numbers:
            self.root_numbers[key] = list(set(self.root_numbers[key]))
        for key in self.inseparable_numbers:
            self.inseparable_numbers[key] = list(set(self.inseparable_numbers[key]))
        for day in self.vibrations_by_day:
            for k in self.vibrations_by_day[day]:
                self.vibrations_by_day[day][k] = list(set(self.vibrations_by_day[day][k]))
        for category in self.delayed_numbers:
            numbers_set = { (item['number'], item['days']) for item in self.delayed_numbers[category] }
            self.delayed_numbers[category] = [ {'number': num, 'days': days} for num, days in numbers_set ]

        logging.info(f"Total de reglas extraídas: {len(data)}")
        logging.debug(f"Datos extraídos: {data}")
        logging.debug(f"Diccionario de mapeo: {self.mapping}")
        logging.debug(f"Vibraciones por día: {self.vibrations_by_day}")
        logging.debug(f"Raíces de números: {self.root_numbers}")
        logging.debug(f"Números inseparables: {self.inseparable_numbers}")
        logging.debug(f"Números atrasados: {self.delayed_numbers}")
        logging.debug(f"Resultados de lotería: {self.lottery_results}")
        logging.debug(f"Charadas: {self.charadas}")
        
        return data

    # Función auxiliar para convertir abreviaturas de días a nombres completos en español
    def day_abbr_to_full_name(self, abbr):
        mapping = {
            'L': 'Lunes',
            'M': 'Martes',
            'X': 'Miércoles',
            'J': 'Jueves',
            'V': 'Viernes',
            'S': 'Sábado',
            'D': 'Domingo',
        }
        return mapping.get(abbr, abbr)  # Retorna la abreviatura si no se encuentra en el diccionario

    def train(self):
        """Entrena el modelo de numerología utilizando los datos preparados."""
        try:
            logging.info("===== [NumerologyModel] Iniciando el entrenamiento del modelo de numerología =====")

            # Preparar los datos
            logging.info("[NumerologyModel] Preparando los datos para el entrenamiento...")
            self.prepare_data()

            # Verificar que los datos de entrada y las etiquetas existen
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                logging.error("[NumerologyModel] Los datos de entrenamiento no están disponibles. Asegúrate de que los datos fueron cargados correctamente.")
                self.is_trained = False
                return

            if self.X.size == 0 or len(self.y) == 0:
                logging.error("[NumerologyModel] No hay datos suficientes para entrenar el modelo. X o y están vacíos.")
                self.is_trained = False
                return

            # Incorporar vibraciones y otros datos en las características usando la función generate_features
            logging.info("[NumerologyModel] Incorporando vibraciones y otros datos en las características...")
            X_features = [self.generate_features(int(input_number)) for input_number in self.X.flatten()]

            # Verificar que X_features no esté vacío antes de calcular max_sequence_length
            if not X_features or len(X_features) == 0:
                logging.error("[NumerologyModel] X_features está vacío o no tiene datos válidos.")
                self.is_trained = False
                return

            # Guardar la longitud máxima de secuencia
            self.max_sequence_length = max(len(seq) for seq in X_features)
            logging.info(f"[NumerologyModel] Longitud máxima de secuencia establecida en: {self.max_sequence_length}")

            # Verificar que max_sequence_length sea válida
            if not isinstance(self.max_sequence_length, int) or self.max_sequence_length <= 0:
                logging.error(f"[NumerologyModel] max_sequence_length no es válido: {self.max_sequence_length}")
                self.is_trained = False
                return

            # Convertir a matriz numpy con padding
            X_train = pad_sequences(X_features, padding='post', dtype='int32', maxlen=self.max_sequence_length)
            logging.info(f"[NumerologyModel] Forma final de X_train después del padding: {X_train.shape}")

            # Preprocesar las etiquetas con MultiLabelBinarizer
            self.mlb = MultiLabelBinarizer()
            y_binarized = self.mlb.fit_transform(self.y)
            logging.info(f"[NumerologyModel] Forma de y_binarized después de aplicar MultiLabelBinarizer: {y_binarized.shape}")

            # Verificar si hay valores NaN o Inf
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                logging.error("[NumerologyModel] X_train contiene valores NaN o Inf. No se puede continuar con el entrenamiento.")
                self.is_trained = False
                return

            if np.isnan(y_binarized).any() or np.isinf(y_binarized).any():
                logging.error("[NumerologyModel] y_binarized contiene valores NaN o Inf. No se puede continuar con el entrenamiento.")
                self.is_trained = False
                return

            num_classes = y_binarized.shape[1]  # Número de clases únicas en las etiquetas
            logging.info(f"[NumerologyModel] Número de clases en y_binarized: {num_classes}")

            # Definir el modelo de red neuronal para secuencias con una arquitectura mejorada
            max_input_value = np.max(X_train) + 1  # Valor máximo de entrada para la capa Embedding
            logging.info(f"[NumerologyModel] El valor máximo de entrada para la capa Embedding será: {max_input_value}")

            # Definir la arquitectura del modelo mejorada
            self.model = Sequential()
            self.model.add(Embedding(input_dim=max_input_value, output_dim=128))
            self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
            self.model.add(Dropout(0.3))
            self.model.add(Bidirectional(LSTM(64)))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(num_classes, activation='sigmoid'))

            # Compilar el modelo
            logging.info("[NumerologyModel] Compilando el modelo...")
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Entrenar el modelo
            logging.info("[NumerologyModel] Entrenando el modelo de numerología...")
            self.model.fit(X_train, y_binarized, epochs=10, batch_size=10, verbose=1)

            # Guardar el modelo entrenado
            self.model.save('numerology_model.keras')
            logging.info("[NumerologyModel] Modelo de numerología entrenado y guardado exitosamente.")

            # Guardar el MultiLabelBinarizer
            with open('mlb_numerologia.pkl', 'wb') as mlb_file:
                pickle.dump(self.mlb, mlb_file)
            logging.info("[NumerologyModel] MultiLabelBinarizer guardado en 'mlb_numerologia.pkl'.")

            # Guardar la longitud máxima de secuencias
            with open('max_sequence_length_numerologia.pkl', 'wb') as seq_file:
                pickle.dump(self.max_sequence_length, seq_file)
            logging.info("[NumerologyModel] Longitud máxima de secuencias guardada en 'max_sequence_length_numerologia.pkl'.")

            # Indicar que el modelo fue entrenado correctamente
            self.is_trained = True
            logging.info("[NumerologyModel] Modelo de numerología entrenado exitosamente.")

        except Exception as e:
            logging.error(f"[NumerologyModel] Error durante el entrenamiento del modelo: {e}")
            self.is_trained = False

    def predict(self, input_number):
        # Usar mapeo directo si existe
        if input_number in self.mapping:
            recommended_numbers = self.mapping[input_number]
            logging.debug(f"Números recomendados (mapeo directo): {recommended_numbers}")
            return recommended_numbers
        elif self.is_trained and self.model and self.mlb and self.max_sequence_length:
            try:
                logging.info(f"Realizando predicción para el número: {input_number}")

                # Generar las características usando la nueva función generate_features
                features = self.generate_features(input_number)
                logging.debug(f"Características generadas para el número {input_number}: {features}")

                # Convertir las características a una matriz numpy y aplicar padding
                input_features = pad_sequences([features], padding='post', dtype='int32', maxlen=self.max_sequence_length)
                logging.debug(f"Características generadas para la predicción: {input_features}")

                # Realizar la predicción con la red neuronal
                prediction = self.model.predict(input_features)
                logging.debug(f"Predicción del modelo (sin procesar): {prediction}")

                # Umbral para considerar una clase como positiva
                threshold = 0.5
                prediction_binary = (prediction > threshold).astype(int)
                logging.debug(f"Predicción binarizada: {prediction_binary}")

                # Invertir la binarización para obtener los números recomendados
                recommended_numbers = self.mlb.inverse_transform(prediction_binary)
                logging.debug(f"Números recomendados (modelo): {recommended_numbers}")

                return list(recommended_numbers[0]) if recommended_numbers else []
            except Exception as e:
                logging.error(f"Error durante la predicción: {e}")
                return []
        else:
            logging.warning(f"No se encontraron recomendaciones para el número {input_number} y el modelo no está entrenado correctamente.")
            return []

    def create_vip_message(self, input_number):
        recommended_numbers = self.predict(input_number)
        current_date = datetime.now()
        day_of_week = current_date.strftime("%A")  # Día de la semana en inglés
        day_of_week_es = self.get_day_in_spanish(day_of_week)  # Traducir a español
        current_time = current_date.strftime("%d/%m/%Y %H:%M:%S")  # Fecha y hora

        # Obtener las vibraciones y datos asociados al día
        day_vibrations_data = self.vibrations_by_day.get(day_of_week_es, {})
        day_digits = day_vibrations_data.get('digits', [])
        day_numbers = day_vibrations_data.get('numbers', [])
        day_parejas = day_vibrations_data.get('parejas', [])

        # Obtener la raíz del número proporcionado si se encuentra en las fórmulas
        root_numbers = self.root_numbers.get(input_number, [])

        # Obtener los números inseparables del número de entrada
        inseparable_numbers = self.inseparable_numbers.get(input_number, [])

        # Obtener la charada asociada al número de entrada
        charada_info = []
        for charada, numbers in self.charadas.items():
            if str(input_number) in numbers:
                charada_info.append(f"{charada}: {' '.join(numbers)}")

        # Crear un diccionario para rastrear las coincidencias de los números a través de los patrones
        pattern_matches = {}
        for numbers in self.mapping.values():
            for number in numbers:
                if number in pattern_matches:
                    pattern_matches[number] += 1
                else:
                    pattern_matches[number] = 1

        # Ordenar los números por la cantidad de coincidencias en los patrones
        most_probable_numbers = sorted(pattern_matches, key=pattern_matches.get, reverse=True)[:5]

        # Obtener los números más atrasados
        most_delayed_numbers = []
        for category, data in self.most_delayed_numbers.items():
            most_delayed_numbers.append(str(data['number']))

        # Verificar si los números más atrasados están en otros patrones
        delayed_in_patterns = []
        for num in most_delayed_numbers:
            if num in recommended_numbers or num in inseparable_numbers or num in root_numbers or num in most_probable_numbers:
                delayed_in_patterns.append(num)

        # Encabezado VIP
        message = "<b>🎉✨ Predicciones Numerológicas VIP ✨🎉</b>\n\n"

        # Números recomendados sin duplicados
        unique_recommended_numbers = list(set(recommended_numbers))
        if unique_recommended_numbers:
            message += "🔮 <b>Números recomendados para el número {}</b>:\n".format(input_number)
            message += '<code>' + ' '.join(unique_recommended_numbers) + '</code>\n\n'

        # Números inseparables del número de entrada
        if inseparable_numbers:
            message += f"🔗 <b>Números inseparables del {input_number}:</b>\n"
            message += '<code>' + ' '.join(inseparable_numbers) + '</code>\n\n'

        # Raíz del número de entrada
        if root_numbers:
            message += f"🌿 <b>Raíz del número {input_number}:</b>\n"
            message += '<code>' + ' '.join(root_numbers) + '</code>\n\n'

        # Información de charadas
        if charada_info:
            message += f"🎲 <b>Charada para el número {input_number}:</b>\n"
            message += '<code>' + '\n'.join(charada_info) + '</code>\n\n'

        # Números más propensos a salir según las coincidencias de patrones
        if most_probable_numbers:
            message += "🌟 <b>Números más fuertes según patrones:</b>\n"
            message += '<code>' + ' '.join(most_probable_numbers) + '</code>\n\n'

        # Números más atrasados que coinciden con otros patrones
        if delayed_in_patterns:
            message += "⏳ <b>Números más atrasados que coinciden con otros patrones:</b>\n"
            message += '<code>' + ' '.join(delayed_in_patterns) + '</code>\n\n'
        else:
            message += "⏳ <b>Números más atrasados:</b>\n"
            message += '<code>' + ' '.join(most_delayed_numbers) + '</code>\n\n'

        # Vibraciones del día
        if day_numbers:
            message += f"📊 <b>Vibraciones para {day_of_week_es}:</b>\n"
            message += '<code>' + ' '.join(day_numbers) + '</code>\n\n'

        # Dígitos semanales obligatorios
        if day_digits:
            message += f"📅 <b>Dígitos semanales obligatorios para {day_of_week_es}:</b>\n"
            message += '<code>' + ' '.join(day_digits) + '</code>\n\n'

        # Parejas del día (solo números con el mismo dígito como 00, 11, 22, ...)
        valid_parejas = [num for num in day_parejas if num[0] == num[1]]
        if valid_parejas:
            message += f"🤝 <b>Parejas para {day_of_week_es}:</b>\n"
            message += '<code>' + ' '.join(valid_parejas) + '</code>\n\n'

        # Sección final con firma
        message += "💼 <b>Predicción VIP Personalizada</b> \n"
        message += f"📅 <i>Fecha y hora de consulta: {current_time}</i>\n"

        return message

    def get_vibrations_for_day(self, day_of_week_es):
        try:
            # Obtener las vibraciones del día desde self.vibrations_by_day
            vibrations = self.vibrations_by_day.get(day_of_week_es, {})
            if not vibrations:
                logging.warning(f"No se encontraron vibraciones para el día {day_of_week_es}.")
                return {}
            logging.info(f"Vibraciones encontradas para {day_of_week_es}: {vibrations}")
            return vibrations
        except Exception as e:
            logging.error(f"Error al obtener las vibraciones para el día {day_of_week_es}: {e}")
            return {}


    def get_day_in_spanish(self, day_in_english):
        days_mapping = {
            "Monday": "Lunes",
            "Tuesday": "Martes",
            "Wednesday": "Miércoles",
            "Thursday": "Jueves",
            "Friday": "Viernes",
            "Saturday": "Sábado",
            "Sunday": "Domingo"
        }
        return days_mapping.get(day_in_english, day_in_english)

# Class Conversar
class Conversar:
    def __init__(self, db):
        """
        Inicializa la clase Conversar con una conexión a la base de datos, un modelo no entrenado y un tokenizer.

        Args:
            db (object): Conexión a la base de datos para cargar o guardar datos relacionados con el bot.
        """
        if db is None or not hasattr(db, 'get_all_messages'):
            raise ValueError("La conexión a la base de datos no está configurada correctamente o es inválida.")

        self.db = db  # Conexión a la base de datos

        # Inicialización del modelo y tokenizer (se cargarán más tarde)
        self.model = None
        self.tokenizer = None

        # Variables para los modelos de inferencia
        self.encoder_model = None
        self.decoder_model = None

        # Variable que indica si el modelo está entrenado
        self.is_trained = False

        # Longitud máxima de las secuencias de entrada y salida
        self.max_encoder_seq_length = 20
        self.max_decoder_seq_length = 20

        # Tamaño de la capa LSTM
        self.latent_dim = 256

        # Límite de vocabulario a las 10,000 palabras más frecuentes
        self.num_words = 10000

        # Definir nombres de archivos específicos para Conversar
        self.model_filename = 'conversational_model_conversar.keras'
        self.tokenizer_filename = 'tokenizer_conversar.pkl'
        self.ajuste_fino_file = 'ajuste_fino_datos_conversar.pkl'

        # Cargar el modelo y el tokenizer si están disponibles
        self.cargar_modelo()

        # Validar si el modelo y el tokenizer se cargaron correctamente
        if self.model and self.tokenizer:
            self.is_trained = True
            logging.info("Modelo y tokenizer cargados exitosamente.")
        else:
            self.is_trained = False
            logging.warning("Modelo o tokenizer no cargados. Entrenamiento será necesario.")

    
    def cargar_modelo_y_tokenizer(self):
        """Carga el modelo conversacional y el tokenizer."""
        try:
            start_time = time.time()

            # Cargar el modelo desde un archivo
            logging.info(f"Intentando cargar el modelo conversacional desde '{self.model_filename}'...")
            self.model = tf.keras.models.load_model(self.model_filename)
            logging.info(f"Modelo conversacional cargado exitosamente en {time.time() - start_time:.2f} segundos.")
            
            # Cargar el tokenizer desde un archivo
            logging.info(f"Intentando cargar el tokenizer desde '{self.tokenizer_filename}'...")
            with open(self.tokenizer_filename, 'rb') as f:
                self.tokenizer = pickle.load(f)

            # Verificar si el tokenizer fue cargado correctamente
            if self.tokenizer:
                logging.info("Tokenizer cargado exitosamente.")
            else:
                logging.warning("El tokenizer no se cargó correctamente.")
            
            # Indicar que el modelo ha sido cargado y está listo para usarse
            self.is_trained = True
            logging.info("El modelo y el tokenizer están listos para usarse.")

        except Exception as e:
            # Registrar el error en los logs
            logging.error(f"Error al cargar el modelo o el tokenizer: {e}")
            self.is_trained = False

    def cargar_modelo(self):
        """Carga el modelo conversacional y el tokenizer, y verifica si es compatible."""
        try:
            start_time = time.time()
            logging.info(f"Intentando cargar el modelo conversacional desde '{self.model_filename}'...")
            self.model = tf.keras.models.load_model(self.model_filename, compile=False)
            logging.info(f"Modelo conversacional cargado exitosamente en {time.time() - start_time:.2f} segundos.")

            # Verificar que el modelo no es None
            if self.model is None:
                logging.error("El modelo no se cargó correctamente. self.model es None.")
                self.is_trained = False
                return

            # Intentar cargar el tokenizer desde el archivo
            logging.info(f"Intentando cargar el tokenizer desde '{self.tokenizer_filename}'...")
            with open(self.tokenizer_filename, 'rb') as f:
                self.tokenizer = pickle.load(f)
            logging.info("Tokenizer cargado exitosamente.")

            # Verificar si el modelo es compatible
            if not self.verificar_compatibilidad_modelo():
                logging.error("El modelo cargado no es compatible con el código actual.")
                self.is_trained = False
                return

            # Configurar los modelos de inferencia
            self.setup_inference_models()

            # Indicar que el modelo ha sido cargado y está listo para usarse
            self.is_trained = True
            logging.info("El modelo y el tokenizer están listos para usarse.")
        except Exception as e:
            logging.error(f"Error al cargar el modelo o el tokenizer: {e}")
            self.is_trained = False


    def verificar_compatibilidad_modelo(self):
        """Verifica si el modelo cargado tiene las capas necesarias con los nombres correctos."""
        required_layers = ['encoder_inputs', 'encoder_lstm', 'decoder_inputs', 'decoder_embedding', 'decoder_lstm', 'decoder_dense']
        existing_layers = [layer.name for layer in self.model.layers]

        missing_layers = [layer for layer in required_layers if layer not in existing_layers]
        if missing_layers:
            logging.error(f"Faltan las siguientes capas necesarias en el modelo: {missing_layers}")
            return False
        else:
            logging.info("El modelo cargado es compatible.")
            return True

    
    def generate_response(self, input_text):
        """Genera una respuesta usando primero GPT-4 y luego, en caso de fallo, el modelo local."""
        logger.info(f"Generando respuesta para el mensaje: {input_text}")

        # Intentar obtener la respuesta de GPT-4
        try:
            gpt_response = self.gpt4o_generate_response(input_text)
        except Exception as e:
            gpt_response = None
            logger.error(f"Error al intentar generar respuesta desde GPT-4o: {e}")

        # Si GPT-4 genera una respuesta válida, la usamos
        if gpt_response:
            logger.info(f"Respuesta de GPT-4o recibida: {gpt_response}")
            # Almacenar para ajuste fino si la respuesta es válida y adecuada
            if len(gpt_response.split()) > 3:  # Asegurarse de que la respuesta es suficientemente larga
                self.almacenar_para_ajuste_fino(input_text, gpt_response)
                logger.info("Respuesta almacenada para ajuste fino.")
            else:
                logger.warning("Respuesta de GPT-4o es demasiado corta. No se almacenará para ajuste fino.")
            return gpt_response

        # Si no se obtuvo respuesta de GPT-4, generar con el modelo local
        logger.warning("No se obtuvo respuesta de GPT-4o. Generando con el modelo local.")

        # Verificar si el modelo local está entrenado antes de proceder
        if not self.is_trained or self.model is None:
            logger.error("El modelo local no está entrenado o no ha sido cargado. No se puede generar una respuesta coherente.")
            return (
                "El modelo local aún no está listo para generar una respuesta completa. "
                "Estamos trabajando para mejorar esta función."
            )

        # Verificar que el tokenizer esté inicializado
        if self.tokenizer is None:
            logger.error("El tokenizer no ha sido inicializado. No se puede procesar el mensaje.")
            return "El modelo local no está listo para procesar este mensaje."

        # Intentar generar la respuesta con el modelo local
        try:
            logger.info("Generando respuesta con el modelo local.")
            local_response = self.model_generate_response(input_text)
        except Exception as e:
            logger.error(f"Error al generar respuesta con el modelo local: {e}")
            return "Se ha producido un error al generar la respuesta local. Por favor, intenta más tarde."

        # Post-procesar la respuesta para mejorar la coherencia
        final_response = self.post_process_response(local_response)

        # Verificar si la respuesta local es válida
        if not final_response or len(final_response.split()) < 3:
            logger.error("La respuesta generada localmente no fue coherente o fue demasiado corta.")
            return "No he podido generar una respuesta adecuada. Por favor, intenta reformular tu pregunta."

        logger.info(f"Respuesta local generada exitosamente: {final_response}")

        # Retornamos la respuesta local procesada
        return final_response


    def gpt4o_generate_response(self, input_text):
        api_url = 'https://api.openai.com/v1/chat/completions'  # URL actualizada para chat completions
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}'
        }
        data = {
            'model': 'gpt-4o-mini',
            'messages': [
                {
                    "role": "system", 
                    "content": (
                        "Eres un experto científico de numerología especializado en análisis de patrones numéricos y predicciones de loterías. "
                        "Proporciona respuestas precisas basadas en reglas matemáticas, con un enfoque analítico y siempre en español, "
                        "a menos que el usuario hable en otro idioma."
                    )
                },
                {"role": "user", "content": input_text}
            ],
            'max_tokens': 120,  # Limitar la longitud de la respuesta
            'temperature': 0.6  # Reducir la creatividad para mantener respuestas coherentes y precisas
        }

        response = requests.post(api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            logger.error(f"Error al generar la respuesta desde OpenAI: {response.status_code}, {response.text}")
            return None
    
    def almacenar_para_ajuste_fino(self, input_text, output_text):
        """Almacena las entradas y salidas de GPT-4 para realizar un ajuste fino, adaptado al modelo Encoder-Decoder."""
        try:
            # Validar que el texto de entrada y la respuesta no sean vacíos o incoherentes
            if not input_text or not output_text:
                logging.warning("El texto de entrada o la respuesta están vacíos. No se almacenarán.")
                return

            # Verificar si la respuesta es suficientemente larga para el ajuste fino
            if len(output_text.split()) < 3:
                logging.warning(f"Respuesta demasiado corta, no se almacenará para ajuste fino: {output_text}")
                return

            # Limpiar los textos
            cleaned_input = self.clean_text(input_text)
            cleaned_response = self.clean_text(output_text)

            # Agregar tokens de inicio y fin a la respuesta
            cleaned_response = '<start> ' + cleaned_response + ' <end>'

            # Crear los datos que serán almacenados
            data = {"input": cleaned_input, "response": cleaned_response}

            # Nombre del archivo para el ajuste fino del modelo conversacional
            ajuste_fino_file = 'ajuste_fino_datos_conversar.pkl'
            
            # Verificar si el archivo ya existe para evitar duplicados
            data_exists = False
            if os.path.exists(ajuste_fino_file):
                with open(ajuste_fino_file, 'rb') as f:
                    try:
                        while True:
                            existing_data = pickle.load(f)
                            if existing_data == data:
                                data_exists = True
                                logging.info("Datos duplicados encontrados. No se almacenarán nuevamente.")
                                break
                    except EOFError:
                        pass  # Fin del archivo alcanzado
                    except pickle.UnpicklingError as e:
                        logging.error(f"Error al deserializar datos en el archivo existente: {e}")

            # Si no hay datos duplicados, guardarlos
            if not data_exists:
                with open(ajuste_fino_file, 'ab') as f:
                    pickle.dump(data, f)
                logging.info("Datos almacenados para ajuste fino correctamente.")

            # Opcional: Verificar el tamaño del archivo tras cada escritura
            file_size = os.path.getsize(ajuste_fino_file)
            logging.info(f"El tamaño actual del archivo de ajuste fino es {file_size} bytes.")

        except Exception as e:
            logging.error(f"Error al almacenar datos para ajuste fino: {e}")

        
    def limpiar_datos(self, nuevos_datos):
        """
        Limpia los datos nuevos aplicando la función clean_text.

        Args:
            nuevos_datos (list): Lista de diccionarios con claves 'input' y 'response'.

        Returns:
            list: Lista de diccionarios con los datos limpios.
        """
        try:
            if not nuevos_datos or len(nuevos_datos) == 0:
                logging.warning("Los datos proporcionados están vacíos o son nulos.")
                return []

            datos_limpios = []
            for data in nuevos_datos:
                if isinstance(data, dict) and 'input' in data and 'response' in data:
                    # Limitar la longitud de los mensajes
                    if len(data['input']) <= 200 and len(data['response']) <= 200:
                        cleaned_input = self.clean_text(data['input'])
                        cleaned_response = self.clean_text(data['response'])
                        mensaje_hash = hash(cleaned_input + cleaned_response)
                        if mensaje_hash in self.processed_messages:
                            logging.info("Mensaje duplicado, se omite.")
                            continue
                        self.processed_messages.add(mensaje_hash)
                        datos_limpios.append({'input': cleaned_input, 'response': cleaned_response})
                    else:
                        logging.warning(f"Mensaje demasiado largo, se omite: {data}")
                else:
                    logging.error(f"Formato de datos incorrecto o faltan claves: {data}")

            if len(datos_limpios) == 0:
                logging.warning("Todos los datos proporcionados estaban vacíos o no válidos después de la limpieza.")
                return []

            logging.info(f"Datos limpios generados: {len(datos_limpios)} ejemplos.")
            return datos_limpios

        except TypeError as te:
            logging.error(f"Tipo de datos no válido al limpiar los datos: {te}")
            return []
        except Exception as e:
            logging.error(f"Error inesperado al limpiar los datos para ajuste fino: {e}")
            return []

    def realizar_ajuste_fino(self, epochs=2):
        """Realiza el ajuste fino del modelo conversacional utilizando los datos de ajuste fino."""
        logger.info("[Conversar] Iniciando el ajuste fino del modelo conversacional.")

        try:
            # Nombre del archivo de ajuste fino para el modelo conversacional
            ajuste_fino_file = self.ajuste_fino_file

            # Verificar si el archivo de ajuste fino existe
            if not os.path.exists(ajuste_fino_file):
                logger.warning(f"[Conversar] No se encontró el archivo de ajuste fino '{ajuste_fino_file}'.")
                return  # No se puede proceder sin datos de ajuste fino

            # Preparar los datos del archivo de ajuste fino
            data_pairs = []
            with open(ajuste_fino_file, 'rb') as f:
                try:
                    while True:
                        data = pickle.load(f)
                        if 'input' in data and 'response' in data:
                            data_pairs.append((data['input'], data['response']))
                        else:
                            logger.warning(f"[Conversar] Datos inválidos encontrados en el archivo: {data}")
                except EOFError:
                    pass  # Fin del archivo alcanzado
                except pickle.UnpicklingError as e:
                    logger.error(f"[Conversar] Error al deserializar datos del archivo '{ajuste_fino_file}': {e}")
                    return

            if not data_pairs:
                logger.warning("[Conversar] No se encontraron datos de ajuste fino válidos.")
                return

            # Actualizar el tokenizer y preparar los datos
            input_texts = []
            target_texts = []

            for input_text, target_text in data_pairs:
                input_texts.append(self.clean_text(input_text))
                target_texts.append('<start> ' + self.clean_text(target_text) + ' <end>')

            # **Cargar el tokenizer existente si no está inicializado**
            if self.tokenizer is None:
                try:
                    with open(self.tokenizer_filename, 'rb') as f:
                        self.tokenizer = pickle.load(f)
                    logger.info("Tokenizer cargado exitosamente para ajuste fino.")
                except FileNotFoundError:
                    logger.warning("Tokenizer no encontrado. Se creará uno nuevo.")
                    self.tokenizer = Tokenizer(num_words=self.num_words, oov_token="<OOV>")

            # Actualizar el tokenizer con los nuevos datos
            self.tokenizer.fit_on_texts(input_texts + target_texts)

            # Convertir textos a secuencias
            encoder_input_sequences = self.tokenizer.texts_to_sequences(input_texts)
            decoder_input_sequences = self.tokenizer.texts_to_sequences(target_texts)

            # Preparar las secuencias de entrada y salida
            encoder_input_data = pad_sequences(encoder_input_sequences, maxlen=self.max_encoder_seq_length, padding='post')
            decoder_input_data = pad_sequences(decoder_input_sequences, maxlen=self.max_decoder_seq_length, padding='post')
            decoder_target_sequences = [seq[1:] for seq in decoder_input_sequences]
            decoder_target_data = pad_sequences(decoder_target_sequences, maxlen=self.max_decoder_seq_length, padding='post')

            logger.info(f"[Conversar] Se encontraron {len(input_texts)} ejemplos de ajuste fino para procesar.")

            # Asegurarse de que el modelo está construido
            if self.model is None:
                logger.info("[Conversar] El modelo no está construido. Construyendo el modelo...")
                self.build_model()

            # Entrenar el modelo con los nuevos datos
            logger.info(f"[Conversar] Realizando ajuste fino con {len(encoder_input_data)} ejemplos.")
            self.model.fit(
                [encoder_input_data, decoder_input_data],
                np.expand_dims(decoder_target_data, -1),
                batch_size=64,
                epochs=epochs,
                validation_split=0.2
            )

            logger.info("[Conversar] Ajuste fino completado exitosamente.")

            # Actualizar los modelos de inferencia después del ajuste fino
            self.setup_inference_models()

            # **Guardar el modelo y el tokenizer actualizados**
            self.guardar_modelo()

        except Exception as e:
            logger.error(f"[Conversar] Error durante el ajuste fino del modelo: {e}")


    def model_generate_response(self, input_text):
        """Genera una respuesta utilizando el modelo Encoder-Decoder en modo inferencia."""
        logger = logging.getLogger(__name__)
        logger.info(f"Generando respuesta local para el mensaje: {input_text}")

        # Verificar si los modelos de inferencia están configurados
        if not hasattr(self, 'encoder_model') or not hasattr(self, 'decoder_model') or self.encoder_model is None or self.decoder_model is None:
            logger.info("Configurando modelos de inferencia para el Encoder-Decoder.")
            self.setup_inference_models()
            if self.encoder_model is None or self.decoder_model is None:
                logger.error("No se pudieron configurar los modelos de inferencia.")
                return "Error al configurar los modelos de inferencia."

        # Preprocesar el texto de entrada
        input_seq = self.tokenizer.texts_to_sequences([self.clean_text(input_text)])
        input_seq = pad_sequences(input_seq, maxlen=self.max_encoder_seq_length, padding='post')

        # Obtener las salidas y estados del encoder
        encoder_outputs, state_h, state_c = self.encoder_model.predict(input_seq)

        # Inicializar la secuencia del decoder
        target_seq = np.array([[self.tokenizer.word_index.get('<start>', 1)]])

        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq, encoder_outputs, state_h, state_c]
            )

            # Obtener la palabra más probable
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.tokenizer.index_word.get(sampled_token_index, '')

            if (sampled_word == '<end>' or len(decoded_sentence.split()) > self.max_decoder_seq_length):
                stop_condition = True
            else:
                decoded_sentence += ' ' + sampled_word

            # Actualizar la secuencia de entrada del decoder
            target_seq = np.array([[sampled_token_index]])

            # Actualizar los estados
            state_h, state_c = h, c

        return decoded_sentence.strip()

    
    def post_process_response(self, response):
        """
        Aplica filtros y ajustes finales a la respuesta generada para mejorar la coherencia.
        Elimina repeticiones, corrige errores gramaticales simples y mejora la estructura.
        """
        logger = logging.getLogger(__name__)

        # Eliminar etiquetas <OOV> y limpiar espacios resultantes
        response = response.replace('<OOV>', '').strip()
        response = re.sub(r'\s+', ' ', response)  # Eliminar espacios adicionales

        # Eliminar repeticiones de palabras consecutivas
        words = response.split()
        filtered_words = []
        for i, word in enumerate(words):
            if i > 0 and word.lower() == words[i-1].lower():  # Ignorar repeticiones consecutivas (case-insensitive)
                continue
            filtered_words.append(word)

        # Reconstruir la respuesta filtrada
        response = ' '.join(filtered_words)

        # Corregir la gramática y estructura con spaCy (si está disponible)
        if 'nlp' in globals() and nlp:  # Verificación más robusta para spaCy
            doc = nlp(response)
            response = ' '.join([token.text for token in doc])

            # Eliminar frases redundantes (opcional)
            sentences = list(doc.sents)
            filtered_sentences = []
            for i, sent in enumerate(sentences):
                if i > 0 and str(sentences[i-1]).strip() == str(sent).strip():
                    continue  # Omitir oraciones redundantes
                filtered_sentences.append(str(sent))
            response = ' '.join(filtered_sentences)

        # Corregir posibles errores de puntuación y gramática básica
        response = re.sub(r'\s+', ' ', response)  # Unificar múltiples espacios
        response = re.sub(r'\.\.+', '.', response)  # Limitar múltiples puntos a un solo punto
        response = re.sub(r'\s,', ',', response)    # Corregir espacio antes de las comas
        response = re.sub(r'\s\.', '.', response)   # Corregir espacio antes de los puntos
        response = re.sub(r' ,', ',', response)     # Corregir coma con espacio previo incorrecto
        response = re.sub(r'\?+', '?', response)    # Limitar múltiples signos de interrogación a uno

        logger.info(f"Respuesta post-procesada: {response}")
        return response

    def build_model(self):
        """Construye un modelo secuencial para predecir el próximo token en una secuencia."""
        try:
            logging.info("[Conversar] Construyendo un modelo de lenguaje secuencial...")

            # Tamaño del vocabulario y dimensiones
            vocab_size = len(self.tokenizer.word_index) + 1
            embedding_dim = 256  # Dimensión de embedding para las palabras
            latent_dim = 512     # Dimensión de la capa LSTM

            logging.info(f"[Conversar] Tamaño del vocabulario: {vocab_size}")
            logging.info(f"[Conversar] Dimensión de embedding: {embedding_dim}")
            logging.info(f"[Conversar] Dimensión latente (LSTM units): {latent_dim}")

            # Definir la arquitectura del modelo
            self.model = tf.keras.models.Sequential()

            # Capa de embedding
            self.model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=self.max_sequence_length))

            # Capa LSTM
            self.model.add(tf.keras.layers.LSTM(latent_dim, return_sequences=False))

            # Capa densa para predecir el siguiente token
            self.model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

            # Compilar el modelo
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            logging.info("[Conversar] Modelo secuencial construido y compilado exitosamente.")

        except Exception as e:
            logging.error(f"[Conversar] Error al construir el modelo: {e}")
            self.model = None


    def setup_inference_models(self):
        """Configura los modelos de inferencia para el Encoder y el Decoder."""
        try:
            # Dimensión latente ajustada (el doble por la LSTM bidireccional)
            latent_dim = 512
            decoder_latent_dim = latent_dim * 2

            # Encoder - Modelo de inferencia
            encoder_inputs = self.model.input[0]  # Entrada del encoder en el modelo de entrenamiento
            encoder_embedding = self.model.get_layer('encoder_embedding')(encoder_inputs)
            encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.model.get_layer('bidirectional').output
            state_h_enc = Concatenate()([forward_h, backward_h])
            state_c_enc = Concatenate()([forward_c, backward_c])
            encoder_states = [state_h_enc, state_c_enc]
            self.encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)
            logging.info("[Conversar] Modelo de inferencia del encoder configurado.")

            # Decoder - Modelo de inferencia
            decoder_inputs = self.model.input[1]  # Entrada del decoder en el modelo de entrenamiento
            decoder_state_input_h = Input(shape=(decoder_latent_dim,), name='decoder_state_input_h')
            decoder_state_input_c = Input(shape=(decoder_latent_dim,), name='decoder_state_input_c')
            decoder_hidden_state_input = Input(shape=(None, latent_dim * 2), name='decoder_hidden_state_input')

            # Embedding
            decoder_embedding = self.model.get_layer('decoder_embedding')(decoder_inputs)

            # LSTM del decoder
            decoder_lstm = self.model.get_layer('decoder_lstm')
            decoder_outputs, state_h, state_c = decoder_lstm(
                decoder_embedding, initial_state=[decoder_state_input_h, decoder_state_input_c]
            )

            # Atención
            attn_layer = self.model.get_layer('attention_dot')
            attn_out = attn_layer([decoder_outputs, decoder_hidden_state_input])

            # Concatenar
            decoder_concat_input = self.model.get_layer('decoder_concat')([decoder_outputs, attn_out])

            # Capa densa
            decoder_dense = self.model.get_layer('decoder_dense')
            decoder_outputs = decoder_dense(decoder_concat_input)

            self.decoder_model = Model(
                [decoder_inputs, decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
                [decoder_outputs, state_h, state_c]
            )
            logging.info("[Conversar] Modelo de inferencia del decoder configurado.")

        except Exception as e:
            logging.error(f"[Conversar] Error al configurar los modelos de inferencia: {e}")
            self.encoder_model = None
            self.decoder_model = None

    
    def guardar_modelo(self):
        """Guarda el modelo entrenado y el tokenizer."""
        try:
            # Guardar el modelo conversacional con el nombre adecuado
            self.model.save('conversational_model_conversar.keras')
            
            # Guardar el tokenizer con el nombre adecuado
            with open('tokenizer_conversar.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
                
            logging.info("[Conversar] Modelo y tokenizer guardados exitosamente.")
        except Exception as e:
            logging.error(f"[Conversar] Error al guardar el modelo o el tokenizer: {e}")


    def prepare_data(self):
        """Prepara los datos para el modelo Encoder-Decoder usando el archivo de ajuste fino de conversaciones."""
        logging.info("[Conversar] Iniciando la preparación de datos para el modelo Encoder-Decoder...")

        try:
            # Lista para almacenar los pares de datos
            data_pairs = []

            # Nombre del archivo para ajuste fino de conversaciones
            ajuste_fino_file = self.ajuste_fino_file

            # Verificar si el archivo de ajuste fino existe
            if os.path.exists(ajuste_fino_file):
                logging.info(f"[Conversar] Cargando datos de ajuste fino desde '{ajuste_fino_file}'.")
                with open(ajuste_fino_file, 'rb') as f:
                    try:
                        while True:
                            data = pickle.load(f)
                            if 'input' in data and 'response' in data:
                                data_pairs.append((data['input'], data['response']))
                            else:
                                logging.warning(f"[Conversar] Datos inválidos encontrados en el archivo: {data}")
                    except EOFError:
                        pass  # Fin del archivo alcanzado
                    except pickle.UnpicklingError as e:
                        logging.error(f"[Conversar] Error al deserializar datos del archivo '{ajuste_fino_file}': {e}")
            else:
                logging.warning(f"[Conversar] No se encontró el archivo de datos para ajuste fino '{ajuste_fino_file}'.")

            # Verificar si tenemos datos para preparar
            if not data_pairs:
                logging.error("[Conversar] No se encontraron datos para entrenar el modelo.")
                return False

            logging.info(f"[Conversar] Se encontraron {len(data_pairs)} pares de datos para preparar.")

            # Limpiar y procesar los mensajes y respuestas
            input_texts = []
            target_texts = []

            for input_text, target_text in data_pairs:
                input_text = self.clean_text(input_text)
                target_text = self.clean_text(target_text)

                # Verificar y agregar tokens de inicio y fin a las respuestas si es necesario
                if not target_text.startswith('<start>'):
                    target_text = '<start> ' + target_text
                if not target_text.endswith('<end>'):
                    target_text = target_text + ' <end>'

                input_texts.append(input_text)
                target_texts.append(target_text)

            # Verificar si tenemos suficientes pares de datos
            if not input_texts or not target_texts:
                logging.error("[Conversar] No se pudieron generar pares de datos válidos para entrenar el modelo.")
                return False

            # Cargar o crear el tokenizer
            if self.tokenizer is None:
                self.tokenizer = Tokenizer(num_words=self.num_words, oov_token="<OOV>")
                logging.info("[Conversar] Tokenizer creado.")

            # Actualizar el tokenizer con el vocabulario combinado
            self.tokenizer.fit_on_texts(input_texts + target_texts)
            logging.info("[Conversar] Tokenizer ajustado con los textos de entrada y salida.")

            # Actualizar num_words
            self.num_words = min(self.num_words, len(self.tokenizer.word_index) + 1)
            logging.info(f"[Conversar] Tamaño del vocabulario establecido en: {self.num_words}")

            # Convertir textos a secuencias
            encoder_input_sequences = self.tokenizer.texts_to_sequences(input_texts)
            decoder_input_sequences = self.tokenizer.texts_to_sequences(target_texts)

            # Crear decoder_target_sequences desplazadas
            decoder_target_sequences = []
            for seq in decoder_input_sequences:
                decoder_target_sequences.append(seq[1:])  # Desplazar un paso hacia la izquierda

            # Padding de las secuencias
            self.max_encoder_seq_length = max([len(seq) for seq in encoder_input_sequences])
            self.max_decoder_seq_length = max([len(seq) for seq in decoder_input_sequences])
            logging.info(f"[Conversar] Longitud máxima de secuencia del encoder: {self.max_encoder_seq_length}")
            logging.info(f"[Conversar] Longitud máxima de secuencia del decoder: {self.max_decoder_seq_length}")

            encoder_input_data = pad_sequences(encoder_input_sequences, maxlen=self.max_encoder_seq_length, padding='post')
            decoder_input_data = pad_sequences(decoder_input_sequences, maxlen=self.max_decoder_seq_length, padding='post')
            decoder_target_data = pad_sequences(decoder_target_sequences, maxlen=self.max_decoder_seq_length, padding='post')

            # Almacenar los datos preparados
            self.encoder_input_data = encoder_input_data
            self.decoder_input_data = decoder_input_data
            self.decoder_target_data = decoder_target_data

            logging.info("[Conversar] Datos preparados correctamente para el modelo Encoder-Decoder.")
            return True

        except Exception as e:
            logging.error(f"[Conversar] Error durante la preparación de los datos: {e}")
            return False

    def clean_text(self, text):
        """Elimina emojis, caracteres especiales, convierte a minúsculas, y normaliza espacios."""
        
        # Regex para eliminar emojis y algunos caracteres unicode adicionales
        emoji_pattern = re.compile(
            "[" u"\U0001F600-\U0001F64F"  # Emoticones
            u"\U0001F300-\U0001F5FF"  # Símbolos y pictogramas
            u"\U0001F680-\U0001F6FF"  # Transportes y símbolos de mapas
            u"\U0001F1E0-\U0001F1FF"  # Banderas
            u"\U00002702-\U000027B0"  # Otros pictogramas
            u"\U000024C2-\U0001F251"  # Símbolos adicionales
            "]+", flags=re.UNICODE)
        
        # Eliminar emojis
        text = emoji_pattern.sub(r'', text)

        # Eliminar caracteres especiales, excepto letras, números y espacios
        text = re.sub(r'[^\w\s]', '', text)

        # Reemplazar múltiples espacios por un solo espacio
        text = re.sub(r'\s+', ' ', text)

        # Convertir el texto a minúsculas y eliminar espacios extras
        return text.strip().lower()

    def tokenize_messages(self, messages):
        try:
            if not messages or len(messages) == 0:
                logging.error("No se proporcionaron mensajes para tokenizar.")
                return None

            self.tokenizer = Tokenizer(oov_token="<OOV>")  # No establecer num_words aquí
            valid_messages = [msg for msg in messages if msg.strip()]
            self.tokenizer.fit_on_texts(valid_messages)

            # Actualizar num_words después de ajustar el tokenizer
            self.num_words = len(self.tokenizer.word_index) + 1

            sequences = self.tokenizer.texts_to_sequences(valid_messages)
            return sequences

        except Exception as e:
            logging.error(f"Error al tokenizar los mensajes: {e}")
            return None

    def generate_sequences_and_labels(self, sequences):
        X, y = []

        num_classes = self.num_words
        if num_classes is None or num_classes <= 0:
            logging.error("El número de clases (num_words en el tokenizer) no está definido correctamente.")
            return None, None

        try:
            for seq_index, seq in enumerate(sequences):
                if len(seq) == 0:
                    continue

                for i in range(1, len(seq)):
                    input_sequence = seq[:i]
                    target_word = seq[i] - 1  # Ajustar el índice

                    input_sequence_padded = pad_sequences([input_sequence], maxlen=self.max_sequence_length)[0]
                    X.append(input_sequence_padded)
                    y.append(target_word)

            X_np = np.array(X)
            y_np = np.array(y)

            if X_np.shape[0] == 0 or y_np.shape[0] == 0:
                logging.error("No se generaron secuencias o etiquetas válidas.")
                return None, None

            return X_np, y_np

        except Exception as e:
            logging.error(f"Error durante la generación de secuencias y etiquetas: {e}")
            return None, None

    def train(self, epochs=10, batch_size=64, validation_split=0.2):
        """Entrena el modelo conversacional con los mensajes de los usuarios."""
        try:
            logging.info("===== [Conversar] Iniciando el entrenamiento del modelo conversacional =====")

            # Preparar los datos
            logging.info("[Conversar] Preparando los datos para el entrenamiento...")

            # Verificar si existe el archivo de ajuste fino
            if not os.path.exists(self.ajuste_fino_file):
                logging.warning(f"No se encontró el archivo '{self.ajuste_fino_file}', obteniendo datos de la base de datos.")
                mensajes = self.db.get_all_messages()  # Obtener mensajes de la base de datos

                if not mensajes:
                    logging.error("No se encontraron mensajes en la base de datos. Abortando entrenamiento.")
                    self.is_trained = False
                    return

                # Limpiar mensajes duplicados y vacíos
                mensajes = list(set([mensaje.strip() for mensaje in mensajes if mensaje.strip()]))

                # Guardar los mensajes en el archivo de ajuste fino para futuras ejecuciones
                with open(self.ajuste_fino_file, 'wb') as f:
                    pickle.dump(mensajes, f)
                logging.info(f"Datos guardados en '{self.ajuste_fino_file}' para ajuste fino futuro.")
            else:
                logging.info(f"Cargando datos desde el archivo '{self.ajuste_fino_file}'.")
                with open(self.ajuste_fino_file, 'rb') as f:
                    mensajes = pickle.load(f)

            # Limpiar y preparar los mensajes
            cleaned_input = [self.clean_text(mensaje) for mensaje in mensajes]

            # Asegurarse de que haya suficientes datos
            if not cleaned_input:
                logging.error("No se encontraron datos válidos para entrenar. Abortando entrenamiento.")
                self.is_trained = False
                return

            # Tokenizar los mensajes
            if self.tokenizer is None:
                self.tokenizer = Tokenizer(num_words=self.num_words, oov_token="<OOV>")
                self.tokenizer.fit_on_texts(cleaned_input)

            # Convertir los textos a secuencias
            input_sequences = self.tokenizer.texts_to_sequences(cleaned_input)

            # Preparar las secuencias y etiquetas
            X, y = [], []
            for seq in input_sequences:
                for i in range(1, len(seq)):
                    X.append(seq[:i])  # Secuencia de entrada hasta el i-ésimo token
                    y.append(seq[i])   # El token a predecir es el i-ésimo token

            # Aplicar padding a las secuencias
            self.max_sequence_length = max([len(seq) for seq in X])
            X = pad_sequences(X, maxlen=self.max_sequence_length, padding='post')
            y = np.array(y)

            # Asegurarse de que el modelo está construido
            if self.model is None:
                logging.info("[Conversar] El modelo no está construido. Construyendo el modelo...")
                self.build_model()

            # Verificar que el modelo se construyó correctamente
            if self.model is None:
                logging.error("[Conversar] Error al construir el modelo. Abortando entrenamiento.")
                self.is_trained = False
                return

            # Iniciar el entrenamiento
            logging.info(f"[Conversar] Iniciando el entrenamiento del modelo con {epochs} épocas y batch_size de {batch_size}.")

            # Entrenar el modelo
            history = self.model.fit(
                X, y,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split
            )

            logging.info("[Conversar] Entrenamiento completado con éxito.")

            # Marcar el modelo como entrenado
            self.is_trained = True
            logging.info("[Conversar] El modelo ha sido marcado como entrenado.")

            # **Guardar el modelo y el tokenizer actualizados**
            self.guardar_modelo()

        except Exception as e:
            logging.error(f"[Conversar] Error durante el entrenamiento del modelo: {e}")
            self.is_trained = False

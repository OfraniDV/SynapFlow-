#model.py

# Importaciones de librerías estándar
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

import json

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
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape


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

            # **Generar características y etiquetas a partir de las fórmulas**
            nuevas_X = [self.generate_features(input_number) for input_number in inputs]
            nuevas_X = pad_sequences(nuevas_X, maxlen=self.max_sequence_length)
            nuevas_y = self.mlb.fit_transform(respuestas)

            # **Verificar la forma de las características para que sea compatible con CNN 1D**
            if len(nuevas_X.shape) == 2:
                nuevas_X = np.expand_dims(nuevas_X, axis=-1)  # Añadir una tercera dimensión para la capa CNN 1D

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

            # **Realizar el ajuste fino del modelo basado en las fórmulas**
            logging.info("Comenzando el ajuste fino del modelo.")
            self.model.fit(nuevas_X, nuevas_y, epochs=2, batch_size=10)
            logging.info("Ajustes finos aplicados exitosamente basados en las fórmulas.")

            # Guardar el modelo ajustado
            self.model.save('numerology_model_finetuned.keras')
            logging.info("Modelo ajustado guardado como 'numerology_model_finetuned.keras'.")

        except Exception as e:
            logging.error(f"Error al aplicar ajustes finos: {e}")
            
    def cargar_estado_procesamiento(self):
        """Carga el ID de la última fila procesada desde el archivo 'estado_procesamiento.json'."""
        try:
            if not os.path.exists('estado_procesamiento.json'):
                logging.info("Archivo 'estado_procesamiento.json' no encontrado. Creando uno nuevo con valor inicial 0.")
                with open('estado_procesamiento.json', 'w') as file:
                    json.dump({"last_processed_id": 0}, file)
                return 0
            else:
                with open('estado_procesamiento.json', 'r') as file:
                    data = json.load(file)
                    return data.get('last_processed_id', 0)
        except Exception as e:
            logging.error(f"Error al cargar el estado de procesamiento: {e}")
            return 0

    def actualizar_estado_procesamiento(self, last_processed_id):
        """Actualiza el ID de la última fila procesada en el archivo 'estado_procesamiento.json'."""
        try:
            with open('estado_procesamiento.json', 'w') as file:
                json.dump({"last_processed_id": last_processed_id}, file)
            logging.info(f"Estado de procesamiento actualizado a ID: {last_processed_id}")
        except Exception as e:
            logging.error(f"Error al actualizar el estado de procesamiento: {e}")

    def ajuste_fino(self):
        """Realiza un ajuste fino del modelo utilizando mensajes nuevos desde la última fila procesada."""
        try:
            # Cargar el último ID procesado desde el archivo 'estado_procesamiento.json'
            last_processed_id = self.cargar_estado_procesamiento()
            logging.info(f"⚙️ [Ajuste Fino] Último ID procesado: {last_processed_id}")

            # Obtener los nuevos mensajes desde la base de datos
            nuevos_mensajes = self.db.get_messages_since(last_processed_id)

            if not nuevos_mensajes:
                logging.info("🔄 [Ajuste Fino] No se encontraron nuevos mensajes para el ajuste fino.")
                return

            logging.info(f"🔄 [Ajuste Fino] Se encontraron {len(nuevos_mensajes)} nuevos mensajes para el ajuste fino.")

            # Obtener todas las fórmulas de la base de datos para extraer reglas
            logging.info("📝 [Ajuste Fino] Obteniendo fórmulas para extraer nuevas reglas...")
            formulas = self.db.get_all_formulas()
            if not formulas:
                logging.warning("⚠️ [Ajuste Fino] No se encontraron fórmulas para actualizar las reglas.")
                return

            # Extraer reglas a partir de las fórmulas
            logging.info("🔍 [Ajuste Fino] Extrayendo reglas de las fórmulas obtenidas...")
            extracted_rules = self.extract_rules_from_formulas(formulas)
            if not extracted_rules:
                logging.warning("⚠️ [Ajuste Fino] No se pudieron extraer reglas de las fórmulas.")
                return

            # Procesar los nuevos mensajes para ajustar el modelo
            logging.info("🔧 [Ajuste Fino] Ajustando el modelo basado en las nuevas reglas...")
            X_train = [self.generate_features(rule['input_number']) for rule in extracted_rules]
            X_train = pad_sequences(X_train, maxlen=self.max_sequence_length)
            y_train = self.mlb.fit_transform([rule['recommended_numbers'] for rule in extracted_rules])

            # Realizar el ajuste fino del modelo
            if X_train.shape[0] > 0 and y_train.shape[0] > 0:
                logging.info("🛠️ [Ajuste Fino] Realizando ajuste fino del modelo...")
                self.model.fit(X_train, y_train, epochs=2, batch_size=10)

                # Actualizar el archivo con el nuevo 'last_processed_id'
                nuevo_id = max([mensaje['id'] for mensaje in nuevos_mensajes])
                self.actualizar_estado_procesamiento(nuevo_id)
                logging.info(f"✅ [Ajuste Fino] Estado actualizado. Último ID procesado: {nuevo_id}")

                logging.info("✔️ [Ajuste Fino] Ajuste fino del modelo completado exitosamente.")
            else:
                logging.warning("⚠️ [Ajuste Fino] No se encontraron suficientes datos para realizar el ajuste fino.")

        except Exception as e:
            logging.error(f"❌ [Ajuste Fino] Error durante el ajuste fino: {e}")



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

        # Crear DataFrame con los datos extraídos
        self.data = pd.DataFrame(data, columns=['input_number', 'recommended_numbers'])

        # Preprocesar los datos
        self.X = self.data['input_number'].values.reshape(-1, 1)
        self.y = self.data['recommended_numbers'].apply(lambda x: [int(num) for num in x])

        # Preparar los números atrasados más significativos
        # Asumimos que self.delayed_numbers se ha llenado en extract_rules_from_formulas
        self.most_delayed_numbers = {}
        for category in self.delayed_numbers:
            if self.delayed_numbers[category]:
                # Obtener el número con más días de atraso en la categoría
                max_delay = max(self.delayed_numbers[category], key=lambda x: x['days'])
                self.most_delayed_numbers[category] = max_delay

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

            # Obtener todas las fórmulas de la base de datos
            logging.info("[NumerologyModel] Obteniendo todas las fórmulas desde la base de datos...")
            formulas = self.db.get_all_formulas()
            if not formulas:
                logging.error("[NumerologyModel] No se encontraron fórmulas para el entrenamiento.")
                self.is_trained = False
                return

            # Extraer reglas a partir de las fórmulas
            logging.info("[NumerologyModel] Extrayendo reglas a partir de las fórmulas...")
            extracted_rules = self.extract_rules_from_formulas(formulas)
            if not extracted_rules:
                logging.error("[NumerologyModel] No se pudieron extraer reglas de las fórmulas.")
                self.is_trained = False
                return

            # Preparar los datos
            logging.info("[NumerologyModel] Preparando los datos para el entrenamiento...")

            # Generar las características usando la función generate_features
            X_features = [self.generate_features(rule['input_number']) for rule in extracted_rules]

            # Verificar que X_features no esté vacío antes de calcular max_sequence_length
            if not X_features or len(X_features) == 0:
                logging.error("[NumerologyModel] X_features está vacío o no tiene datos válidos.")
                self.is_trained = False
                return

            # Guardar la longitud máxima de secuencia
            self.max_sequence_length = max(len(seq) for seq in X_features)
            logging.info(f"[NumerologyModel] Longitud máxima de secuencia establecida en: {self.max_sequence_length}")

            # Convertir a matriz numpy con padding
            X_train = pad_sequences(X_features, padding='post', dtype='int32', maxlen=self.max_sequence_length)
            logging.info(f"[NumerologyModel] Forma final de X_train después del padding: {X_train.shape}")

            # Preprocesar las etiquetas con MultiLabelBinarizer
            self.mlb = MultiLabelBinarizer()
            y_binarized = self.mlb.fit_transform([rule['recommended_numbers'] for rule in extracted_rules])
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

            # Arquitectura del modelo (simplificada)
            self.model = Sequential()

            # Capa Embedding
            self.model.add(Embedding(input_dim=np.max(X_train) + 1, output_dim=128, input_length=self.max_sequence_length))

            # Capas LSTM Bidireccionales
            self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
            self.model.add(Dropout(0.3))
            self.model.add(Bidirectional(LSTM(64)))
            self.model.add(Dropout(0.3))

            # Capa Densa
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

        # Limitar la cantidad de números recomendados a un máximo de 10
        max_recommended_numbers = 10
        if len(recommended_numbers) > max_recommended_numbers:
            recommended_numbers = recommended_numbers[:max_recommended_numbers]  # Limitar a los primeros 10 números

        current_date = datetime.now()
        day_of_week = current_date.strftime("%A")  # Día de la semana en inglés
        day_of_week_es = self.get_day_in_spanish(day_of_week)  # Traducir a español
        current_time = current_date.strftime("%d/%m/%Y %H:%M:%S")  # Fecha y hora

        # Obtener las vibraciones y datos asociados al día
        day_vibrations_data = self.vibrations_by_day.get(day_of_week_es, {})
        day_digits = day_vibrations_data.get('digits', [])
        day_numbers = day_vibrations_data.get('numbers', [])
        day_parejas = day_vibrations_data.get('parejas', [])

        # Obtener la raíz del número proporcionado si se encuentra en las fórmulas, limitando a 10 números
        root_numbers = self.root_numbers.get(input_number, [])
        root_numbers = root_numbers[:10]  # Limitar a un máximo de 10 números

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

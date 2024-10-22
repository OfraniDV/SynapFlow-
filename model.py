#model.py

import requests
import json
import os
import pickle
import time
import pandas as pd
import numpy as np
import logging

import spacy
import re

# Cargar el modelo de lenguaje de spaCy (asegúrate de tener instalado el modelo adecuado)
nlp = spacy.load("es_core_news_md")  # Modelo en español, ajusta según el idioma

import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense, Embedding, Bidirectional
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from difflib import SequenceMatcher


from dotenv import load_dotenv

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
            # Cargar el modelo
            self.model = tf.keras.models.load_model(model_file)

            # Cargar el MultiLabelBinarizer
            mlb_path = 'mlb.pkl'
            if os.path.exists(mlb_path):
                with open(mlb_path, 'rb') as mlb_file:
                    self.mlb = pickle.load(mlb_file)
                logging.info("MultiLabelBinarizer cargado exitosamente.")
            else:
                logging.error(f"Archivo {mlb_path} no encontrado.")
                self.is_trained = False
                return

            # Cargar la longitud máxima de secuencias
            seq_length_path = 'max_sequence_length.pkl'
            if os.path.exists(seq_length_path):
                with open(seq_length_path, 'rb') as seq_file:
                    self.max_sequence_length = pickle.load(seq_file)
                logging.info("Longitud máxima de secuencias cargada exitosamente.")
            else:
                logging.error(f"Archivo {seq_length_path} no encontrado.")
                self.is_trained = False
                return

            # Extraer reglas de las fórmulas inmediatamente después de cargar el modelo
            formulas = self.db.get_all_formulas()
            if formulas:
                logging.info("Extrayendo reglas de las fórmulas después de cargar el modelo...")
                self.extract_rules_from_formulas(formulas)
            else:
                logging.error("No se encontraron fórmulas al cargar el modelo.")
                return

            # Marcar el modelo como cargado
            self.is_trained = True
            logging.info("Modelo de numerología cargado exitosamente.")

            # Aplicar ajustes finos si existen suficientes datos
            ajuste_fino_path = 'ajuste_fino_datos.pkl'
            if os.path.exists(ajuste_fino_path):
                logging.info("Aplicando ajustes finos guardados.")
                self.aplicar_ajustes_finos(ajuste_fino_path)

        except Exception as e:
            logging.error(f"Error al cargar el modelo de numerología: {e}")
            self.is_trained = False

    def aplicar_ajustes_finos(self, ajuste_fino_path):
        """Aplica los ajustes finos guardados al modelo basado en las fórmulas extraídas."""
        try:
            # Cargar los datos para el ajuste fino
            datos_para_ajuste = []
            with open(ajuste_fino_path, 'rb') as f:
                while True:
                    try:
                        datos_para_ajuste.append(pickle.load(f))
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
        try:
            # Preparar los datos
            logging.info("Iniciando la preparación de los datos...")
            self.prepare_data()

            # Verificar que los datos de entrada y las etiquetas existen
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                logging.error("Los datos de entrenamiento no están disponibles. Asegúrate de que los datos fueron cargados correctamente.")
                self.is_trained = False
                return

            if self.X.size == 0 or len(self.y) == 0:
                logging.error("No hay datos suficientes para entrenar el modelo. X o y están vacíos.")
                self.is_trained = False
                return

            # Incorporar vibraciones y otros datos en las características usando la nueva función generate_features
            logging.info("Incorporando vibraciones y otros datos en las características...")
            X_features = [self.generate_features(int(input_number)) for input_number in self.X.flatten()]

            # Verificar que X_features no esté vacío antes de calcular max_sequence_length
            if not X_features or len(X_features) == 0:
                logging.error("X_features está vacío o no tiene datos válidos.")
                self.is_trained = False
                return

            # Guardar la longitud máxima de secuencia
            self.max_sequence_length = max(len(seq) for seq in X_features)
            logging.info(f"Longitud máxima de secuencia establecida en: {self.max_sequence_length}")

            # Verificar que max_sequence_length sea válida
            if not isinstance(self.max_sequence_length, int) or self.max_sequence_length <= 0:
                logging.error(f"max_sequence_length no es válido: {self.max_sequence_length}")
                self.is_trained = False
                return

            # Convertir a matriz numpy con padding
            X_train = pad_sequences(X_features, padding='post', dtype='int32', maxlen=self.max_sequence_length)
            logging.info(f"Forma final de X_train después del padding: {X_train.shape}")

            # Preprocesar las etiquetas con MultiLabelBinarizer
            self.mlb = MultiLabelBinarizer()
            y_binarized = self.mlb.fit_transform(self.y)
            logging.info(f"Forma de y_binarized después de aplicar MultiLabelBinarizer: {y_binarized.shape}")

            # Verificar si hay valores NaN o Inf
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                logging.error("X_train contiene valores NaN o Inf. No se puede continuar con el entrenamiento.")
                self.is_trained = False
                return

            if np.isnan(y_binarized).any() or np.isinf(y_binarized).any():
                logging.error("y_binarized contiene valores NaN o Inf. No se puede continuar con el entrenamiento.")
                self.is_trained = False
                return

            num_classes = y_binarized.shape[1]  # Número de clases únicas en las etiquetas
            logging.info(f"Número de clases en y_binarized: {num_classes}")

            # Definir el modelo de red neuronal para secuencias
            max_input_value = np.max(X_train) + 1  # Valor máximo de entrada para la capa Embedding
            logging.info(f"El valor máximo de entrada para la capa Embedding será: {max_input_value}")

            # Definir la arquitectura del modelo
            self.model = Sequential()
            self.model.add(Embedding(input_dim=max_input_value, output_dim=64))
            self.model.add(LSTM(64))
            self.model.add(Dense(num_classes, activation='sigmoid'))

            # Compilar el modelo
            logging.info("Compilando el modelo...")
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Entrenar el modelo
            logging.info("Entrenando el modelo...")
            self.model.fit(X_train, y_binarized, epochs=10, batch_size=10, verbose=1)

            # Indicar que el modelo fue entrenado correctamente
            self.is_trained = True
            logging.info("Modelo entrenado exitosamente con red neuronal.")

        except Exception as e:
            logging.error(f"Error durante el entrenamiento del modelo: {e}")
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
            # Aquí deberías realizar la consulta a la base de datos
            # para obtener las vibraciones del día en cuestión.
            vibrations = self.db.get_vibrations_by_day(day_of_week_es)
            if not vibrations:
                #logging.warning(f"No se encontraron vibraciones para el día {day_of_week_es}.")
                return []
            #logging.info(f"Vibraciones encontradas para {day_of_week_es}: {vibrations}")
            return vibrations
        except Exception as e:
            #logging.error(f"Error al obtener las vibraciones para el día {day_of_week_es}: {e}")
            return []

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
        
        # Variable que indica si el modelo está entrenado
        self.is_trained = False
        
        # Longitud máxima de las secuencias de entrada
        self.max_sequence_length = 100
        
        # Límite de vocabulario a las 10,000 palabras más frecuentes
        self.num_words = 10000
        
        # Conjunto para almacenar los mensajes únicos procesados
        self.processed_messages = set()

        # Cargar el modelo y el tokenizer si están disponibles
        self.cargar_modelo_y_tokenizer()

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
            logging.info("Intentando cargar el modelo conversacional desde 'conversational_model.keras'...")
            self.model = tf.keras.models.load_model('conversational_model.keras')
            logging.info(f"Modelo conversacional cargado exitosamente en {time.time() - start_time:.2f} segundos.")
            
            # Cargar el tokenizer desde un archivo
            logging.info("Intentando cargar el tokenizer desde 'tokenizer.pkl'...")
            with open('tokenizer.pkl', 'rb') as f:
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
        """Carga el modelo conversacional y el tokenizer."""
        try:
            start_time = time.time()
            
            # Intentar cargar el modelo de la red neuronal desde el archivo
            logging.info("Intentando cargar el modelo conversacional desde 'conversational_model.keras'...")
            self.model = tf.keras.models.load_model('conversational_model.keras')
            logging.info(f"Modelo conversacional cargado exitosamente. Tiempo de carga: {time.time() - start_time:.2f} segundos.")
            print(f"[INFO] Modelo conversacional cargado exitosamente en {time.time() - start_time:.2f} segundos.")

            # Intentar cargar el tokenizer desde el archivo
            logging.info("Intentando cargar el tokenizer desde 'tokenizer.pkl'...")
            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            if self.tokenizer:
                logging.info("Tokenizer cargado exitosamente.")
                print("[INFO] Tokenizer cargado exitosamente.")
            else:
                logging.warning("El tokenizer no se cargó correctamente.")
                print("[WARNING] El tokenizer no se cargó correctamente.")
            
            # Indicar que el modelo ha sido cargado y está listo para usarse
            self.is_trained = True
            logging.info("El modelo y el tokenizer están listos para usarse.")
            print("[INFO] El modelo y el tokenizer están listos para usarse.")

        except Exception as e:
            # Registrar el error en los logs y mostrarlo en consola
            logging.error(f"Error al cargar el modelo o el tokenizer: {e}")
            print(f"[ERROR] Error al cargar el modelo o el tokenizer: {e}")
            self.is_trained = False

    def analyze_message(self, input_text):
        """
        Realiza un análisis previo del mensaje antes de generar una respuesta.

        Args:
            input_text (str): El mensaje del usuario que será analizado.

        Returns:
            str or None: Devuelve una respuesta si se detecta una condición especial,
                        o None si no se detectan condiciones especiales.
        """
        logger.info(f"Analizando el mensaje: {input_text}")

        # Convertir el texto a minúsculas una vez para mejorar la eficiencia
        input_text_lower = input_text.lower()

        # Definir palabras clave para detectar condiciones especiales
        help_keywords = ['ayuda', 'soporte', 'problema', 'error']
        offensive_keywords = ['maldición', 'insulto', 'grosería']  # Puedes expandir esta lista según sea necesario

        # Análisis 1: Detectar si el mensaje contiene palabras clave relacionadas con ayuda
        if any(keyword in input_text_lower for keyword in help_keywords):
            logger.info("El mensaje contiene palabras clave relacionadas con ayuda.")
            return "Parece que necesitas ayuda. ¿En qué puedo asistirte?"

        # Análisis 2: Detectar si el mensaje es muy corto (menos de 3 palabras)
        word_count = len(input_text.split())
        if word_count < 3:
            logger.info(f"Mensaje demasiado corto (solo {word_count} palabras), generando respuesta simple.")
            return None  # Puedes ajustar la respuesta aquí para simplificar aún más la interacción

        # Análisis 3: Detectar si el mensaje es muy largo (más de 50 palabras)
        if word_count > 50:
            logger.info(f"Mensaje demasiado largo ({word_count} palabras), solicitando un resumen.")
            return "Tu mensaje es un poco largo. ¿Podrías resumirlo para que pueda entender mejor?"

        # Análisis 4: Detectar lenguaje inapropiado (expandir según sea necesario)
        if any(offensive_word in input_text_lower for offensive_word in offensive_keywords):
            logger.warning("Se detectó lenguaje inapropiado en el mensaje.")
            return "Por favor, evita usar lenguaje inapropiado."

        # (Opcional) Análisis 5: Otros análisis (añadir más reglas personalizadas según el caso)
        # Ejemplo: Detectar preguntas o comandos específicos
        if input_text_lower.startswith('comando'):
            logger.info("Se detectó un comando especial.")
            return "Has activado un comando especial. Procesando..."

        # Si no se detectan condiciones especiales, continuar con la generación de respuesta
        logger.info("No se detectaron condiciones especiales en el mensaje.")
        return None

    def generate_response(self, input_text, temperature=0.7, max_words=20):
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
            local_response = self.model_generate_response(input_text, temperature, max_words)
        except Exception as e:
            logger.error(f"Error al generar respuesta con el modelo local: {e}")
            return "Se ha producido un error al generar la respuesta local. Por favor, intenta más tarde."

        # Post-procesar la respuesta para eliminar '<OOV>' y mejorar la coherencia
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

    def comparar_respuestas(self, local_response, gpt_response):
        """Compara la respuesta generada localmente con la de GPT-4 para decidir cuál es mejor."""
        
        # 1. Verificar si alguna de las respuestas es demasiado corta o no tiene sentido
        if len(local_response.split()) < 3 or local_response == "<UNK>":
            logger.info("La respuesta local es demasiado corta o incoherente. Se seleccionará la respuesta de GPT-4.")
            return True  # GPT-4 es mejor en este caso

        # 2. Comparar similitud de las respuestas (usando una métrica básica de similitud)
        similarity = SequenceMatcher(None, local_response, gpt_response).ratio()
        logger.info(f"Similitud entre la respuesta local y la de GPT-4: {similarity:.2f}")

        # Si la similitud es muy alta (por ejemplo, más del 90%), se consideran casi iguales
        if similarity > 0.9:
            logger.info("Las respuestas local y GPT-4 son similares.")
            return False  # No es necesario preferir una sobre la otra

        # 3. Verificar si alguna de las respuestas es significativamente más larga (lo que podría significar más información)
        if len(gpt_response.split()) > len(local_response.split()) + 5:
            logger.info("La respuesta de GPT-4 es significativamente más larga, seleccionando GPT-4.")
            return True  # GPT-4 es mejor si es mucho más detallada
        
        # 4. Comparar otras condiciones que puedas definir (puedes añadir reglas adicionales aquí)

        # Si ninguna de las condiciones previas aplica, se considera que la respuesta local es suficiente
        logger.info("Se seleccionará la respuesta local.")
        return False

    def almacenar_para_ajuste_fino(self, input_text, output_text):
        """Almacena las entradas y las salidas de GPT-4 para realizar un ajuste fino."""
        try:
            # Validar que el texto de entrada y la respuesta no sean vacíos o incoherentes
            if not input_text or not output_text:
                logging.warning("El texto de entrada o la respuesta están vacíos. No se almacenarán.")
                return

            # Verificar si la respuesta es suficientemente larga para el ajuste fino
            if len(output_text.split()) < 3:
                logging.warning(f"Respuesta demasiado corta, no se almacenará para ajuste fino: {output_text}")
                return
            
            # Crear los datos que serán almacenados
            data = {"input": input_text, "response": output_text}

            # Verificar si el archivo ya existe para evitar duplicados
            ajuste_fino_file = 'ajuste_fino_datos.pkl'
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

    def primer_ajuste_fino(self, datos_entrenamiento=None, epochs=5):
        """
        Realiza el ajuste fino inicial del modelo conversacional utilizando los datos almacenados
        o datos proporcionados manualmente.
        
        Args:
            datos_entrenamiento (list, optional): Lista de datos de entrenamiento. Si no se proporcionan,
                                                se usarán los datos guardados previamente.
            epochs (int): Número de épocas para entrenar el modelo.
        """
        logger.info("Iniciando el primer ajuste fino del modelo conversacional.")

        # Si no se proporcionan datos de entrenamiento, cargarlos desde los archivos existentes
        if datos_entrenamiento is None:
            logger.info("No se proporcionaron datos de entrenamiento. Cargando desde archivos.")
            datos_entrenamiento = []
            try:
                for file in os.listdir('.'):
                    if file.startswith('ajuste_fino_datos') and file.endswith('.pkl'):
                        with open(file, 'rb') as f:
                            while True:
                                try:
                                    data = pickle.load(f)
                                    if isinstance(data, dict) and 'input' in data and 'response' in data:
                                        datos_entrenamiento.append(data)
                                    else:
                                        logger.error(f"Formato de datos incorrecto o faltan claves en {file}: {data}")
                                except EOFError:
                                    break
                                except pickle.UnpicklingError as e:
                                    logger.error(f"Error al deserializar datos del archivo {file}: {e}")
                                    break
            except Exception as e:
                logger.error(f"Error al cargar los datos almacenados para ajuste fino: {e}")

        # Si no se encontraron datos, abortar el proceso
        if not datos_entrenamiento:
            logger.warning("No se encontraron datos suficientes para realizar el ajuste fino.")
            return

        # Preprocesar los datos y tokenizarlos
        inputs = [data["input"] for data in datos_entrenamiento if isinstance(data["input"], str)]
        respuestas = [data["response"] for data in datos_entrenamiento if isinstance(data["response"], str)]

        if len(inputs) == 0 or len(respuestas) == 0:
            logger.error("No se encontraron entradas o respuestas válidas para el ajuste fino.")
            return

        # Tokenización y padding de secuencias
        sequences = self.tokenizer.texts_to_sequences(inputs)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)

        response_sequences = self.tokenizer.texts_to_sequences(respuestas)
        y = pad_sequences(response_sequences, maxlen=self.max_sequence_length)

        if len(X) == 0 or len(y) == 0:
            logger.error("No se generaron secuencias o etiquetas válidas.")
            return

        # Ajuste fino del modelo
        try:
            logger.info(f"Entrenando el modelo local con {len(X)} ejemplos nuevos para ajuste fino.")
            self.model.fit(X, y, epochs=epochs, batch_size=32)
            logger.info("Ajuste fino del modelo local completado con éxito.")
        except Exception as e:
            logger.error(f"Error durante el ajuste fino del modelo: {e}")


    def realizar_ajuste_fino(self, nuevos_datos=None, epochs=2):
        """
        Realiza un ajuste fino en el modelo local utilizando datos proporcionados directamente
        o los datos almacenados previamente.
        
        Args:
            nuevos_datos (list, optional): Lista de nuevos datos para ajuste fino. Si no se proporcionan,
                                        se usarán los datos almacenados en archivos.
            epochs (int): Número de épocas para entrenar el modelo.
        """
        logger.info("Iniciando el ajuste fino del modelo conversacional.")
        
        datos_para_ajuste = []

        try:
            if nuevos_datos:
                # Si se proporcionan nuevos datos, limpiarlos y agregarlos directamente
                logger.info("Usando nuevos datos para el ajuste fino.")
                nuevos_datos_limpios = self.limpiar_datos(nuevos_datos)
                datos_para_ajuste.extend(nuevos_datos_limpios)
            else:
                # Cargar los archivos que contienen los datos de ajuste fino si no se proporcionaron nuevos datos
                logger.info("Cargando datos almacenados para el ajuste fino.")
                for file in os.listdir('.'):
                    if file.startswith('ajuste_fino_datos') and file.endswith('.pkl'):
                        with open(file, 'rb') as f:
                            while True:
                                try:
                                    data = pickle.load(f)
                                    if isinstance(data, dict) and 'input' in data and 'response' in data:
                                        datos_para_ajuste.append(data)
                                    else:
                                        logger.error(f"Formato de datos incorrecto o faltan claves en {file}: {data}")
                                except EOFError:
                                    break
                                except pickle.UnpicklingError as e:
                                    logger.error(f"Error al deserializar datos del archivo {file}: {e}")
                                    break

            if not datos_para_ajuste:
                logger.warning("No hay suficientes datos para realizar un ajuste fino.")
                return
            
            # Tokenizar las entradas y generar secuencias
            inputs = [data["input"] for data in datos_para_ajuste if isinstance(data["input"], str)]
            respuestas = [data["response"] for data in datos_para_ajuste if isinstance(data["response"], str)]

            if len(inputs) == 0 or len(respuestas) == 0:
                logger.error("No se encontraron entradas o respuestas válidas.")
                return

            # Tokenización y padding de secuencias
            sequences = self.tokenizer.texts_to_sequences(inputs)
            X = pad_sequences(sequences, maxlen=self.max_sequence_length)

            response_sequences = self.tokenizer.texts_to_sequences(respuestas)
            y = pad_sequences(response_sequences, maxlen=self.max_sequence_length)

            if len(X) == 0 or len(y) == 0:
                logger.error("No se generaron secuencias o etiquetas válidas.")
                return

            # Ajuste fino del modelo
            logger.info(f"Entrenando el modelo local con {len(X)} ejemplos nuevos para ajuste fino.")
            self.model.fit(X, y, epochs=epochs, batch_size=32)
            logger.info("Ajuste fino del modelo local completado con éxito.")

        except Exception as e:
            logger.error(f"Error durante el ajuste fino Conversar Model: {e}")


    def model_generate_response(self, input_text, temperature=1.0, max_words=20):
        """Genera una respuesta utilizando el modelo local."""
        
        logger.info(f"Generando respuesta local para el mensaje: {input_text}")

        # Verificar si el tokenizer está correctamente inicializado
        if self.tokenizer is None:
            logger.error("El tokenizer no está inicializado.")
            return "El modelo no está listo para procesar este mensaje."

        # Preprocesar el texto de entrada para convertirlo a secuencias de índices
        input_sequence = self.tokenizer.texts_to_sequences([input_text])
        if not input_sequence or len(input_sequence[0]) == 0:
            logger.warning("No se pudo procesar el mensaje de entrada, probablemente no se entiende el mensaje.")
            return "Lo siento, no entiendo lo que quieres decir."

        # Aplicar padding para asegurar que la longitud de la secuencia sea consistente con lo que el modelo espera
        input_sequence = pad_sequences(input_sequence, maxlen=self.max_sequence_length)
        logger.debug(f"Secuencia de entrada después del padding: {input_sequence}")

        generated_response = []  # Lista para almacenar las palabras generadas

        # Comenzar a generar una secuencia de palabras para la respuesta
        for _ in range(max_words):
            # Realizar la predicción del siguiente token/palabra
            try:
                predicted_probs = self.model.predict(input_sequence)
            except Exception as e:
                logger.error(f"Error durante la predicción del modelo: {e}")
                return "Lo siento, ha ocurrido un error al generar la respuesta."

            logger.debug(f"Probabilidades predichas para la próxima palabra: {predicted_probs}")

            # Aplicar control de temperatura para ajustar la aleatoriedad en la generación de palabras
            predicted_probs = np.asarray(predicted_probs).astype('float64')
            predicted_probs = np.log(predicted_probs + 1e-8) / temperature
            exp_preds = np.exp(predicted_probs)
            predicted_probs = exp_preds / np.sum(exp_preds)  # Normalizar las probabilidades
            logger.debug(f"Probabilidades ajustadas después de aplicar temperatura: {predicted_probs}")

            # Seleccionar el índice de la palabra predicha
            predicted_word_index = np.random.choice(range(self.num_words), p=predicted_probs.ravel())
            predicted_word = self.tokenizer.index_word.get(predicted_word_index, '<OOV>')  # Recuperar la palabra correspondiente al índice

            # Detener la generación si se encuentra una palabra desconocida o inválida
            if predicted_word == '<OOV>' or predicted_word == '':
                logger.warning("Se predijo una palabra desconocida o inválida, deteniendo la generación de respuesta.")
                break

            # Agregar la palabra predicha a la respuesta generada
            generated_response.append(predicted_word)
            logger.debug(f"Palabra generada: {predicted_word}")

            # Actualizar la secuencia de entrada con la palabra recién generada
            input_sequence = pad_sequences([input_sequence[0].tolist() + [predicted_word_index]], maxlen=self.max_sequence_length)

        # Combinar las palabras generadas en una oración final
        response = ' '.join(generated_response)
        logger.info(f"Respuesta local generada: {response}")

        # Si la respuesta es muy corta o carece de coherencia, proporcionar una respuesta por defecto
        if len(response.split()) < 3:
            logger.warning("La respuesta generada es demasiado corta o carece de coherencia.")
            return "Lo siento, no estoy seguro de cómo responder a eso. ¿Podrías reformular tu pregunta?"

        return response
    
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

    def build_model(self, output_activation='sigmoid', learning_rate=0.0001):
        """
        Construir e inicializar el modelo secuencial.
        
        Args:
            output_activation (str): Tipo de activación para la capa de salida ('sigmoid' o 'softmax').
            learning_rate (float): Tasa de aprendizaje para el optimizador Adam.
        """
        logging.info("Inicializando el modelo...")

        try:
            # Verificar que los parámetros clave están inicializados correctamente
            if not self.num_words or not self.max_sequence_length:
                raise ValueError("Los parámetros 'num_words' o 'max_sequence_length' no están definidos correctamente.")
            
            self.model = Sequential()
            
            # Capa Embedding
            logging.info(f"Agregando capa de Embedding con input_dim={self.num_words}, output_dim=128, input_length={self.max_sequence_length}")
            self.model.add(Embedding(input_dim=self.num_words, output_dim=128, input_length=self.max_sequence_length))
            
            # Capas LSTM Bidireccional con Dropout
            logging.info("Agregando capas LSTM Bidireccional con Dropout.")
            self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
            self.model.add(Dropout(0.3))
            self.model.add(Bidirectional(LSTM(128)))
            self.model.add(Dropout(0.3))
            
            # Capa densa intermedia
            logging.info("Agregando capa densa intermedia con 128 unidades y activación ReLU.")
            self.model.add(Dense(128, activation='relu'))
            
            # Capa de salida
            logging.info(f"Agregando capa de salida con activación {output_activation}.")
            if output_activation == 'softmax':
                self.model.add(Dense(self.num_words, activation='softmax'))
            else:
                self.model.add(Dense(self.num_words, activation='sigmoid'))  # Mantiene sigmoid como predeterminado
            
            # Compilar el modelo
            optimizer = Adam(learning_rate=learning_rate)
            logging.info(f"Compilando el modelo con Adam optimizer (learning_rate={learning_rate}) y sparse_categorical_crossentropy.")
            self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            logging.info("Modelo inicializado y compilado correctamente.")
        except ValueError as ve:
            logging.error(f"Error de valor en los parámetros del modelo: {ve}")
            raise ve
        except Exception as e:
            logging.error(f"Error inesperado al construir el modelo: {e}")
            raise e

    def prepare_data(self):
        """Prepara los datos para el modelo de conversación"""
        logging.info("Iniciando la preparación de datos para el modelo de conversación...")

        try:
            # Obtener todos los mensajes de la base de datos
            all_messages = self.db.get_all_messages()
            if not all_messages or len(all_messages) == 0:
                logging.error("No se encontraron mensajes en la base de datos.")
                return None

            logging.info(f"Se encontraron {len(all_messages)} mensajes en la base de datos.")

            # Limpiar y procesar los mensajes, evitando duplicados
            messages = []
            for message_text in all_messages:
                cleaned_message = self.clean_text(message_text)

                # Verificar si el mensaje ya fue procesado
                if self.db.is_message_processed(cleaned_message):
                    logging.debug(f"Mensaje ya procesado, omitiendo: {cleaned_message}")
                    continue  # Ignorar mensajes ya procesados
                
                self.processed_messages.add(cleaned_message)
                messages.append(cleaned_message)

            if not messages:
                logging.warning("No se encontraron nuevos mensajes únicos para procesar.")
                return None  # Asegurarse de que se retorna None si no hay mensajes limpios

            logging.info(f"Se encontraron {len(messages)} nuevos mensajes únicos después del procesamiento.")

            # Tokenizar los mensajes
            sequences = self.tokenize_messages(messages)
            if not sequences or len(sequences) == 0:
                logging.error("La tokenización no generó secuencias válidas.")
                return None
            logging.info(f"Se generaron {len(sequences)} secuencias después de la tokenización.")

            # Generar las secuencias de entrada y etiquetas
            self.X, self.y = self.generate_sequences_and_labels(sequences)

            # Verificar que las secuencias y etiquetas se generaron correctamente
            if self.X is None or len(self.X) == 0 or self.y is None or len(self.y) == 0:
                logging.error("La preparación de los datos no produjo secuencias o etiquetas válidas.")
                return None

            logging.info(f"Datos preparados correctamente: X tiene forma {self.X.shape}, y tiene forma {self.y.shape}")
            return True

        except Exception as e:
            logging.error(f"Error durante la preparación de los datos: {e}")
            return None


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
        """Tokeniza los mensajes y devuelve secuencias y el tokenizer."""
        try:
            if not messages or len(messages) == 0:
                logging.error("No se proporcionaron mensajes para tokenizar.")
                return None, None
            
            # Inicializar el tokenizer con el número máximo de palabras y token OOV
            self.tokenizer = Tokenizer(num_words=self.num_words, oov_token="<OOV>")

            # Filtrar mensajes vacíos o nulos antes de tokenizar
            valid_messages = [msg for msg in messages if msg.strip()]
            if len(valid_messages) != len(messages):
                logging.warning(f"Se han encontrado {len(messages) - len(valid_messages)} mensajes vacíos o inválidos que se omitieron.")

            # Ajustar el tokenizer con los mensajes
            self.tokenizer.fit_on_texts(valid_messages)
            
            # Tokenizar los mensajes
            sequences = self.tokenizer.texts_to_sequences(valid_messages)
            
            logging.info(f"Se tokenizaron {len(valid_messages)} mensajes, generando {len(sequences)} secuencias.")
            return sequences, self.tokenizer

        except Exception as e:
            logging.error(f"Error al tokenizar los mensajes: {e}")
            return None, None

    def generate_sequences_and_labels(self, sequences):
        """Genera secuencias de entrada y etiquetas a partir de los mensajes tokenizados."""
        X, y = [], []

        # Verificar si num_classes está correctamente inicializado
        num_classes = self.tokenizer.num_words
        if num_classes is None or num_classes <= 0:
            logging.error("El número de clases (num_words en el tokenizer) no está definido correctamente.")
            return None, None
        
        logging.info(f"Comenzando a generar secuencias y etiquetas. Num clases: {num_classes}")

        try:
            # Recorrer las secuencias tokenizadas
            for seq_index, seq in enumerate(sequences):
                if len(seq) == 0:
                    logging.warning(f"Secuencia vacía encontrada en el índice {seq_index}, omitiendo.")
                    continue  # Omitir secuencias vacías

                logging.debug(f"Procesando secuencia {seq_index + 1}/{len(sequences)}. Longitud de la secuencia: {len(seq)}")

                for i in range(1, len(seq)):
                    input_sequence = seq[:i]
                    target_word = seq[i]

                    # Padding para asegurar que todas las secuencias tengan la misma longitud
                    input_sequence_padded = pad_sequences([input_sequence], maxlen=self.max_sequence_length)[0]

                    # Agregar la secuencia y la etiqueta
                    X.append(input_sequence_padded)

                    # Convertir la etiqueta (target_word) a one-hot encoding, asegurando que no exceda el número de clases
                    if target_word < num_classes:
                        one_hot_target = to_categorical(target_word, num_classes=num_classes)
                        y.append(one_hot_target)
                    else:
                        logging.warning(f"Palabra objetivo {target_word} fuera de los límites en la secuencia {seq_index}. Omitiendo.")

                    logging.debug(f"Secuencia de entrada (padded): {input_sequence_padded}")
                    logging.debug(f"Palabra objetivo: {target_word}, One-hot encoded: {one_hot_target}")

            # Convertir las listas a matrices numpy
            X_np = np.array(X)
            y_np = np.array(y)

            # Verificar si se generaron secuencias y etiquetas válidas
            if X_np.shape[0] == 0 or y_np.shape[0] == 0:
                logging.error("No se generaron secuencias o etiquetas válidas.")
                return None, None

            logging.info(f"Generación de secuencias completada. Shape de X: {X_np.shape}, Shape de y: {y_np.shape}")
            return X_np, y_np

        except Exception as e:
            logging.error(f"Error durante la generación de secuencias y etiquetas: {e}")
            return None, None

    def train(self, epochs=10, batch_size=32, validation_split=0.2, use_callbacks=False):
        """Entrena el modelo conversacional"""
        try:
            # Preparar los datos
            logging.info("Iniciando la preparación de los datos para el entrenamiento...")
            self.prepare_data()

            # Validar si los datos fueron cargados correctamente
            if self.X is None or len(self.X) == 0 or self.y is None or len(self.y) == 0:
                logging.error("No se pudieron preparar los datos de entrenamiento. X o y son inválidos.")
                self.is_trained = False
                return

            logging.info(f"Datos preparados: X tiene forma {self.X.shape}, y tiene forma {self.y.shape}")

            # Asegurarse de que el modelo está inicializado
            if self.model is None:
                logging.info("El modelo no está definido. Inicializando el modelo...")
                self.build_model()

            # Callbacks opcionales para mejorar el entrenamiento
            callbacks = []
            if use_callbacks:
                from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
                ]
                logging.info("Callbacks para EarlyStopping y ModelCheckpoint agregados.")

            # Iniciar el entrenamiento del modelo
            logging.info(f"Iniciando el entrenamiento del modelo con {epochs} épocas y batch_size de {batch_size}.")
            history = self.model.fit(
                self.X, self.y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,  # División para validación
                callbacks=callbacks if use_callbacks else None
            )

            logging.info("Entrenamiento completado con éxito.")

            # Marcar el modelo como entrenado
            self.is_trained = True
            logging.info("El modelo ha sido marcado como entrenado.")

        except Exception as e:
            logging.error(f"Error durante el entrenamiento del modelo: {e}")
            self.is_trained = False
     
    def ajustar_temperatura(self, input_text, short_temp=0.7, long_temp=1.5, standard_temp=1.0, short_limit=3, long_limit=10):
        """
        Ajusta la temperatura en función de la longitud del input_text.

        Args:
            input_text (str): Texto de entrada proporcionado por el usuario.
            short_temp (float): Temperatura para textos cortos.
            long_temp (float): Temperatura para textos largos.
            standard_temp (float): Temperatura estándar para textos de longitud media.
            short_limit (int): Límite de palabras para considerar un texto como corto.
            long_limit (int): Límite de palabras para considerar un texto como largo.

        Returns:
            float: Temperatura ajustada en función de la longitud del texto.
        """
        if not input_text or len(input_text.strip()) == 0:
            logging.warning("El texto de entrada está vacío o solo contiene espacios.")
            return standard_temp  # Valor estándar si el texto está vacío

        word_count = len(input_text.split())

        if word_count <= short_limit:
            logging.info(f"Texto corto detectado ({word_count} palabras). Usando temperatura {short_temp}.")
            return short_temp  # Temperatura para textos cortos
        elif word_count > long_limit:
            logging.info(f"Texto largo detectado ({word_count} palabras). Usando temperatura {long_temp}.")
            return long_temp  # Temperatura para textos largos
        else:
            logging.info(f"Texto de longitud media detectado ({word_count} palabras). Usando temperatura {standard_temp}.")
            return standard_temp  # Temperatura estándar
    
    def filtrar_predicciones(self, predicted_probs):
        """
        Aplica un filtro para penalizar palabras con muy baja o alta frecuencia.

        Args:
            predicted_probs (numpy array): Array de probabilidades predichas para cada palabra en el vocabulario.

        Returns:
            numpy array: Probabilidades ajustadas con penalizaciones y aumentos aplicados.
        """
        # Verificar que las probabilidades predichas son válidas
        if predicted_probs is None or len(predicted_probs) != self.num_words:
            logging.error("El tamaño de las probabilidades predichas no coincide con el tamaño del vocabulario.")
            return predicted_probs

        # Inicializar las listas de palabras frecuentes y raras si no están definidas
        if not hasattr(self, 'frequent_words'):
            self.frequent_words = []
            logging.warning("La lista 'frequent_words' no estaba definida. Se inicializa como lista vacía.")
            
        if not hasattr(self, 'rare_words'):
            self.rare_words = []
            logging.warning("La lista 'rare_words' no estaba definida. Se inicializa como lista vacía.")
        
        # Crear una copia de las probabilidades para aplicar los cambios
        penalized_probs = np.copy(predicted_probs)

        try:
            # Penalizar palabras frecuentes
            if len(self.frequent_words) > 0:
                penalized_probs[self.frequent_words] *= 0.5
                logging.info(f"Se penalizaron {len(self.frequent_words)} palabras frecuentes.")

            # Aumentar la probabilidad de palabras raras
            if len(self.rare_words) > 0:
                penalized_probs[self.rare_words] *= 1.5
                logging.info(f"Se aumentó la probabilidad de {len(self.rare_words)} palabras raras.")

            # Asegurarse de que las probabilidades no excedan los límites [0, 1]
            penalized_probs = np.clip(penalized_probs, 0, 1)

            # Reescalar las probabilidades para que sumen 1
            penalized_probs /= np.sum(penalized_probs)
            logging.info("Las probabilidades han sido reescaladas correctamente.")

        except Exception as e:
            logging.error(f"Error durante el filtrado de predicciones: {e}")
            return predicted_probs  # Retornar las probabilidades originales si ocurre un error

        return penalized_probs
    
    def generar_respuestas_multiples(self, input_text, n_respuestas=3):
        """
        Genera múltiples respuestas y selecciona la mejor basada en la probabilidad o métrica de coherencia.
        
        Args:
            input_text (str): El texto de entrada para generar respuestas.
            n_respuestas (int): Número de respuestas a generar.

        Returns:
            str: La mejor respuesta generada.
        """
        respuestas = []

        try:
            # Generar múltiples respuestas
            logging.info(f"Generando {n_respuestas} respuestas para el texto: {input_text}")
            for i in range(n_respuestas):
                respuesta = self.generate_response(input_text)
                
                # Verificar que la respuesta no sea vacía o inválida
                if respuesta and len(respuesta.strip()) > 0:
                    respuestas.append(respuesta)
                    logging.debug(f"Respuesta {i + 1}: {respuesta}")
                else:
                    logging.warning(f"Se generó una respuesta vacía o inválida en la iteración {i + 1}.")
            
            if not respuestas:
                logging.error("No se generaron respuestas válidas.")
                return None

            logging.info(f"Se generaron {len(respuestas)} respuestas válidas.")

            # Seleccionar la mejor respuesta, puedes añadir más criterios aquí
            mejor_respuesta = max(set(respuestas), key=respuestas.count)
            logging.info(f"La mejor respuesta seleccionada: {mejor_respuesta}")

            # Si hay empate en el conteo de respuestas, puedes aplicar otro criterio, como la longitud
            respuestas_empatadas = [resp for resp in respuestas if respuestas.count(resp) == respuestas.count(mejor_respuesta)]
            
            if len(respuestas_empatadas) > 1:
                # En caso de empate, selecciona la respuesta más larga
                mejor_respuesta = max(respuestas_empatadas, key=len)
                logging.info(f"Hubo empate. Seleccionada la respuesta más larga: {mejor_respuesta}")

            return mejor_respuesta

        except Exception as e:
            logging.error(f"Error al generar respuestas múltiples: {e}")
            return None

    def limpiar_datos(self, nuevos_datos):
        """
        Limpia los datos nuevos aplicando la función clean_text.

        Args:
            nuevos_datos (list): Lista de datos de texto que necesitan limpieza.

        Returns:
            list: Lista de datos limpios.
        """
        try:
            # Validar que los datos no sean nulos o vacíos
            if not nuevos_datos or len(nuevos_datos) == 0:
                logging.warning("Los datos proporcionados están vacíos o son nulos.")
                return []

            # Limpiar los datos utilizando la función clean_text
            datos_limpios = [self.clean_text(texto) for texto in nuevos_datos if texto.strip()]
            
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

    def generar_secuencias_y_etiquetas(self, nuevos_datos_limpios):
        """
        Genera secuencias y etiquetas a partir de los datos nuevos limpios.

        Args:
            nuevos_datos_limpios (list): Lista de datos limpios (texto) para procesar.

        Returns:
            tuple: Dos arreglos numpy (X, y) con las secuencias de entrada y las etiquetas, respectivamente.
        """
        try:
            # Verificar que los datos de entrada no estén vacíos
            if not nuevos_datos_limpios or len(nuevos_datos_limpios) == 0:
                logging.warning("Los datos limpios están vacíos o no son válidos.")
                return np.array([]), np.array([])

            # Tokenizar los datos limpios
            nuevas_secuencias = self.tokenizer.texts_to_sequences(nuevos_datos_limpios)
            
            # Verificar que se generaron secuencias válidas
            if len(nuevas_secuencias) == 0:
                logging.warning("La tokenización no generó secuencias válidas.")
                return np.array([]), np.array([])

            logging.info(f"Se generaron {len(nuevas_secuencias)} secuencias a partir de los datos limpios.")

            # Generar las secuencias de entrada y etiquetas
            nuevas_X, nuevas_y = self.generate_sequences_and_labels(nuevas_secuencias)

            # Validar que las secuencias y etiquetas son válidas
            if nuevas_X is None or nuevas_y is None or len(nuevas_X) == 0 or len(nuevas_y) == 0:
                logging.warning("Las secuencias o etiquetas generadas no son válidas.")
                return np.array([]), np.array([])

            logging.info(f"Secuencias de entrada y etiquetas generadas exitosamente: X.shape={nuevas_X.shape}, y.shape={nuevas_y.shape}.")
            return nuevas_X, nuevas_y

        except Exception as e:
            logging.error(f"Error al generar secuencias y etiquetas: {e}")
            return np.array([]), np.array([])

    def mantener_contexto(self, user_id, input_text, contexto_dict):
        """
        Mantiene el contexto de la conversación para generar respuestas más coherentes. 
        Verifica las interacciones previas del usuario.

        Args:
            user_id (str): ID del usuario que envía el mensaje.
            input_text (str): El mensaje actual del usuario.
            contexto_dict (dict): Diccionario que almacena el contexto de los usuarios. La clave es el user_id.

        Returns:
            str: Respuesta generada basada en el contexto.
        """
        # Verificar si ya existe un contexto previo para este usuario
        if user_id not in contexto_dict:
            contexto_dict[user_id] = []  # Iniciar el contexto para el usuario si no existe

        # Agregar el nuevo mensaje al contexto del usuario
        contexto_dict[user_id].append(input_text)

        # Mantener un máximo de 5 interacciones en el contexto
        if len(contexto_dict[user_id]) > 5:
            contexto_dict[user_id].pop(0)

        # Unir el contexto en una cadena de texto para generar una respuesta más coherente
        texto_completo = " ".join(contexto_dict[user_id])
        logging.info(f"Contexto actual para el usuario {user_id}: {texto_completo}")

        # Generar la respuesta basada en todo el contexto actual
        respuesta = self.generate_response(texto_completo)

        return respuesta

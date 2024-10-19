# model.py

import pickle
import pandas as pd
import numpy as np
import logging
import re
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model  # Importamos load_model aquí
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM

class NumerologyModel:
    def __init__(self, db):
        self.db = db
        self.model = None
        self.mlb = None
        self.is_trained = False
        self.mapping = {}  # Diccionario para el mapeo directo
        self.vibrations_by_day = {}  # Inicializar vibraciones por día
        self.most_delayed_numbers = {}  # Inicializar números más atrasados
        
        # Intentar cargar el modelo entrenado y el MultiLabelBinarizer
        try:
            self.model = load_model('numerology_model.keras')
            with open('mlb.pkl', 'rb') as f:
                self.mlb = pickle.load(f)
            with open('max_sequence_length.pkl', 'rb') as f:
                self.max_sequence_length = pickle.load(f)
            self.is_trained = True
            logging.info("Modelo, MultiLabelBinarizer y max_sequence_length cargados exitosamente.")
        except Exception as e:
            logging.warning(f"No se pudo cargar el modelo entrenado: {e}")

    # Método para cargar el modelo de numerología y el MultiLabelBinarizer
    def load_model(self, model_path):
        try:
            # Cargar el modelo entrenado
            self.model = load_model(model_path)
            logging.info(f"Modelo de numerología cargado exitosamente desde {model_path}.")

            # Cargar el MultiLabelBinarizer
            with open('mlb.pkl', 'rb') as f:
                self.mlb = pickle.load(f)
            logging.info("MultiLabelBinarizer cargado exitosamente.")

            # Cargar max_sequence_length
            with open('max_sequence_length.pkl', 'rb') as f:
                self.max_sequence_length = pickle.load(f)
            logging.info("max_sequence_length cargado exitosamente.")

            self.is_trained = True
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {e}")
            self.is_trained = False

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
                match = re.match(r'^.*?(\d{1,2})=\((\d{1,2}v?)\)=([\d{1,2}]+).*?$', line)
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
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
                    logging.debug(f"Patrón 6 encontrado: {line}")
                    matches_found = True
                    continue

                # Patrón 7: Formato especial para parejas XX=YY
                match = re.match(r'.*(\d{1,2})=(\d{1,2}v?)', line)
                if match:
                    input_number = int(match.group(1))
                    raw_numbers = [match.group(2)]
                    recommended_numbers = process_numbers_with_v(raw_numbers)
                    data.append({'input_number': input_number, 'recommended_numbers': recommended_numbers})
                    self.mapping.setdefault(input_number, []).extend(recommended_numbers)
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

                # Patrón 13: Formato de números atrasados (Centenas, Decenas, Terminales)
                match = re.match(r'^.*?(\d{1,2})[-–](\d+)\s*d[ií]as.*?$', line)
                if match and current_category:
                    number = int(match.group(1))
                    days = int(match.group(2))
                    category = current_category  # Necesitamos identificar la categoría actual
                    # Almacenar en delayed_numbers
                    self.delayed_numbers.setdefault(category, []).append({'number': number, 'days': days})
                    logging.debug(f"Patrón 13 (Números atrasados) encontrado en categoría {category}: {line}")
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

            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                logging.error("Los datos de entrenamiento no están disponibles. Asegúrate de que los datos fueron cargados correctamente.")
                self.is_trained = False
                return

            if self.X.size == 0 or len(self.y) == 0:
                logging.error("No hay datos suficientes para entrenar el modelo. X o y están vacíos.")
                self.is_trained = False
                return

            # Incorporar las vibraciones y otros datos en las características
            logging.info("Incorporando vibraciones y otros datos en las características...")
            X_features = []
            for idx, input_number in enumerate(self.X.flatten()):
                # Convertir el input_number a entero
                input_number = int(input_number)
                # Crear una lista de características
                features = [input_number]

                # Agregar día de la semana como característica
                current_date = datetime.now()
                day_of_week_es = self.get_day_in_spanish(current_date.strftime("%A"))
                day_vibrations = self.vibrations_by_day.get(day_of_week_es, {})
                digits = day_vibrations.get('digits', [])
                features.extend([int(digit) for digit in digits if digit.isdigit()])

                # Agregar vibraciones del número de entrada si existen
                number_vibrations = self.mapping.get(input_number, [])
                features.extend([int(num) for num in number_vibrations if num.isdigit()])

                # Agregar indicadores si el número de entrada es igual al más atrasado en cada categoría
                for category in ['CENTENAS', 'DECENAS', 'TERMINALES', 'PAREJAS']:
                    most_delayed = self.most_delayed_numbers.get(category)
                    is_most_delayed = 1 if most_delayed and input_number == most_delayed['number'] else 0
                    features.append(is_most_delayed)

                X_features.append(features)

            # Guardar la longitud máxima de secuencia
            self.max_sequence_length = max(len(seq) for seq in X_features)
            logging.info(f"Longitud máxima de secuencia establecida en: {self.max_sequence_length}")

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

            # Guardar el modelo en formato nativo de Keras
            self.model.save('numerology_model.keras')
            logging.info("Modelo guardado exitosamente como 'numerology_model.keras'.")

            # Guardar el MultiLabelBinarizer
            with open('mlb.pkl', 'wb') as f:
                pickle.dump(self.mlb, f)
            logging.info("MultiLabelBinarizer guardado exitosamente en 'mlb.pkl'.")

            # Guardar max_sequence_length
            with open('max_sequence_length.pkl', 'wb') as f:
                pickle.dump(self.max_sequence_length, f)
            logging.info("max_sequence_length guardado exitosamente.")

        except Exception as e:
            logging.error(f"Error durante el entrenamiento del modelo: {e}")
            self.is_trained = False


    def predict(self, input_number):
        # Usar mapeo directo si existe
        if input_number in self.mapping:
            recommended_numbers = self.mapping[input_number]
            logging.debug(f"Números recomendados (mapeo directo): {recommended_numbers}")
            return recommended_numbers
        elif self.is_trained:
            try:
                # Crear las mismas características que en el entrenamiento
                features = [input_number]
                logging.debug(f"Número de entrada: {input_number}")

                # Agregar día de la semana como característica
                current_date = datetime.now()
                day_of_week_es = self.get_day_in_spanish(current_date.strftime("%A"))
                day_vibrations = self.vibrations_by_day.get(day_of_week_es, {})
                digits = day_vibrations.get('digits', [])
                features.extend([int(digit) for digit in digits if digit.isdigit()])
                logging.debug(f"Vibraciones del día {day_of_week_es}: {digits}")

                # Agregar vibraciones del número de entrada si existen
                number_vibrations = self.mapping.get(input_number, [])
                features.extend([int(num) for num in number_vibrations if num.isdigit()])
                logging.debug(f"Vibraciones del número de entrada: {number_vibrations}")

                # Agregar indicadores si el número de entrada es igual al más atrasado en cada categoría
                for category in ['CENTENAS', 'DECENAS', 'TERMINALES', 'PAREJAS']:
                    most_delayed = self.most_delayed_numbers.get(category)
                    is_most_delayed = 1 if most_delayed and input_number == most_delayed['number'] else 0
                    features.append(is_most_delayed)
                    logging.debug(f"Indicador para categoría {category}: {is_most_delayed}")

                # Convertir las características a una matriz numpy y aplicar padding
                input_features = pad_sequences([features], padding='post', dtype='int32', maxlen=self.max_sequence_length)
                logging.debug(f"Características generadas para la predicción: {input_features}")

                # Realizar la predicción con la red neuronal
                prediction = self.model.predict(input_features)
                logging.debug(f"Predicción del modelo: {prediction}")

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
            logging.warning(f"No se encontraron recomendaciones para el número {input_number} y el modelo no está entrenado.")
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
            message += '<code>' + ', '.join(unique_recommended_numbers) + '</code>\n\n'

        # Números inseparables del número de entrada
        if inseparable_numbers:
            message += f"🔗 <b>Números inseparables del {input_number}:</b>\n"
            message += '<code>' + ', '.join(inseparable_numbers) + '</code>\n\n'

        # Raíz del número de entrada
        if root_numbers:
            message += f"🌿 <b>Raíz del número {input_number}:</b>\n"
            message += '<code>' + ', '.join(root_numbers) + '</code>\n\n'

        # Números más propensos a salir según las coincidencias de patrones
        if most_probable_numbers:
            message += "🌟 <b>Números más fuertes según patrones:</b>\n"
            message += '<code>' + ', '.join(most_probable_numbers) + '</code>\n\n'

        # Números más atrasados que coinciden con otros patrones
        if delayed_in_patterns:
            message += "⏳ <b>Números más atrasados que coinciden con otros patrones:</b>\n"
            message += '<code>' + ', '.join(delayed_in_patterns) + '</code>\n\n'
        else:
            message += "⏳ <b>Números más atrasados:</b>\n"
            message += '<code>' + ', '.join(most_delayed_numbers) + '</code>\n\n'

        # Vibraciones del día
        if day_numbers:
            message += f"📊 <b>Vibraciones para {day_of_week_es}:</b>\n"
            message += '<code>' + ', '.join(day_numbers) + '</code>\n\n'

        # Dígitos semanales obligatorios
        if day_digits:
            message += f"📅 <b>Dígitos semanales obligatorios para {day_of_week_es}:</b>\n"
            message += '<code>' + ', '.join(day_digits) + '</code>\n\n'

        # Parejas del día
        if day_parejas:
            message += f"🤝 <b>Parejas para {day_of_week_es}:</b>\n"
            message += '<code>' + ', '.join(day_parejas) + '</code>\n\n'

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
        self.db = db
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        self.max_sequence_length = 100
        self.num_words = 10000  # Limitar el vocabulario a 10,000 palabras más frecuentes
        self.processed_messages = set()  # Conjunto para almacenar mensajes únicos procesados

    def clean_text(self, text):
        # Eliminar emojis y caracteres especiales
        text = re.sub(r'[^\w\s]', '', text)  # Conserva solo caracteres alfanuméricos y espacios
        return text.strip().lower()  # Convertir a minúsculas para evitar duplicados por diferencia de mayúsculas

    def prepare_data(self):
        # Obtener todos los mensajes desde la base de datos
        logging.info("Preparando datos para el modelo de conversación...")
        all_messages = self.db.get_all_messages()  # Obtener todos los mensajes de la base de datos
        if not all_messages:
            logging.error("No se encontraron mensajes en la base de datos.")
            return

        messages = []
        for message_text in all_messages:
            # Limpiar el texto del mensaje
            cleaned_message = self.clean_text(message_text)

            # Si el mensaje ya ha sido procesado, ignorarlo
            if cleaned_message in self.processed_messages:
                #logging.info(f"Ignorando mensaje duplicado: {cleaned_message}")
                continue

            # Añadir el mensaje al conjunto de mensajes procesados
            self.processed_messages.add(cleaned_message)

            # Agregar el mensaje limpio a la lista de mensajes
            messages.append(cleaned_message)

        if not messages:
            logging.error("No se encontraron nuevos mensajes únicos para procesar.")
            return

        # Tokenizar los mensajes
        from tensorflow.keras.preprocessing.text import Tokenizer
        self.tokenizer = Tokenizer(num_words=self.num_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(messages)

        # Convertir los mensajes en secuencias numéricas
        sequences = self.tokenizer.texts_to_sequences(messages)
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)

        # Crear las etiquetas (siguiente palabra en la secuencia)
        y = []
        for seq in sequences:
            for i in range(1, len(seq)):
                input_sequence = seq[:i]
                target_word = seq[i]

                # Padding para que todas las secuencias tengan la misma longitud
                input_sequence = pad_sequences([input_sequence], maxlen=self.max_sequence_length)[0]
                X = np.vstack([X, input_sequence])
                y.append(target_word)

        self.X = X
        self.y = np.array(y)

        logging.debug(f"Ejemplo de secuencias generadas: {self.X[:5]}")
        logging.debug(f"Ejemplo de etiquetas generadas: {self.y[:5]}")

    def train(self):
        try:
            # Preparar los datos
            logging.info("Iniciando la preparación de los datos...")
            self.prepare_data()

            # Validar si los datos fueron cargados correctamente
            if self.X is None or len(self.X) == 0:
                logging.error("No se pudieron preparar los datos de entrenamiento.")
                self.is_trained = False
                return

            logging.info("Iniciando el entrenamiento del modelo conversacional...")

            # Verificar las dimensiones de X y y
            logging.info(f"Forma de X (entrada): {self.X.shape}")
            logging.info(f"Forma de y (objetivo): {self.y.shape}")

            # Definir el modelo secuencial LSTM
            self.model = Sequential()
            self.model.add(Embedding(input_dim=10000, output_dim=128, input_length=self.max_sequence_length))
            self.model.add(LSTM(64, return_sequences=False))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(10000, activation='softmax'))  # 10,000 clases (palabras más frecuentes)

            # Compilar el modelo
            logging.info("Compilando el modelo...")
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Entrenar el modelo
            logging.info("Entrenando el modelo conversacional...")
            self.model.fit(self.X, self.y, epochs=10, batch_size=32, verbose=1)

            # Indicar que el modelo fue entrenado correctamente
            self.is_trained = True
            logging.info("Entrenamiento del modelo conversacional completado con éxito.")

            # Guardar el modelo entrenado en formato Keras
            self.model.save('conversational_model.keras')
            logging.info("Modelo conversacional guardado exitosamente como 'conversational_model.keras'.")

            # Guardar el tokenizador u otros objetos necesarios
            with open('tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)  # Asume que tienes un tokenizador
            logging.info("Tokenizador guardado exitosamente en 'tokenizer.pkl'.")

        except Exception as e:
            logging.error(f"Error durante el entrenamiento del modelo conversacional: {e}")
            self.is_trained = False

    def generate_response(self, input_text):
        if not self.is_trained:
            logging.error("El modelo conversacional no ha sido entrenado.")
            return None

        logging.info(f"Generando respuesta para: {input_text}")

        # Preprocesar el texto de entrada
        input_sequence = self.tokenizer.texts_to_sequences([input_text])
        input_sequence = pad_sequences(input_sequence, maxlen=self.max_sequence_length)

        logging.info(f"Secuencia de entrada tokenizada: {input_sequence}")

        # Predecir la siguiente secuencia de texto
        predicted_sequence = self.model.predict(input_sequence)
        logging.info(f"Predicción generada por el modelo: {predicted_sequence}")

        predicted_word_index = np.argmax(predicted_sequence, axis=-1)
        logging.info(f"Índice de la palabra predicha: {predicted_word_index}")

        # Convertir el índice predicho en palabra
        predicted_word = self.tokenizer.index_word.get(predicted_word_index[0], '<UNK>')
        logging.info(f"Palabra predicha: {predicted_word}")

        return predicted_word if predicted_word != '<UNK>' else 'No puedo generar una respuesta adecuada.'

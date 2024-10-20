# database.py
import psycopg2
import os
import time
from datetime import datetime, timedelta

import logging
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )

    # Ya no es necesario crear tablas, eliminamos `create_tables`

    def get_all_formulas(self):
        with self.conn.cursor() as cur:
            # Seleccionar todos los mensajes que no sean nulos o vacíos
            cur.execute("""
            SELECT mensaje FROM logsfirewallids
            WHERE mensaje IS NOT NULL AND mensaje != ''
            """)
            result = cur.fetchall()

            # Devolver todos los mensajes sin modificar
            return [row[0] for row in result if row[0]]


    def get_all_interactions(self):
        with self.conn.cursor() as cur:
            # Seleccionamos todos los mensajes que no sean nulos o vacíos
            cur.execute("""
            SELECT mensaje FROM logsfirewallids
            WHERE mensaje IS NOT NULL AND mensaje != ''
            """)
            result = cur.fetchall()
            interactions = []

            for row in result:
                if row[0]:
                    try:
                        # Intentar dividir el mensaje en dos partes: input y recomendaciones
                        # Usamos una división más flexible sin el formato específico "User input: "
                        user_input, recommendations = row[0].split(", Recommendations: ")
                        interactions.append((user_input.strip(), recommendations.split(',')))
                    except ValueError as e:
                        #logging.warning(f"Error al procesar el mensaje: {row[0]}, error: {e}")
                        continue  # Evitar errores en mensajes mal formateados

            return interactions  # Retorna una lista de tuplas (user_input, recommendations)


    def get_all_messages(self):
        with self.conn.cursor() as cur:
            # Seleccionar todos los mensajes que no sean nulos o vacíos
            cur.execute("""
            SELECT mensaje FROM logsfirewallids
            WHERE mensaje IS NOT NULL AND mensaje != ''
            """)
            result = cur.fetchall()

            # Filtrar mensajes que sean cadenas no vacías
            messages = [row[0].strip() for row in result if row[0] and isinstance(row[0], str)]

            return messages

        self.processed_messages = set()  # Conjunto para almacenar mensajes únicos procesados

    def get_new_messages(self):
        """Obtiene los mensajes de los últimos 90 minutos que no se hayan procesado antes."""
        with self.conn.cursor() as cur:
            # Calcular la hora actual menos 90 minutos
            time_threshold = datetime.now() - timedelta(minutes=90)

            # Obtener los mensajes cuyo 'fechareciente' sea mayor que la hora calculada
            cur.execute("""
                SELECT mensaje FROM logsfirewallids
                WHERE mensaje IS NOT NULL 
                AND mensaje != ''
                AND fechareciente >= %s
                ORDER BY fechareciente ASC
            """, (time_threshold,))
            
            result = cur.fetchall()

            # Filtrar mensajes que no han sido procesados antes
            new_messages = []
            for row in result:
                message = row[0].strip()
                if message and isinstance(message, str) and message not in self.processed_messages:
                    new_messages.append(message)
                    self.processed_messages.add(message)  # Marcar como procesado

            return new_messages

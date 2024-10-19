# database.py

import psycopg2
import os
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
        self.create_tables()

    def create_tables(self):
        with self.conn.cursor() as cur:
            # Tabla de interacciones
            cur.execute("""
            CREATE TABLE IF NOT EXISTS interacciones (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                user_input TEXT NOT NULL,
                recommendations TEXT NOT NULL,
                fecha_interaccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            # Tabla de cache de respuestas
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cache_respuestas (
                id SERIAL PRIMARY KEY,
                cache_key TEXT UNIQUE NOT NULL,
                respuesta TEXT NOT NULL
            );
            """)
            # Tabla de numerología
            cur.execute("""
            CREATE TABLE IF NOT EXISTS numerologia (
                id SERIAL PRIMARY KEY,
                formula TEXT NOT NULL,
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            self.conn.commit()

    def save_interaction(self, user_id, user_input, recommendations):
        with self.conn.cursor() as cur:
            cur.execute("""
            INSERT INTO interacciones (user_id, user_input, recommendations)
            VALUES (%s, %s, %s)
            """, (user_id, user_input, ','.join(map(str, recommendations))))
            self.conn.commit()
        # Agregar log
        logging.info(f"Interacción guardada: usuario {user_id}, entrada '{user_input}', recomendaciones {recommendations}")


    def cache_response(self, cache_key, respuesta):
        with self.conn.cursor() as cur:
            cur.execute("""
            INSERT INTO cache_respuestas (cache_key, respuesta)
            VALUES (%s, %s)
            ON CONFLICT (cache_key) DO UPDATE SET respuesta = EXCLUDED.respuesta
            """, (cache_key, respuesta))
            self.conn.commit()

    def get_cached_response(self, cache_key):
        with self.conn.cursor() as cur:
            cur.execute("""
            SELECT respuesta FROM cache_respuestas WHERE cache_key = %s
            """, (cache_key,))
            result = cur.fetchone()
            return result[0] if result else None

    def insert_formula(self, formula):
        with self.conn.cursor() as cur:
            cur.execute("""
            INSERT INTO numerologia (formula)
            VALUES (%s)
            """, (formula,))
            self.conn.commit()

    def get_all_formulas(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            SELECT formula FROM numerologia
            """)
            result = cur.fetchall()
            return [row[0] for row in result]
        
    def get_all_interactions(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            SELECT user_input, recommendations FROM interacciones
            """)
            result = cur.fetchall()
            return result  # Retorna una lista de tuplas (user_input, recommendations)


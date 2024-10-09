# src/db/database.py

import os
import psycopg2
from dotenv import load_dotenv

# Cargar las variables de entorno desde .env
load_dotenv()

# Función para conectar a la base de datos
def connect_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT')
        )
        print("✅ Conexión a la base de datos establecida correctamente.")
        return conn
    except Exception as e:
        print(f"❌ Error al conectar con la base de datos: {e}")
        return None

# Función para crear tablas en la base de datos
def crear_tablas(conn):
    queries = [
        """
        CREATE TABLE IF NOT EXISTS datos_entrenamiento (
            id SERIAL PRIMARY KEY,
            input_data TEXT NOT NULL,
            output_data TEXT NOT NULL,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS logs_entrenamiento (
            id SERIAL PRIMARY KEY,
            log_message TEXT,
            log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS interacciones (
            id SERIAL PRIMARY KEY,
            user_id BIGINT,
            message TEXT,
            chat_type VARCHAR(10),
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]

    if conn:
        cursor = conn.cursor()
        for query in queries:
            try:
                cursor.execute(query)
                print(f"✅ Tabla creada o ya existente: {query.split('(')[0].strip()}")
            except Exception as e:
                print(f"❌ Error al crear la tabla: {e}")
        conn.commit()
        cursor.close()
        print("✅ Todas las tablas han sido creadas o ya existían.")
    else:
        print("❌ No se pudieron crear las tablas, no hay conexión a la base de datos.")

# Función para guardar interacciones en la base de datos
def guardar_interaccion(conn, user_id, message, chat_type):
    if conn:
        cursor = conn.cursor()
        query = "INSERT INTO interacciones (user_id, message, chat_type) VALUES (%s, %s, %s)"
        try:
            cursor.execute(query, (user_id, message, chat_type))
            conn.commit()
            print(f"✅ Interacción guardada: {message} de usuario {user_id}.")
        except Exception as e:
            print(f"❌ Error al guardar la interacción: {e}")
        finally:
            cursor.close()
    else:
        print("❌ No se pudo guardar la interacción, no hay conexión a la base de datos.")

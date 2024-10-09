# src/db/database.py

import os
import psycopg2
from dotenv import load_dotenv

# Cargar las variables de entorno
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
# Función para crear tablas (recibe la conexión ya existente)

def crear_tablas(conn):
    # Lista de consultas SQL para crear las tablas si no existen
    queries = [
        """
        -- Crea la tabla 'datos_entrenamiento' para almacenar datos de entrenamiento de IA
        CREATE TABLE IF NOT EXISTS datos_entrenamiento (
            id SERIAL PRIMARY KEY,                        -- Identificador único autoincremental
            input_data TEXT NOT NULL,                     -- Datos de entrada (texto)
            output_data TEXT NOT NULL,                    -- Datos de salida (resultado esperado)
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Fecha y hora en que se creó el registro
        );
        """,
        """
        -- Crea la tabla 'logs_entrenamiento' para almacenar logs durante el proceso de entrenamiento
        CREATE TABLE IF NOT EXISTS logs_entrenamiento (
            id SERIAL PRIMARY KEY,                        -- Identificador único autoincremental
            log_message TEXT,                             -- Mensaje del log
            log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Fecha y hora en que se generó el log
        );
        """,
        """
        -- Crea la tabla 'interacciones' para almacenar las interacciones de los usuarios con el bot
        CREATE TABLE IF NOT EXISTS interacciones (
            id SERIAL PRIMARY KEY,                        -- Identificador único autoincremental
            user_id BIGINT,                               -- ID del usuario de Telegram
            message TEXT,                                 -- Mensaje enviado por el usuario
            chat_type VARCHAR(10),                        -- Tipo de chat ('private' o 'group')
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP     -- Fecha y hora de la interacción
        );
        """
    ]

    # Verifica si la conexión a la base de datos está disponible
    if conn:
        # Crear un cursor para ejecutar las consultas SQL
        cursor = conn.cursor()
        
        # Iterar sobre las consultas SQL en la lista de queries
        for query in queries:
            try:
                # Ejecutar la consulta
                cursor.execute(query)
                # Imprimir un mensaje indicando qué tabla fue creada o ya existía
                print(f"✅ Tabla creada o ya existente para la consulta: {query.split('(')[0].strip()}")
            except Exception as e:
                # Capturar y mostrar cualquier error que ocurra al ejecutar la consulta
                print(f"❌ Error al crear la tabla: {e}")
        
        # Confirmar los cambios en la base de datos (commit)
        conn.commit()
        
        # Cerrar el cursor una vez que todas las consultas hayan sido ejecutadas
        cursor.close()
        
        # Imprimir un mensaje final indicando que todas las tablas han sido procesadas
        print("✅ Todas las tablas han sido creadas o ya existían.")
    else:
        # Si la conexión no está disponible, mostrar un mensaje de error
        print("❌ No se pudieron crear las tablas, no hay conexión a la base de datos.")

# Función para guardar interacciones (sin cerrar la conexión)
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
        
if __name__ == "__main__":
    # Conectarse a la base de datos y crear las tablas si no existen
    conexion = connect_db()
    if conexion:
        crear_tablas(conexion)
        conexion.close()

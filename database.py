import psycopg2
import os
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
        self.processed_messages = set()  # Conjunto para almacenar mensajes únicos procesados

        # Crear las tablas si no existen
        self.create_tables()

    def create_tables(self):
        """Crea las tablas necesarias si no existen"""
        with self.conn.cursor() as cur:
            # Tabla para mensajes procesados
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_messages (
                    id SERIAL PRIMARY KEY,
                    mensaje TEXT UNIQUE NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Tabla logsfirewallids (ya está en NodeJS pero la incluyo por seguridad)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS logsfirewallids (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    nombre TEXT,
                    alias TEXT,
                    grupo BIGINT,
                    nombregrupo TEXT,
                    privado BOOLEAN,
                    fechareciente TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ultima_notificacion TIMESTAMP DEFAULT NULL,
                    notificado BOOLEAN DEFAULT false,
                    expulsado BOOLEAN DEFAULT false,
                    biografia TEXT,
                    chatgpt BOOLEAN DEFAULT false,
                    mensaje TEXT
                );
            """)

            # Nueva tabla group_converse
            cur.execute("""
                CREATE TABLE IF NOT EXISTS group_converse (
                    id SERIAL PRIMARY KEY,
                    type TEXT,
                    serial TEXT,
                    group_id BIGINT UNIQUE NOT NULL,
                    inserted_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            self.conn.commit()

    def add_group(self, group_id, group_type, serial):
        """Añade un nuevo grupo a la tabla group_converse si no existe"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO group_converse (group_id, type, serial)
                VALUES (%s, %s, %s)
                ON CONFLICT (group_id) DO NOTHING
            """, (group_id, group_type, serial))
            self.conn.commit()

    def is_group_registered(self, group_id):
        """Verifica si un grupo ya está registrado en la base de datos"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM group_converse WHERE group_id = %s
            """, (group_id,))
            result = cur.fetchone()
            return result[0] > 0
        
    def delete_group(self, group_id):
        """
        Elimina un grupo de la tabla group_converse basado en el group_id.
        
        Args:
            group_id (int): El ID del grupo que deseas eliminar.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                DELETE FROM group_converse WHERE group_id = %s
            """, (group_id,))
            self.conn.commit()
            logging.info(f"Grupo con group_id {group_id} eliminado exitosamente.")

    def get_groups(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT group_id, type FROM group_converse")
                return cur.fetchall()
        except Exception as e:
            logging.error(f"Error al obtener los grupos: {e}")
            return []



    def get_owner_info(self, owner_id):
        """Obtiene la información más reciente del propietario desde logsfirewallids"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT nombre, alias FROM logsfirewallids
                WHERE user_id = %s
                ORDER BY fechareciente DESC LIMIT 1
            """, (owner_id,))
            result = cur.fetchone()
            if result:
                return {"nombre": result[0], "alias": result[1]}
            return None


    def get_all_formulas(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            SELECT mensaje FROM logsfirewallids
            WHERE mensaje IS NOT NULL AND mensaje != ''
            """)
            result = cur.fetchall()
            return [row[0] for row in result if row[0]]
    
    def get_all_interactions(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            SELECT mensaje FROM logsfirewallids
            WHERE mensaje IS NOT NULL AND mensaje != ''
            """)
            result = cur.fetchall()
            interactions = []
            for row in result:
                if row[0]:
                    try:
                        user_input, recommendations = row[0].split(", Recommendations: ")
                        interactions.append((user_input.strip(), recommendations.split(',')))
                    except ValueError as e:
                        continue
            return interactions

    def get_new_messages(self):
        """Obtiene los mensajes de los últimos 90 minutos que no se hayan procesado antes."""
        with self.conn.cursor() as cur:
            time_threshold = datetime.now() - timedelta(minutes=90)
            cur.execute("""
                SELECT mensaje FROM logsfirewallids
                WHERE mensaje IS NOT NULL 
                AND mensaje != ''
                AND fechareciente >= %s
                ORDER BY fechareciente ASC
            """, (time_threshold,))
            result = cur.fetchall()
            new_messages = []
            for row in result:
                message = row[0].strip()
                if message and isinstance(message, str) and not self.is_message_processed(message):
                    new_messages.append(message)
                    self.save_processed_message(message)
            return new_messages

    def is_message_processed(self, message):
        """Verifica si el mensaje ya fue procesado"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM processed_messages WHERE mensaje = %s", (message,))
            result = cur.fetchone()
            return result[0] > 0

    def save_processed_message(self, message):
        """Guarda un mensaje como procesado en la base de datos"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO processed_messages (mensaje) 
                VALUES (%s) 
                ON CONFLICT DO NOTHING
            """, (message,))
            self.conn.commit()

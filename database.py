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


    def get_conn(self):
        return self.connection_pool.getconn()

    def put_conn(self, conn):
        self.connection_pool.putconn(conn)

    def close_connection(self):
        """Cierra la conexión a la base de datos"""
        try:
            if self.conn:
                self.conn.close()
                logging.info("Conexión a la base de datos cerrada correctamente.")
        except Exception as e:
            logging.error(f"Error al cerrar la conexión de la base de datos: {e}")

    def create_tables(self):
        """Crea las tablas necesarias si no existen"""
        try:
            with self.conn.cursor() as cur:
                # Crear la tabla de charadas con la restricción UNIQUE en la columna numero
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS charadas (
                        id SERIAL PRIMARY KEY,
                        numero INTEGER NOT NULL,
                        significado TEXT NOT NULL,
                        UNIQUE(numero, significado)  -- Agregar restricción única en número y significado
                    );
                """)
                self.conn.commit()
                logging.info("Tablas de la base de datos verificadas/creadas.")
        except Exception as e:
            logging.error(f"Error al crear tablas en la base de datos: {e}")


        """Crea las tabla processed_messages"""
        try:
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

                # Crear tabla para retroalimentación
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        feedback_type TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                self.conn.commit()
                logging.info("Tablas de la base de datos verificadas/creadas.")

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
        except Exception as e:
            logging.error(f"Error al crear tablas en la base de datos: {e}")


    def add_group(self, group_id, group_type, serial):
        """Añade un nuevo grupo a la tabla group_converse si no existe"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO group_converse (group_id, type, serial)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (group_id) DO NOTHING
                """, (group_id, group_type, serial))
                self.conn.commit()
                logging.info(f"Grupo {group_id} añadido exitosamente a la base de datos.")
        except Exception as e:
            logging.error(f"Error al añadir el grupo {group_id}: {e}")


    def is_group_registered(self, group_id):
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                logging.info(f"Verificando si el grupo {group_id} está registrado en la base de datos")
                cur.execute("""
                    SELECT COUNT(*) FROM group_converse WHERE group_id = %s
                """, (group_id,))
                result = cur.fetchone()
                logging.info(f"Resultado de la consulta para el grupo {group_id}: {result[0]}")
                return result[0] > 0
        except Exception as e:
            logging.error(f"Error verificando el grupo: {e}")
            conn.rollback()
            return False
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool



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
    
    def get_numerology_adjustments(self):
        """Obtiene los nuevos datos o interacciones para realizar el ajuste fino del modelo de numerología."""
        try:
            with self.conn.cursor() as cur:
                # Seleccionar los mensajes más recientes que contienen ajustes de numerología
                cur.execute("""
                    SELECT mensaje FROM logsfirewallids
                    WHERE mensaje IS NOT NULL
                    AND mensaje != ''
                    AND fechareciente >= NOW() - INTERVAL '1 DAY'  -- Ajuste de las últimas 24 horas
                    ORDER BY fechareciente ASC
                """)
                result = cur.fetchall()
                return [row[0] for row in result if row[0]]
        except Exception as e:
            logging.error(f"Error al obtener los ajustes de numerología: {e}")
            return []

    def verificar_grupo_activado(self, group_id):
        """Verifica si un grupo está activado en la tabla group_converse."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM group_converse WHERE group_id = %s
                """, (group_id,))
                result = cur.fetchone()
                return result[0] > 0  # Devuelve True si el grupo está registrado, False si no
        except Exception as e:
            self.conn.rollback()  # Realiza un rollback en caso de error
            logging.error(f"Error verificando si el grupo está activado: {e}")
            return False


    def save_feedback(self, user_id, feedback_type):
        try:
            with self.conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO feedback (user_id, feedback_type)
                    VALUES (%s, %s)
                ''', (user_id, feedback_type))
                self.conn.commit()
                logging.info(f"Retroalimentación guardada para el usuario {user_id}.")
        except Exception as e:
            logging.error(f"Error al guardar retroalimentación en la base de datos: {e}")
            raise e

    def add_charada(self, numero, significado):
        """Añade o actualiza el significado de un número en la tabla charada."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO charada (numero, significado) 
                    VALUES (%s, %s) 
                    ON CONFLICT (numero) DO UPDATE SET significado = EXCLUDED.significado
                """, (numero, significado))
                self.conn.commit()
                logging.info(f"Charada para el número {numero} añadida/actualizada exitosamente.")
        except Exception as e:
            logging.error(f"Error al guardar la charada: {e}")

    def get_charada_by_numero(self, numero):
        """Obtiene el significado de un número específico en la tabla charada."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT significado FROM charada WHERE numero = %s", (numero,))
                result = cur.fetchone()
                if result:
                    return result[0]  # Retorna el significado
                else:
                    return None
        except Exception as e:
            logging.error(f"Error al obtener charada para el número {numero}: {e}")
            return None

    def get_charada_by_keyword(self, keyword):
        """Obtiene todos los números relacionados con una palabra clave en la tabla charada."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT numero FROM charada WHERE significado ILIKE %s", (f"%{keyword}%",))
                result = cur.fetchall()
                return [row[0] for row in result]
        except Exception as e:
            logging.error(f"Error al obtener charadas por palabra clave '{keyword}': {e}")
            return []

    def get_significado_por_numero(self, numero):
        """Obtiene los significados asociados a un número en la charada."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT significado FROM charada WHERE numero = %s", (numero,))
            resultado = cur.fetchone()
            if resultado:
                return resultado[0].split(',')  # Retorna los significados como una lista
            return None

    def buscar_numeros_por_significado(self, palabra_clave):
        """Busca todos los números en los que aparece una palabra en los significados."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT numero, significado 
                FROM charada 
                WHERE significado ILIKE %s
            """, (f"%{palabra_clave}%",))
            resultados = cur.fetchall()
            if resultados:
                return [(numero, significado.split(',')) for numero, significado in resultados]
            return None


    def actualizar_charada(self, numero, significados_nuevos):
        """Actualiza o inserta significados en la charada para un número dado."""
        with self.conn.cursor() as cur:
            # Obtener los significados actuales
            cur.execute("SELECT significado FROM charada WHERE numero = %s", (numero,))
            result = cur.fetchone()

            if result:
                # Actualizar el significado si ya existe
                significados_existentes = result[0].split(',')
                nuevos_significados = list(set(significados_existentes + significados_nuevos))
                cur.execute("UPDATE charada SET significado = %s WHERE numero = %s", (','.join(nuevos_significados), numero))
                logging.info(f"Actualizados significados para el número {numero}: {nuevos_significados}")
            else:
                # Insertar nuevo número y significado si no existe
                cur.execute("INSERT INTO charada (numero, significado) VALUES (%s, %s)", (numero, ','.join(significados_nuevos)))
                logging.info(f"Insertado número {numero} con significados: {significados_nuevos}")
            
            self.conn.commit()
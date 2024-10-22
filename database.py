import psycopg2
from psycopg2 import pool
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import hashlib


load_dotenv()

class Database:
    def __init__(self):
        # Inicializar el pool de conexiones con un rango de 1 a 10 conexiones
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # Mínimo 1, máximo 10 conexiones
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
        """Obtener una conexión del pool"""
        return self.connection_pool.getconn()

    def put_conn(self, conn):
        """Devolver la conexión al pool"""
        self.connection_pool.putconn(conn)

    def close_all_connections(self):
        """Cerrar todas las conexiones en el pool"""
        self.connection_pool.closeall()

    def create_tables(self):
        """Crea las tablas necesarias si no existen"""
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS charadas (id SERIAL PRIMARY KEY, numero INTEGER NOT NULL, significado TEXT NOT NULL, UNIQUE(numero, significado));""")
                cur.execute("""CREATE TABLE IF NOT EXISTS processed_messages (id SERIAL PRIMARY KEY, mensaje TEXT UNIQUE NOT NULL, processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);""")
                cur.execute("""CREATE TABLE IF NOT EXISTS logsfirewallids (id SERIAL PRIMARY KEY, user_id BIGINT, nombre TEXT, alias TEXT, grupo BIGINT, nombregrupo TEXT, privado BOOLEAN, fechareciente TIMESTAMP DEFAULT CURRENT_TIMESTAMP, ultima_notificacion TIMESTAMP DEFAULT NULL, notificado BOOLEAN DEFAULT false, expulsado BOOLEAN DEFAULT false, biografia TEXT, chatgpt BOOLEAN DEFAULT false, mensaje TEXT);""")
                cur.execute("""CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL, feedback_type TEXT NOT NULL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);""")
                cur.execute("""CREATE TABLE IF NOT EXISTS group_converse (id SERIAL PRIMARY KEY, type TEXT, serial TEXT, group_id BIGINT UNIQUE NOT NULL, inserted_at TIMESTAMP DEFAULT NOW());""")
                conn.commit()
                logging.info("Tablas de la base de datos verificadas/creadas.")
        except Exception as e:
            logging.error(f"Error al crear tablas en la base de datos: {e}")
        finally:
            self.put_conn(conn)


    def add_group(self, group_id, group_type, serial):
        """Añade un nuevo grupo a la tabla group_converse si no existe"""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO group_converse (group_id, type, serial)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (group_id) DO NOTHING
                """, (group_id, group_type, serial))
                conn.commit()
                logging.info(f"Grupo {group_id} añadido exitosamente a la base de datos.")
        except Exception as e:
            logging.error(f"Error al añadir el grupo {group_id}: {e}")
            conn.rollback()
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool

    def is_group_registered(self, group_id):
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                logging.info(f"Verificando si el grupo {group_id} está registrado en la base de datos")
                cur.execute("SELECT COUNT(*) FROM group_converse WHERE group_id = %s", (group_id,))
                result = cur.fetchone()
                logging.info(f"Resultado de la consulta para el grupo {group_id}: {result[0]}")
                return result[0] > 0
        except Exception as e:
            logging.error(f"Error verificando el grupo: {e}")
            conn.rollback()
            return False
        finally:
            self.put_conn(conn)



    def delete_group(self, group_id):
        """
        Elimina un grupo de la tabla group_converse basado en el group_id.
        """
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM group_converse WHERE group_id = %s
                """, (group_id,))
                conn.commit()
                logging.info(f"Grupo con group_id {group_id} eliminado exitosamente.")
        except Exception as e:
            logging.error(f"Error al eliminar el grupo {group_id}: {e}")
            conn.rollback()
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def get_groups(self):
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT group_id, type FROM group_converse")
                return cur.fetchall()
        except Exception as e:
            logging.error(f"Error al obtener los grupos: {e}")
            return []
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def get_owner_info(self, owner_id):
        """Obtiene la información más reciente del propietario desde logsfirewallids"""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT nombre, alias FROM logsfirewallids
                    WHERE user_id = %s
                    ORDER BY fechareciente DESC LIMIT 1
                """, (owner_id,))
                result = cur.fetchone()
                if result:
                    return {"nombre": result[0], "alias": result[1]}
                return None
        except Exception as e:
            logging.error(f"Error al obtener la información del propietario: {e}")
            return None
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool



    def get_all_formulas(self):
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT mensaje FROM logsfirewallids
                WHERE mensaje IS NOT NULL AND mensaje != ''
                """)
                result = cur.fetchall()
                return [row[0] for row in result if row[0]]
        except Exception as e:
            logging.error(f"Error al obtener las fórmulas: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
            return []
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool

    
    def get_all_interactions(self):
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
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
        except Exception as e:
            logging.error(f"Error al obtener las interacciones: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
            return []
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool

    def get_new_messages(self):
        """Obtiene los mensajes de los últimos 90 minutos que no se hayan procesado antes."""
        conn = self.get_conn()  # Obtener conexión del pool
        new_messages = []  # Inicializar lista para mensajes nuevos
        try:
            time_threshold = datetime.now() - timedelta(minutes=90)  # Calcular el umbral de tiempo

            with conn.cursor() as cur:
                # Optimizar la consulta para excluir mensajes ya procesados
                cur.execute("""
                    SELECT mensaje 
                    FROM logsfirewallids
                    WHERE mensaje IS NOT NULL 
                    AND mensaje != ''
                    AND fechareciente >= %s
                    AND NOT EXISTS (
                        SELECT 1 FROM processed_messages 
                        WHERE md5(mensaje) = md5(logsfirewallids.mensaje)
                    )
                    ORDER BY fechareciente ASC
                """, (time_threshold,))
                
                result = cur.fetchall()
                logging.info(f"Se encontraron {len(result)} mensajes nuevos en la base de datos.")

                # Procesar los resultados
                for row in result:
                    message = row[0].strip()
                    if message and isinstance(message, str):
                        new_messages.append(message)
                        self.save_processed_message(message)  # Guardar el mensaje como procesado

                logging.info(f"Se procesaron {len(new_messages)} mensajes nuevos correctamente.")

            return new_messages

        except psycopg2.DatabaseError as db_error:
            logging.error(f"Error en la base de datos al obtener los nuevos mensajes: {db_error}")
            conn.rollback()  # Revertir la transacción en caso de error de base de datos
            return []
        
        except Exception as e:
            logging.error(f"Error general al obtener los nuevos mensajes: {e}")
            return []
        
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def is_message_processed(self, message):
        """Verifica si el mensaje ya fue procesado"""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            # Generar hash MD5 del mensaje
            message_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM processed_messages WHERE mensaje = %s", (message_hash,))
                result = cur.fetchone()
                return result[0] > 0
        except Exception as e:
            logging.error(f"Error al verificar si el mensaje fue procesado: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
            return False
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool



    def save_processed_message(self, message):
        """Guarda un mensaje como procesado en la base de datos"""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            # Generar hash MD5 del mensaje para almacenar un identificador más pequeño
            message_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processed_messages (mensaje) 
                    VALUES (%s) 
                    ON CONFLICT DO NOTHING
                """, (message_hash,))
                conn.commit()  # Confirmar la transacción
        except Exception as e:
            logging.error(f"Error al guardar el mensaje procesado: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool

    
    def get_numerology_adjustments(self):
        """Obtiene los nuevos datos o interacciones para realizar el ajuste fino del modelo de numerología."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
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
            conn.rollback()  # Revertir la transacción en caso de error
            return []
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def verificar_grupo_activado(self, group_id):
        """Verifica si un grupo está activado en la tabla group_converse."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM group_converse WHERE group_id = %s
                """, (group_id,))
                result = cur.fetchone()
                return result[0] > 0  # Devuelve True si el grupo está registrado, False si no
        except Exception as e:
            logging.error(f"Error verificando si el grupo está activado: {e}")
            conn.rollback()  # Realiza un rollback en caso de error
            return False
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool



    def save_feedback(self, user_id, feedback_type):
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO feedback (user_id, feedback_type)
                    VALUES (%s, %s)
                ''', (user_id, feedback_type))
                conn.commit()  # Confirmar la transacción
                logging.info(f"Retroalimentación guardada para el usuario {user_id}.")
        except Exception as e:
            logging.error(f"Error al guardar retroalimentación en la base de datos: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
            raise e
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def add_charada(self, numero, significado):
        """Añade o actualiza el significado de un número en la tabla charada."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO charada (numero, significado) 
                    VALUES (%s, %s) 
                    ON CONFLICT (numero) DO UPDATE SET significado = EXCLUDED.significado
                """, (numero, significado))
                conn.commit()  # Confirmar la transacción
                logging.info(f"Charada para el número {numero} añadida/actualizada exitosamente.")
        except Exception as e:
            logging.error(f"Error al guardar la charada: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def get_charada_by_numero(self, numero):
        """Obtiene el significado de un número específico en la tabla charada."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT significado FROM charada WHERE numero = %s", (numero,))
                result = cur.fetchone()
                if result:
                    return result[0]  # Retorna el significado
                else:
                    return None
        except Exception as e:
            logging.error(f"Error al obtener charada para el número {numero}: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
            return None
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def get_charada_by_keyword(self, keyword):
        """Obtiene todos los números relacionados con una palabra clave en la tabla charada."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT numero FROM charada WHERE significado ILIKE %s", (f"%{keyword}%",))
                result = cur.fetchall()
                return [row[0] for row in result]
        except Exception as e:
            logging.error(f"Error al obtener charadas por palabra clave '{keyword}': {e}")
            conn.rollback()  # Revertir la transacción en caso de error
            return []
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def get_significado_por_numero(self, numero):
        """Obtiene los significados asociados a un número en la charada."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT significado FROM charada WHERE numero = %s", (numero,))
                resultado = cur.fetchone()
                if resultado:
                    return resultado[0].split(',')  # Retorna los significados como una lista
                return None
        except Exception as e:
            logging.error(f"Error al obtener los significados para el número {numero}: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
            return None
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


    def buscar_numeros_por_significado(self, palabra_clave):
        """Busca todos los números en los que aparece una palabra en los significados."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT numero, significado 
                    FROM charada 
                    WHERE significado ILIKE %s
                """, (f"%{palabra_clave}%",))
                resultados = cur.fetchall()
                if resultados:
                    return [(numero, significado.split(',')) for numero, significado in resultados]
                return None
        except Exception as e:
            logging.error(f"Error al buscar números por significado con la palabra clave '{palabra_clave}': {e}")
            conn.rollback()  # Revertir la transacción en caso de error
            return None
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool



    def actualizar_charada(self, numero, significados_nuevos):
        """Actualiza o inserta significados en la charada para un número dado."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
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
                
                conn.commit()  # Confirmar la transacción
        except Exception as e:
            logging.error(f"Error al actualizar o insertar charada para el número {numero}: {e}")
            conn.rollback()  # Revertir la transacción en caso de error
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool

    def get_all_messages(self):
        """Obtiene todos los mensajes de la tabla logsfirewallids."""
        conn = self.get_conn()  # Obtener conexión del pool
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT mensaje FROM logsfirewallids
                    WHERE mensaje IS NOT NULL 
                    AND mensaje != ''
                """)
                result = cur.fetchall()
                return [row[0] for row in result if row[0]]
        except Exception as e:
            logging.error(f"Error al obtener los mensajes de la tabla logsfirewallids: {e}")
            return []
        finally:
            self.put_conn(conn)  # Devolver la conexión al pool


# db_utils.py
import psycopg2

def obtener_interacciones():
    conn = psycopg2.connect(database="tu_db", user="tu_usuario", password="tu_password", host="localhost", port="5432")
    cursor = conn.cursor()
    cursor.execute("SELECT message FROM interacciones")
    mensajes = cursor.fetchall()
    cursor.close()
    conn.close()
    return [mensaje[0] for mensaje in mensajes]

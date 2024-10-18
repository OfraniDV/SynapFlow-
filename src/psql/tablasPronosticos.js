import pool from './db.js';  // Importamos la conexión a la base de datos

const createTables = async () => {
  const numerologiaQuery = `
    CREATE TABLE IF NOT EXISTS numerologia (
      id SERIAL PRIMARY KEY,
      formula TEXT NOT NULL,
      fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `;

  const interaccionesQuery = `
    CREATE TABLE IF NOT EXISTS interacciones (
      id SERIAL PRIMARY KEY,
      user_id BIGINT NOT NULL,
      firstname VARCHAR(255),
      username VARCHAR(255),
      interaccion TEXT NOT NULL,
      fecha_interaccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `;

  try {
    await pool.query(numerologiaQuery);
    console.log('Tabla "numerologia" creada o ya existe.');
    
    await pool.query(interaccionesQuery);
    console.log('Tabla "interacciones" creada o ya existe.');
  } catch (err) {
    console.error('Error al crear las tablas:', err.stack);
  }
};

createTables();  // Ejecutamos la función para crear las tablas

export default createTables;

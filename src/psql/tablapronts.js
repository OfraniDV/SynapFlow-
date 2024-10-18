import pool from './db.js';  // Importamos la conexión a la base de datos

const createProntsTable = async () => {
  const query = `
    CREATE TABLE IF NOT EXISTS pronts (
      id SERIAL PRIMARY KEY,
      pregunta VARCHAR(255) NOT NULL,
      prompt TEXT NOT NULL
    );
  `;

  try {
    await pool.query(query);
    console.log('Tabla "pronts" creada o ya existe.');
  } catch (err) {
    console.error('Error al crear la tabla "pronts":', err.stack);
  }
};

createProntsTable();  // Ejecutamos la función para crear la tabla

export default createProntsTable;

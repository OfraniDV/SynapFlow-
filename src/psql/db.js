import pkg from 'pg';  // Importa todo el módulo como pkg
const { Pool } = pkg;  // Extrae Pool de pkg

import dotenv from 'dotenv';
dotenv.config();  // Cargar variables de entorno

// Crea una nueva instancia de Pool usando las variables de entorno
const pool = new Pool({
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  host: process.env.DB_URL,
  database: process.env.DB_NAME,
  /*ssl: {
    rejectUnauthorized: false  // Descomentar si usas SSL
  }*/
});

// Conexión a la base de datos y manejo de errores
pool.connect((err) => {
  if (err) {
    console.error('Error conectando a la base de datos:', err.stack);
  } else {
    console.log('Conexión exitosa con la base de datos');
  }
});

export default pool;

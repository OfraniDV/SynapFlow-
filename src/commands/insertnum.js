import pool from '../psql/db.js';  // Importar la conexión a la base de datos
import dotenv from 'dotenv';

// Cargar las variables de entorno
dotenv.config();

export default {
  name: 'insertnum',  // Nombre del comando
  execute: async (ctx) => {
    const OWNER_ID = process.env.OWNER_ID;  // ID del administrador

    // Verificar si el usuario que ejecuta el comando es el administrador
    if (ctx.from.id.toString() !== OWNER_ID) {
      return ctx.reply('No tienes permiso para usar este comando.');
    }

    const messageText = ctx.message.text;
    const formula = messageText.replace(/^\/insertnum\s*/i, '').trim();  // Elimina '/insertnum' y cualquier espacio extra

    // Verificar si hay texto después del comando
    if (!formula) {
      return ctx.reply('Por favor, proporciona la fórmula o texto de numerología que deseas insertar.');
    }

    try {
      // Insertar la información en la tabla numerologia
      const query = 'INSERT INTO numerologia (formula) VALUES ($1) RETURNING id';
      const result = await pool.query(query, [formula]);

      const insertedId = result.rows[0].id;
      console.log(`Nueva fórmula insertada con ID: ${insertedId}`);

      // Confirmar la inserción al administrador
      ctx.reply(`Fórmula insertada correctamente con ID: ${insertedId}`);
    } catch (err) {
      console.error('Error al insertar la fórmula en la base de datos:', err.stack);
      ctx.reply('Ocurrió un error al intentar insertar la fórmula.');
    }
  }
};

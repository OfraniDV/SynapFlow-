import pool from '../psql/db.js';  // Importamos la conexión a la base de datos

export default {
  name: 'pronts',
  execute: async (ctx) => {
    const messageText = ctx.message.text;
    console.log('Mensaje recibido:', messageText);

    // Separar la pregunta y el prompt (Ejemplo: /pronts pregunta | prompt)
    const [pregunta, prompt] = messageText.replace(/^\/pronts\s*/, '').split('|').map(str => str.trim());

    // Validar que se haya proporcionado una pregunta y un prompt
    if (!pregunta || !prompt) {
      return ctx.reply('Por favor, usa el formato: /pronts pregunta | prompt');
    }

    try {
      const query = 'INSERT INTO pronts (pregunta, prompt) VALUES ($1, $2) RETURNING id';
      const values = [pregunta, prompt];

      const result = await pool.query(query, values);

      if (result.rows.length > 0) {
        const promptId = result.rows[0].id;
        ctx.reply(`Nuevo prompt agregado con ID: ${promptId}`);
      } else {
        ctx.reply('Hubo un problema al agregar el prompt.');
      }

    } catch (error) {
      console.error('Error al insertar prompt en la base de datos:', error.stack);
      ctx.reply('Ocurrió un error al agregar el prompt a la base de datos.');
    }
  }
};

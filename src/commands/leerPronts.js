import pool from '../psql/db.js';  // Importamos la conexión a la base de datos

export default {
  name: 'leerPronts',
  execute: async (ctx) => {
    try {
      const query = 'SELECT id, pregunta, prompt FROM pronts';
      const result = await pool.query(query);

      if (result.rows.length > 0) {
        const formattedPronts = result.rows.map(pront => `ID: ${pront.id}\nPregunta: ${pront.pregunta}\nPrompt: ${pront.prompt}`).join('\n\n');
        ctx.reply(`Pronts existentes:\n\n${formattedPronts}`);
      } else {
        ctx.reply('No hay pronts en la base de datos.');
      }

    } catch (error) {
      console.error('Error al obtener los prompts:', error.stack);
      ctx.reply('Ocurrió un error al obtener los pronts de la base de datos.');
    }
  }
};

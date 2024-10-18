import pool from '../psql/db.js';  // Importamos la conexión a la base de datos

export default {
  name: 'eliminarPront',
  execute: async (ctx) => {
    const messageText = ctx.message.text;
    const prontId = messageText.replace(/^\/eliminarPront\s*/, '').trim();

    if (!prontId) {
      return ctx.reply('Por favor, proporciona el ID del pront que deseas eliminar. Ejemplo: /eliminarPront 1');
    }

    try {
      const query = 'DELETE FROM pronts WHERE id = $1 RETURNING id';
      const result = await pool.query(query, [prontId]);

      if (result.rows.length > 0) {
        ctx.reply(`Pront con ID ${prontId} eliminado con éxito.`);
      } else {
        ctx.reply(`No se encontró ningún pront con el ID ${prontId}.`);
      }

    } catch (error) {
      console.error('Error al eliminar el pront:', error.stack);
      ctx.reply('Ocurrió un error al intentar eliminar el pront.');
    }
  }
};

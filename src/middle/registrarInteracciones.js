import pool from '../psql/db.js';  // Importa la conexión a la base de datos

const registrarInteracciones = async (ctx, next) => {
  if (ctx.message && ctx.chat && ctx.chat.type === 'supergroup') {  // Verifica si es un mensaje de grupo
    const { from, text } = ctx.message;
    const user_id = from.id;
    const firstname = from.first_name || 'Desconocido';
    const username = from.username || 'No tiene';
    const interaccion = text || 'Sin mensaje';

    const query = `
      INSERT INTO interacciones (user_id, firstname, username, interaccion, fecha_interaccion)
      VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
    `;

    const values = [user_id, firstname, username, interaccion];

    try {
      await pool.query(query, values);
      console.log(`Interacción registrada para el usuario ${firstname} (${username}).`);
    } catch (err) {
      console.error('Error al registrar la interacción:', err.stack);
    }
  }

  await next();  // Pasa al siguiente middleware o manejador
};

export default registrarInteracciones;

import OpenAI from 'openai';
import dotenv from 'dotenv';
import pool from '../psql/db.js';  // Importar el pool de conexión a la base de datos

dotenv.config(); // Cargar variables de entorno

// Inicializa el cliente de OpenAI con la clave API
const openai = new OpenAI({
  apiKey: process.env.API_AI,  // Usando la variable de entorno correcta
});

export default {
  name: 'synap',  // Nombre del comando que será 'synap'
  execute: async (ctx) => {
    const VIP_GROUP_ID = process.env.ID_GRUPO_VIP;

    // Verificar si el mensaje proviene del grupo VIP
    if (ctx.chat.id.toString() !== VIP_GROUP_ID) {
      console.log(`Mensaje ignorado de chat ID: ${ctx.chat.id}`);
      // Opcional: Puedes enviar un mensaje informativo al usuario
      return ctx.reply('Este comando solo está disponible en el grupo VIP.');
    }

    const messageText = ctx.message.text;
    console.log("Mensaje recibido:", messageText);

    // Eliminar el comando "/synap" y procesar la pregunta
    const userInput = messageText.replace(/^\/synap\s*/i, '').trim();  // Elimina '/synap' y cualquier espacio extra
    console.log("Entrada del usuario para OpenAI:", userInput);

    // Si no hay entrada, envía un mensaje de error al usuario
    if (!userInput) {
      console.log("No se encontró entrada después de 'synap'");
      return ctx.reply('Por favor, escribe una pregunta después de "synap".');
    }

    try {
      // Instrucción a OpenAI para generar consultas SQL válidas que buscan información en la tabla "numerologia"
      const openaiPrompt = `
        Eres un asistente que genera consultas SQL válidas para una base de datos PostgreSQL. 
        La tabla para buscar información se llama 'numerologia'. 
        Genera una consulta SQL válida para la siguiente pregunta en lenguaje natural:
        "${userInput}"
      `;

      // Realiza la llamada a la API de OpenAI utilizando el modelo GPT-3.5
      const response = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [{ role: 'system', content: openaiPrompt }],
        temperature: 0.7,
      });

      console.log("Respuesta de OpenAI:", response);

      // Verificar si OpenAI devolvió una consulta SQL
      if (response.choices && response.choices.length > 0) {
        const sqlQuery = response.choices[0].message.content.trim();
        console.log("Consulta SQL generada:", sqlQuery);

        // Ejecutar la consulta SQL en la base de datos
        pool.query(sqlQuery, (err, result) => {
          if (err) {
            console.error('Error ejecutando la consulta SQL:', err.stack);
            return ctx.reply('Ocurrió un error al ejecutar la consulta en la base de datos.');
          }

          console.log("Resultado de la consulta:", result.rows);
          if (result.rows.length > 0) {
            const formattedResult = result.rows.map(row => `ID: ${row.id}, Fórmula: ${row.formula}`).join('\n');
            ctx.reply(formattedResult);  // Responder con un formato más amigable
          } else {
            ctx.reply('No se encontraron resultados para tu consulta.');
          }
        });

      } else {
        console.log("No se obtuvo respuesta de OpenAI");
        ctx.reply('Lo siento, no pude obtener una respuesta válida de la IA.');
      }

    } catch (error) {
      // Manejo de errores en la API de OpenAI
      console.error('Error en la API de OpenAI:', error.response ? error.response.data : error.message);
      ctx.reply('Ocurrió un error al procesar tu consulta con la IA.');
    }
  }
};

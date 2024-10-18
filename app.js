
import bot from './src/bot.js';  // Importa el bot

// Inicia el bot
bot.launch()
  .then(() => console.log('Bot en funcionamiento...'))
  .catch((err) => console.error('Error al lanzar el bot:', err));

// Manejo adecuado de la finalización del proceso
process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));

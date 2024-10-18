// Importa las dependencias usando `import`
import { Telegraf } from 'telegraf';
import fs from 'fs';
import dotenv from 'dotenv';

// Carga las variables de entorno desde el archivo .env
dotenv.config();

// Inicializa el bot de Telegraf con el token
const bot = new Telegraf(process.env.BOT_TOKEN);

// Cargar automáticamente todos los archivos de comandos desde la carpeta `commands`
const commandFiles = fs.readdirSync('./src/commands').filter(file => file.endsWith('.js'));

for (const file of commandFiles) {
  try {
    const { default: command } = await import(`./commands/${file}`);
    if (command && command.name && command.execute) {
      bot.command(command.name, command.execute);
    } else {
      console.error(`El archivo ${file} no exporta un comando válido.`);
    }
  } catch (error) {
    console.error(`Error al cargar el archivo ${file}:`, error);
  }
}

// Cargar automáticamente todos los archivos de la carpeta `psql`, excepto `db.js`
const psqlFiles = fs.readdirSync('./src/psql').filter(file => file.endsWith('.js') && file !== 'db.js');

for (const file of psqlFiles) {
  try {
    await import(`./psql/${file}`);
    console.log(`Archivo ${file} de la carpeta 'psql' importado con éxito.`);
  } catch (error) {
    console.error(`Error al cargar el archivo ${file} de la carpeta 'psql':`, error);
  }
}

// Cargar automáticamente los middlewares desde la carpeta `middle`
const middleFiles = fs.readdirSync('./src/middle').filter(file => file.endsWith('.js'));

for (const file of middleFiles) {
  try {
    const { default: middle } = await import(`./middle/${file}`);
    bot.use(middle);  // Aplica el middleware
    console.log(`Middleware ${file} aplicado con éxito.`);
  } catch (error) {
    console.error(`Error al cargar el middleware ${file}:`, error);
  }
}

// Exporta el bot
export default bot;

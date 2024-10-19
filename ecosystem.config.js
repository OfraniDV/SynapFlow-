module.exports = {
    apps: [
      {
        name: 'synap',
        script: '/root/SynapFlow-/bot.py',  // Ruta de tu archivo bot.py
        interpreter: '/root/SynapFlow-/venv/bin/python3',  // Ruta de tu Python dentro del entorno virtual
        watch: true,  // Para reiniciar el bot automáticamente si hay cambios
        autorestart: true,
        env: {
          NODE_ENV: 'production'
        },
        time: true // Para mostrar la fecha y hora de logs en pm2
      }
    ]
  };
  
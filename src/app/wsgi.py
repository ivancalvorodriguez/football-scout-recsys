"""Punto de entrada WSGI: `app` listo para un servidor de producción.

    waitress-serve --listen=127.0.0.1:8000 --threads=8 src.app.wsgi:app
    gunicorn -w 1 --threads 8 src.app.wsgi:app        # -w 1, ver más abajo

Existe porque `python -m src.app` construye la app a partir de argumentos de la
línea de comandos, y un servidor WSGI no pasa argumentos: importa un módulo y
busca un objeto. Aquí la configuración llega por **variables de entorno**, que es
lo que un gestor de servicios (systemd, un contenedor, el Programador de tareas)
sabe fijar:

- `SCOUTING_MODELO`   — directorio de artefactos (por defecto, `outputs/modelo/`).
- `SCOUTING_BD`       — base de datos de la extracción (opcional).
- `SCOUTING_USUARIOS` — registro de cuentas (por defecto, `outputs/usuarios.json`).
- `SCOUTING_SECRET_KEY` — clave de firma de la sesión. **Obligatoria aquí**.
- `SCOUTING_LOG`      — nivel de log (INFO por defecto).
- `SCOUTING_COLA`     — URL de redis donde encolar las tareas largas de `/datos`.
  Sin ella, las ejecuta este mismo proceso (ver `src/app/cola.py`).

Se crea en modo producción, así que sin `SCOUTING_SECRET_KEY` el import falla con
un mensaje que explica por qué. Fallar al arrancar es lo correcto: la alternativa
es un servidor en pie con sesiones que se invalidan solas.

**gunicorn con más de un worker NO está soportado.** Los catálogos viven en
memoria del proceso, y sin `SCOUTING_COLA` el estado de las tareas también: con
varios procesos, «una tarea a la vez» deja de cumplirse y el sondeo de
`/datos/tarea` devuelve null cuando cae en otro worker. Lo soportado es UN
proceso con varios HILOS (waitress `--threads`, o `gunicorn -w 1 --threads N`).
Con `SCOUTING_COLA` el estado de las tareas SÍ es compartido, pero los catálogos
siguen sin serlo, así que la regla no cambia. El detalle está en
`docs/app_web.md`.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config
from .factoria import crear_app


def _ruta(variable: str, defecto: Path) -> Path:
    valor = os.environ.get(variable, "").strip()
    return Path(valor) if valor else defecto


app = crear_app(
    _ruta(config.ENV_MODELO, config.Config.model_dir),
    db_path=_ruta(config.ENV_BD, config.Config.db_path),
    ruta_usuarios=_ruta(config.ENV_USUARIOS, config.Config.ruta_usuarios),
    produccion=True,
)

#!/bin/sh
#
# Arranque del worker de tareas dentro del contenedor.
#
# El worker es el que EJECUTA lo que la app encola: `python -m
# src.incremental.ingesta` y `... .reentrenar`. Comparte imagen con la app (por
# eso el Dockerfile copia `src/` entero) y comparte el volumen `outputs/`, que es
# por donde viajan de verdad la base de datos y los artefactos; por redis sólo
# viaja el estado de las tareas.
#
# Cualquier argumento que se le pase se añade al final, así que
# `docker compose run --rm worker --log DEBUG` sigue funcionando.

set -eu

COLA="${SCOUTING_COLA:-}"
MARCA="${SCOUTING_MARCA_TAREA:-outputs/tarea_en_curso.json}"

# Sin cola no hay nada que consumir, y un worker en pie que no consume es peor
# que uno que no arranca: la interfaz aceptaría tareas y no pasaría nada.
if [ -z "$COLA" ]; then
    echo "Error: falta SCOUTING_COLA (URL de redis). El worker no tiene de dónde" >&2
    echo "       sacar trabajo; revisa el docker-compose.yml." >&2
    exit 2
fi

# Los mismos directorios que prepara la app. El worker puede ganarle el arranque
# y es el que primero va a escribir dentro.
mkdir -p outputs/db outputs/modelo

set -- --cola "$COLA" --marca "$MARCA" "$@"

if [ -n "${SCOUTING_LOG:-}" ]; then
    set -- --log "$SCOUTING_LOG" "$@"
fi

exec python -m src.app.worker "$@"

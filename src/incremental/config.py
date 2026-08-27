"""Constantes del flujo incremental (rutas, nombres de fichero, tolerancias).

Sigue el estilo del resto del proyecto: todo lo ajustable vive aqui y los
defaults compartidos se toman de los config ya existentes, para que la ingesta,
el reentrenamiento y los CLI de `src.similitud` miren siempre a los mismos
sitios.
"""

from __future__ import annotations

from pathlib import Path

from src.similitud.config import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_DIR,
    DISTANCIA_SERVIBLE,
    HIPERPARAMETROS_SERVIBLES,
    MODELOS_SERVIBLES,
    hiperparametros_servibles,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_MODEL_DIR",
    "DISTANCIA_SERVIBLE",
    "HIPERPARAMETROS_SERVIBLES",
    "MODELOS_SERVIBLES",
    "hiperparametros_servibles",
    "DIR_COPIAS",
    "NOMBRE_INFORME",
    "FICHEROS_PAQUETE",
]

# --- Ingesta -----------------------------------------------------------------
# Copias de seguridad de la BD antes de escribir. Se guardan fuera de outputs/db
# para que un `--db outputs/db/scouting.db` no acabe leyendo su propia copia.
DIR_COPIAS = Path("outputs/_backup")

# Informe que deja la ingesta (lo consume el reentrenamiento y el humano).
NOMBRE_INFORME = "ingesta.json"

# Estructura esperada de un paquete de partidos. Es la definicion operativa del
# FORMATO: si esto cambia, cambia `docs/incremental.md`.
FICHEROS_PAQUETE = {
    "competiciones": "competitions.json",
    "partidos": "matches/<competition_id>/<season_id>.json",
    "eventos": "events/<match_id>.json",
    "alineaciones": "lineups/<match_id>.json",
}

# El nombre del fichero de estado warm y su ubicacion los decide
# `src.similitud.warm` (es quien define ese formato), no este modulo.

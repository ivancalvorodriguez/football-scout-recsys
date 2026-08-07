"""Constantes y configuración de la app web.

Sigue el estilo del resto del proyecto (`src/extraccion/config.py`,
`src/similitud/config.py`): todo lo ajustable vive aquí, no repartido por las
vistas. Los defaults de modelo se toman de `src.similitud.config` para que la app
y los CLI miren siempre al mismo directorio de artefactos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.similitud import config as config_similitud

# --- Universo de modelos servibles -------------------------------------------
# Mismos valores que aceptan los CLI de `src.similitud`; la app no inventa
# ninguno: solo sirve lo que exista en disco dentro de estas combinaciones.
ENTIDADES: tuple[str, ...] = ("jugador", "equipo")
FORMULACIONES: tuple[str, ...] = ("2", "5")
NORMALIZACIONES: tuple[str, ...] = ("por_liga", "global")

# Etiquetas para la interfaz (el resto del proyecto habla en español).
ETIQUETA_ENTIDAD = {"jugador": "Jugador", "equipo": "Equipo"}
ETIQUETA_PLURAL = {"jugador": "jugadores", "equipo": "equipos"}
ETIQUETA_FORMULACION = {
    "2": "Formulación 2 — SLIM instancia-instancia",
    "5": "Formulación 5 — distribucional (MMD/Sinkhorn) + EASE",
}
ETIQUETA_NORMALIZACION = {
    "por_liga": "z-score por liga",
    "global": "z-score global",
}

# Contra qué se mide el z-score que enseña la interfaz. DEPENDE de la
# normalización del modelo servido: con `por_liga` la media es la de su
# (competition_id, season_id) y con `global` la de todo el dataset, mezclando
# ligas (ver `src.similitud.features.construir`). Escribir «su liga» siempre
# haría que la interfaz mintiese en la mitad de los modelos, así que el texto
# sale de aquí. Redactado para encajar detrás de «frente a», «respecto a» o
# «se separa de».
REFERENCIA_Z = {
    "por_liga": "la media de su liga y temporada",
    "global": "la media global (todas las ligas)",
}
# Sin modelo elegido (páginas de error) no se sabe cuál de las dos es: se dice
# lo único cierto en ambos casos.
REFERENCIA_Z_DEFECTO = "la media de referencia del modelo"

# --- Preferencias de selección -----------------------------------------------
# Orden de preferencia al elegir modelo cuando el usuario no lo especifica. La 5
# va primera solo por ser la más rápida de servir y la que cubre a todas las
# entidades (la 2 deja fuera a las poco conectadas, ver `modelo.top_k`); NO es un
# juicio sobre cuál formulación es mejor — el proyecto las compara, no elige.
PREFERENCIA_FORMULACION: tuple[str, ...] = ("5", "2")
PREFERENCIA_NORMALIZACION: tuple[str, ...] = ("por_liga", "global")

# --- Parámetros de consulta ---------------------------------------------------
TOP_K_DEFECTO = config_similitud.DEFAULT_TOP_K   # 10
TOP_K_MAX = 50
# Features que más explican cada recomendación (0 las desactiva).
N_COINCIDENCIAS = 4
# Métricas destacadas y flojas que muestra la ficha de cada entidad, por lista.
N_RASGOS_FICHA = 4
# Posiciones con reparto de minutos que se listan junto al campo de la ficha.
N_POSICIONES_FICHA = 5
# Sugerencias devueltas por el autocompletado.
MAX_SUGERENCIAS = 12
# Candidatos listados cuando un nombre es ambiguo.
MAX_CANDIDATOS_AMBIGUOS = 25


@dataclass(frozen=True)
class Config:
    """Configuración de una instancia de la app.

    `model_dir` es lo único imprescindible: los artefactos servibles ya traen
    nombres y ligas de cada entidad. `db_path` es opcional y se usa (en lectura)
    para tres adornos: traducir la clave de liga `"11-27"` a `"La Liga
    2015/2016"`, poner el equipo junto al nombre del jugador y dibujar su reparto
    de minutos por posición. Si la BD no está, la app sirve igual sin esos
    bloques.
    """

    model_dir: Path = config_similitud.DEFAULT_MODEL_DIR
    db_path: Path = config_similitud.DEFAULT_DB_PATH
    top_k_defecto: int = TOP_K_DEFECTO
    top_k_max: int = TOP_K_MAX
    n_coincidencias: int = N_COINCIDENCIAS
    n_rasgos_ficha: int = N_RASGOS_FICHA
    max_sugerencias: int = MAX_SUGERENCIAS

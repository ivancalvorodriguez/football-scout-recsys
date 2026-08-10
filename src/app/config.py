"""Constantes y configuración de la app web.

Sigue el estilo del resto del proyecto (`src/extraccion/config.py`,
`src/similitud/config.py`): todo lo ajustable vive aquí, no repartido por las
vistas. Los defaults de modelo se toman de `src.similitud.config` para que la app
y los CLI miren siempre al mismo directorio de artefactos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.extraccion.config import DATA_ROOT
from src.similitud import config as config_similitud

# Raíz del repositorio: `src/app/config.py` -> subir tres niveles. Desde ahí se
# lanzan las tareas de la sección «Datos», igual que se lanzarían a mano.
RAIZ_REPO = Path(__file__).resolve().parents[2]

# Carpeta que propone el formulario de «Añadir partidos». El dataset vendido con
# el proyecto ya tiene el formato de paquete que se espera, así que sirve de
# ejemplo listo para usar.
PAQUETE_DEFECTO = str(DATA_ROOT)

# --- Universo de modelos servibles -------------------------------------------
ENTIDADES: tuple[str, ...] = ("jugador", "equipo")

# Con qué (formulación, normalización) se sirve cada entidad. Lo decide
# `src.similitud.config`, que es donde vive esa elección para todo el proyecto:
# la app no ofrece las cuatro combinaciones por entidad que sabe construir
# `build`, sino UNA por entidad. Lo que el usuario elige aquí es el MODELO
# (el de fábrica o uno reentrenado por él), no la formulación.
MODELO_BASE: dict[str, tuple[str, str]] = dict(config_similitud.MODELOS_SERVIBLES)

# --- Modelos y conjuntos de datos con nombre ----------------------------------
# Dos escalones con la misma forma: lo de fábrica vive en la raíz de su
# directorio y lo que crea el usuario, en una subcarpeta con su nombre.
#
# - Modelos (`catalogo`): `outputs/modelo/` y `outputs/modelo/variantes/<slug>/`.
# - Conjuntos de datos (`conjuntos`): `outputs/db/scouting.db` y
#   `outputs/db/conjuntos/<slug>/scouting.db`.
#
# El slug `base` y la etiqueta valen para los dos (no se cruzan: cada uno se
# resuelve dentro de su catálogo).
VARIANTE_BASE = "base"
NOMBRE_BASE = "Base"
SUBDIR_VARIANTES = "variantes"
FICHERO_VARIANTE = "variante.json"
SUBDIR_CONJUNTOS = "conjuntos"
FICHERO_CONJUNTO = "conjunto.json"
MAX_LARGO_NOMBRE = 60

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

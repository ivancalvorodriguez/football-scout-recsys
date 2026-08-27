"""Constantes y hiperparametros del modelo de similitud.

Todo el pipeline lee de aqui: rutas, listas de features crudas por entidad,
definiciones de features derivadas (ratios/diferencias) e hiperparametros de las
dos formulaciones SLIM. Sigue el estilo de `src/extraccion/config.py`.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from src.extraccion.config import POSITIONS_25, position_slug

# --- Rutas por defecto -------------------------------------------------------
DEFAULT_DB_PATH = Path("outputs/db/scouting.db")
DEFAULT_MODEL_DIR = Path("outputs/modelo")

# --- Modelos servibles: uno por tipo de entidad ------------------------------
# (formulacion, normalizacion) con la que se SIRVE cada entidad. `build` sigue
# generando las cuatro combinaciones de cada entidad —comparar las dos
# formulaciones y las dos normalizaciones es el eje experimental del TFG y eso no
# cambia—, pero la app y el reentrenamiento desde la interfaz trabajan solo con
# estas dos: el usuario elige entre MODELOS (base, reentrenados), no entre
# variantes metodologicas que no puede juzgar desde el navegador.
#
# La eleccion sale de la evaluacion (`outputs/evaluacion/`): son las mejores
# combinaciones del score compuesto de los barridos `jugador_v3` y `equipo_v3`
# ENTRE LAS QUE NO USAN MAHALANOBIS (jugador v87, equipo v349), con la fase 7
# (generalizacion) dentro del score. Las dos son F5 distribucional: cubre a todas
# las entidades (la F2 deja fuera a las poco conectadas, ver `modelo.top_k`) y es
# la unica que se sabe proyectar con fidelidad `FIEL`
# (`foldin.METODOS_PROYECTABLES`). Las dos entidades NO comparten normalizacion
# —jugador `por_liga`, equipo `global`—: lo eligio la medida, no la simetria, y la
# interfaz la rotula en cada respuesta (`config.REFERENCIA_Z` de la app).
MODELOS_SERVIBLES: dict[str, tuple[str, str]] = {
    "jugador": ("5", "por_liga"),
    "equipo": ("5", "global"),
}

# Hiperparametros con los que se ajusto CADA artefacto servido (linaje completo
# en `outputs/modelo/produccion.json`). Es una tabla aparte y no los defaults del
# modulo porque las dos entidades comparten formulacion con VALORES DISTINTOS:
# jugador y equipo salen de combinaciones distintas del barrido, asi que un solo
# `F5_RFF_DIM` no puede reproducir los dos. Se aplica al construir o reentrenar
# la celda servible de una entidad (ver `hiperparametros_servibles` y su uso en
# `build.main` y en `src.incremental.reentrenar`), que es lo que hace que el
# boton «Entrenar» de /datos genere modelos de la MISMA familia que los servidos.
#
# Lo que NO toca es el barrido: ahi los hiperparametros los fija cada combinacion
# (`src/evaluacion/barrido.py`), que es justamente lo que se esta comparando.
HIPERPARAMETROS_SERVIBLES: dict[str, dict[str, object]] = {
    # jugador v91 de outputs/evaluacion/jugador_v3. `F5_EASE_LAMBDA` = 0 es
    # deliberado: es el valor de la combinacion promovida, y significa EASE SIN
    # termino ridge (la forma cerrada invierte S^T S a secas, ver `slim.ease`).
    # Con 2.640 jugadores esa inversa existe y el ajuste es estable, pero es el
    # borde de la rejilla: si al ampliar el catalogo la matriz se acerca a
    # singular, hay que volver a un lambda > 0.
    "jugador": {
        "F5_METODO_JUGADOR": "mmd",
        "F5_RFF_DIM": 1536,
        "F5_EASE_LAMBDA": 0.0,
    },
    # equipo v349 de outputs/evaluacion/equipo_v3
    "equipo": {
        "F5_METODO_EQUIPO": "mmd",
        "F5_RFF_DIM": 1216,
        "F5_EASE_LAMBDA": 0.03,
    },
}


# Distancia entre observaciones con la que se sirve. No esta en
# `MODELOS_SERVIBLES` porque no es una eleccion por entidad: las dos entidades se
# sirven con la MISMA geometria. Desde la promocion del 25-8-2026 es `manhattan`
# (suma de diferencias absolutas), que es la que encabeza el score compuesto en
# las dos entidades ENTRE LAS GEOMETRIAS NO BLANQUEADAS.
#
# Por que no mahalanobis, que puntuaba mas alto: su blanqueo reescala todas las
# direcciones y DESHACE la ponderacion del bloque de posicion
# (`POSITION_SCALING`, mas abajo), medido: el bloque pasa del 2 % al 33 % de la
# distancia^2. Es una interaccion sin resolver, asi que esa geometria no se sirve
# y sus artefactos no se conservan.
#
# `manhattan` no tiene ese problema: su `transformar` es la identidad, de modo que
# el `/sqrt(25)` del bloque de posicion sigue valiendo exactamente lo que dice
# valer. Lo que si tiene es coste: en la F2 el kNN pierde la identidad
# `||a||^2+||b||^2-2ab` y pasa a O(M^2 d) fila a fila (horas sobre las ~50.000
# observaciones de jugador). No afecta a lo que sirve la app, que es F5
# (submuestra + RFF con kernel laplaciano), pero si a un `build --formulacion 2
# --distancia manhattan`.
DISTANCIA_SERVIBLE = "manhattan"


def es_servible(entidad: str, formulacion: str, normalizacion: str,
                distancia: str = DISTANCIA_SERVIBLE) -> bool:
    """¿Es esa celda la que la app sirve para esa entidad?"""
    return (
        MODELOS_SERVIBLES.get(entidad) == (formulacion, normalizacion)
        and distancia == DISTANCIA_SERVIBLE
    )


@contextmanager
def hiperparametros_servibles(entidad: str, formulacion: str, normalizacion: str,
                              distancia: str = DISTANCIA_SERVIBLE):
    """Fija los hiperparametros del artefacto servido, si la celda es la servible.

    Todo el pipeline lee `config.X` en tiempo de llamada, asi que basta con
    ponerlos antes de ajustar y restaurarlos despues (misma mecanica que
    `src.evaluacion.trabajo.config_temporal`, sin depender de el: esto vive en
    `similitud` y aquello en `evaluacion`).

    Cede el diccionario de lo que ha puesto, vacio cuando no es la celda servible
    o la entidad no tiene fila en `HIPERPARAMETROS_SERVIBLES`: asi se puede
    envolver cualquier construccion sin comprobar antes, y quien lo use puede
    decir en el log si esa celda lleva valores propios.
    """
    valores = (
        HIPERPARAMETROS_SERVIBLES.get(entidad, {})
        if es_servible(entidad, formulacion, normalizacion, distancia)
        else {}
    )
    previos = {attr: globals()[attr] for attr in valores}
    globals().update(valores)
    try:
        yield dict(valores)
    finally:
        globals().update(previos)


# --- Posicion como feature del jugador (one-hot ponderado por % de minutos) ---
# Catalogo canonico de las 25 posiciones StatsBomb -> slugs de columna, en orden
# estable de position_id (1..25). Es la unica fuente de verdad del catalogo
# (src.extraccion.config), para que la dimension y el orden del vector no
# dependan de que posiciones aparezcan en un subconjunto de datos.
POSITION_FEATURES: list[str] = [position_slug(n) for n in POSITIONS_25.values()]

# Interruptor: cuando esta activo, la capa de features añade al vector de cada
# jugador-partido las 25 columnas `pos_*` con la FRACCION de minutos disputada en
# cada posicion (one-hot "blando": suma 1 salvo redondeo; one-hot puro si no
# cambia de posicion). Solo aplica a 'jugador' (el equipo no tiene posicion).
#
# ENCENDIDO por defecto a partir del barrido de `outputs/evaluacion/barrido`:
# frente al vector sin posicion gana tambien en las señales NO circulares (F5
# jugador: auto-similitud MRR 0.192 -> 0.281, estabilidad RBO@10 0.377 -> 0.452,
# medido con POSITION_SCALING = "zscore"). Forma parte del modelo base, asi que
# el barrido de hiperparametros ya NO lo mueve: para probar el vector sin
# posicion se pone aqui a False. [[modelo-similitud-posicion]]
USE_POSITION_FEATURES = True

# Como entra ese bloque en el vector. La unidad de referencia es lo que aporta
# cada feature a la distancia^2 entre dos observaciones, que es lo que consumen
# MMD/Sinkhorn (F5) y el coordinate descent (F2): una feature z-scoreada
# (varianza 1) aporta E[(xi-xj)^2] = 2.
#
# - "zscore_sqrt" (por defecto): las 25 columnas se estandarizan y winsorizan
#   como el resto y DESPUES el bloque entero se divide por sqrt(25). El z-score
#   se conserva (cada columna sigue centrada y con varianza 1 antes de dividir),
#   pero el bloque aporta 2*25/25 = 2.0 a la distancia^2, o sea pesa como UNA
#   feature: la posicion informa sin gobernar el ranking.
# - "zscore": el mismo z-score SIN dividir, con lo que el bloque pesa como ~25
#   features y la posicion domina la distancia. Es el modo con el que se midio el
#   barrido (ganaba a no usar posicion tambien en las señales no circulares),
#   pero conviene saber lo que implica: parte de esa ventaja es que la posicion
#   es una etiqueta casi constante entre las dos mitades de la temporada, y
#   ademas una posicion rara (p. ej. 'Center Midfield', 10 apariciones en toda la
#   BD) genera z-scores enormes recortados a F_CLIP_Z.
# - "cruda": la fraccion entra TAL CUAL, en [0,1], FUERA del z-score y del
#   winsorizado. Dos filas one-hot distintas distan sqrt(2) en el bloque, o sea
#   aportan 2.0 a la distancia^2: tambien una feature, pero sin estandarizar (y
#   aqui SI sobraria dividir por sqrt(P): dejaria el bloque P veces por debajo de
#   una feature y la posicion seria casi invisible).
#
# Sea cual sea el modo, la pureza posicional (Fase 0) y el k-NN por posicion
# (Fase 5) usan la posicion como ETIQUETA: con el bloque encendido son
# circulares y no valen como evidencia.
#
# OJO con la geometria: esta ponderacion vale tal cual en las distancias cuyo
# `transformar` NO reescala las columnas (euclidea y manhattan, la servida). Con
# `mahalanobis` NO vale: el blanqueo reescala todas las direcciones y el /sqrt(25)
# se deshace —medido sobre la BD real, el bloque pasa del 2 % al 33 % de la
# distancia^2, casi lo mismo que si no se dividiera (36 %)—. Es consecuencia de
# que la distancia de Mahalanobis sea invariante a cambios lineales de las
# features, asi que ningun valor de este atributo la cambia. Por eso esa geometria
# no se sirve.
POSITION_SCALING = "zscore_sqrt"

# --- Conteos crudos que se convierten a per-90 (jugador) ---------------------
# Se estandarizan como VOLUMEN (una senal distinta de la EFICIENCIA/ratio).
PLAYER_COUNT_FEATURES: list[str] = [
    "passes", "passes_completed", "progressive_passes", "progressive_carries",
    "passes_into_final_third", "xt", "xa", "passes_into_penalty_area",
    "deep_completions", "sca", "shots", "np_shots", "goals", "np_goals",
    "shots_on_target", "xg", "npxg", "touches_in_att_pen_area",
    "take_ons", "take_ons_won", "fouls_drawn", "dispossessed",
    "tackles", "tackles_won", "dribbled_past", "duels_total", "duels_won",
    "aerial_won", "aerial_lost", "aerial_won_off", "aerial_won_def",
    "interceptions", "ball_recoveries", "padj_def_actions",
    "blocks", "clearances", "pressures", "pressure_regains", "counterpressures",
]

# --- Conteos crudos del equipo (se usan por-partido, sin per-90) -------------
# El equipo juega siempre ~un partido completo, asi que el conteo por-partido ya
# es comparable entre equipos (no hay `minutes_played` de equipo en la BD).
TEAM_COUNT_FEATURES: list[str] = [
    "passes", "passes_completed", "ten_plus_pass_sequences", "progressive_passes",
    "passes_into_final_third", "xt", "sca", "gca", "passes_into_penalty_area",
    "open_play_xg", "build_up_attacks", "shots", "np_shots", "goals",
    "shots_on_target", "xg", "npxg", "high_turnovers", "pressures",
    "counterpressures", "direct_attacks", "open_play_sequences",
]

# --- Metricas del equipo que YA son tasas/medias (se usan tal cual) ----------
TEAM_RATE_FEATURES: list[str] = [
    "possession_pct", "ppda", "field_tilt", "absolute_width",
    "sequence_start_distance", "direct_speed", "passes_per_sequence",
]

# --- Features derivadas: ratio = exito / intentos ----------------------------
# (nombre_derivado, numerador, denominador). NaN si el denominador es 0; se
# gestiona en la capa de features (z-score con nanmean/nanstd y relleno a 0).
PLAYER_RATIO_FEATURES: list[tuple[str, str, str]] = [
    ("pass_completion_pct", "passes_completed", "passes"),
    ("take_ons_pct", "take_ons_won", "take_ons"),
    ("aerial_won_pct", "aerial_won", "__aerials_total__"),   # aerial_won+aerial_lost
    ("tackle_pct", "tackles_won", "__tackle_duels__"),       # tackles_won+dribbled_past
    ("duels_won_pct", "duels_won", "duels_total"),
    ("sot_pct", "shots_on_target", "np_shots"),
    ("npxg_per_shot", "npxg", "np_shots"),
]

TEAM_RATIO_FEATURES: list[tuple[str, str, str]] = [
    ("pass_completion_pct", "passes_completed", "passes"),
    ("xg_per_shot", "xg", "shots"),
    ("sot_pct", "shots_on_target", "np_shots"),
]

# Diferencias tipo (rendimiento - modelo). Se calculan sobre valores per-90.
# (nombre_derivado, minuendo, sustraendo)
PLAYER_DIFF_FEATURES: list[tuple[str, str, str]] = [
    ("np_goals_minus_npxg", "np_goals", "npxg"),
]
TEAM_DIFF_FEATURES: list[tuple[str, str, str]] = [
    ("goals_minus_xg", "goals", "xg"),
]

# --- Hiperparametros Formulacion 2 (SLIM instancia-instancia) ----------------
# fsSLIM: cada observacion se reconstruye solo desde sus vecinas -> W tratable.
#
# Los valores de BETA y L1 son los de la combinacion v183 del barrido
# `outputs/evaluacion/f2_equipo_v2`. La F2 ya NO se sirve —el equipo paso a la F5
# con los modelos de agosto de 2026, ver MODELOS_SERVIBLES—, asi que estos
# defaults ya no reproducen ningun artefacto de produccion: son el mejor punto
# conocido de la F2 y lo que construye `build --formulacion 2`, que sigue siendo
# el brazo de comparacion del TFG.
F2_N_NEIGHBORS = 100
F2_BETA = 13.8       # coeficiente L2 (ridge) del objetivo SLIM   [v183]
F2_L1 = 0.48         # coeficiente L1 (sparsity) del objetivo SLIM [v183]
F2_MAX_ITER = 200
F2_TOL = 1e-4

# --- Hiperparametros Formulacion 5 (distribucional + EASE) -------------------
# Etapa 1 (similitud distribucional). Metodo por tipo de entidad.
F5_METODO_JUGADOR = "mmd"      # kernel mean embedding via Random Fourier Features
F5_METODO_EQUIPO = "sinkhorn"  # OT entropico exacto (P pequeno -> barato)
F5_RFF_DIM = 1152              # dimension del embedding RFF para MMD    [v83]
F5_RFF_SEED = 20240714
F5_SINKHORN_REG = 0.5          # regularizacion entropica (log-domain)
F5_SINKHORN_ITERS = 100
# Etapa 2 (EASE de re-ranking sobre la matriz de similitud entidad-entidad).
# 0.0 seria servir la S distribucional tal cual; el barrido v3 encuentra el
# optimo con re-ranking suave.
F5_EASE_LAMBDA = 0.045         # [v83]

# Estos DOS defaults eran los del jugador servido hasta la promocion del
# 16-8-2026 (v83 de `outputs/evaluacion/jugador_v3`) y **ya no lo son**: el
# jugador servido es v147 (RFF_DIM 1536, EASE_LAMBDA 0.02) y el equipo v276
# (1152 y 0.0). No se mueven a proposito. Un default entra en la HUELLA de todo
# artefacto construido sin pasar por la tabla (`src.evaluacion.huella`), asi que
# cambiarlo invalidaria de golpe las ~1.600 combinaciones acumuladas en
# `outputs/evaluacion/` sin cambiar ni un numero de ninguna. Lo que reproduce el
# modelo servido es `HIPERPARAMETROS_SERVIBLES`, arriba, que `build` y
# `src.incremental.reentrenar` aplican en la celda servible de cada entidad.
# Lo de aqui es lo que ve todo lo demas: el resto de celdas de `build` y el punto
# de partida del barrido.

# Recorte (winsorizado) de las features estandarizadas. Los per-90 de partidos
# con muy pocos minutos (p. ej. 0.5') generan z-scores extremos (+-30) que
# distorsionan la seleccion de vecinos y el ajuste; se recortan a +-Z. La
# ponderacion por minutos ya atenua estas observaciones, esto es una salvaguarda.
F_CLIP_Z = 6.0

# Numero de recomendaciones por defecto en el script de prueba.
DEFAULT_TOP_K = 10

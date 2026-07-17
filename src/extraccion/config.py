"""Constantes de dominio: geometría de campo, catálogo de posiciones y rejilla xT.

Todas las coordenadas siguen el sistema StatsBomb: campo de 120x80, el equipo
que ejecuta la acción ataca siempre hacia x=120 (portería rival en (120, 40)).
Las unidades son "yardas StatsBomb"; 1 unidad ≈ 0.9144 m.
"""

from __future__ import annotations

from pathlib import Path

# --- Rutas por defecto -------------------------------------------------------
DATA_ROOT = Path("open-data/data")
DEFAULT_DB_PATH = Path("outputs/db/scouting.db")

# --- Geometría del campo (StatsBomb) -----------------------------------------
FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0
GOAL = (120.0, 40.0)          # centro de la portería rival
OWN_GOAL = (0.0, 40.0)        # centro de la portería propia

YARD_TO_M = 0.9144            # 1 unidad StatsBomb ≈ 0.9144 m

FINAL_THIRD_X = 80.0          # inicio del tercio ofensivo
# Área de penalti rival: x in [102, 120], y in [18, 62]
PEN_AREA_X = 102.0
PEN_AREA_Y_MIN = 18.0
PEN_AREA_Y_MAX = 62.0

# Deep completion (Wyscout): pase que termina a <= 20 m de la portería rival.
DEEP_COMPLETION_RADIUS = 20.0 / YARD_TO_M          # ≈ 21.87 unidades
# High turnover (Opta): recuperación a <= 40 m de la portería rival.
HIGH_TURNOVER_X = FIELD_LENGTH - 40.0 / YARD_TO_M  # ≈ 76.26

# Umbral StatsBomb para "progresivo": la acción reduce >= 25% de la distancia
# restante al centro de la portería rival.
PROGRESSIVE_FRACTION = 0.25

# Ventana de counterpressing / regain (StatsBomb): 5 s tras la presión.
REGAIN_WINDOW_SECONDS = 5.0

# Tipos de pase que son balón parado (se excluyen de progresión/creación).
SET_PIECE_PASS_TYPES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}

# Eventos considerados "toque" (para touches in att pen area y field tilt).
TOUCH_TYPES = {
    "Ball Receipt*", "Carry", "Dribble", "Shot", "Clearance", "Miscontrol", "Goal Keeper",
}

# Acciones ofensivas que pueden acreditarse como Shot-Creating Action (criterio
# FBref: las 2 ultimas acciones del equipo antes del tiro). Compartida por
# `player_stats` y `team_stats`: el SCA de equipo debe ser igual a la suma del de
# sus jugadores, invariante que se rompe si cada capa define su propia lista.
SCA_ACTION_TYPES = {"Pass", "Carry", "Dribble", "Foul Won", "Shot"}
# Numero de acciones previas al tiro que se acreditan (criterio FBref).
SCA_MAX_ACTIONS = 2

# Resultados de tiro que cuentan como "a puerta".
SHOT_ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved to Post"}

# Resultados de Duelo (Tackle) que ganan la posesión.
TACKLE_WON_OUTCOMES = {"Won", "Success In Play", "Success Out"}

# Sub-objetos donde StatsBomb puede marcar `aerial_won`: son las acciones con las
# que el ganador de un salto resuelve el balón. No hay evento propio de "aéreo
# ganado" (el perdedor sí lo tiene: Duel / "Aerial Lost").
AERIAL_WON_KEYS = ("pass", "shot", "clearance", "miscontrol")

# --- Catálogo de las 25 posiciones StatsBomb ---------------------------------
# Se conservan sin agrupar (decisión de proyecto); el one-hot se genera al
# construir la matriz de features a partir de `position_name`/`position_id`.
POSITIONS_25: dict[int, str] = {
    1: "Goalkeeper",
    2: "Right Back",
    3: "Right Center Back",
    4: "Center Back",
    5: "Left Center Back",
    6: "Left Back",
    7: "Right Wing Back",
    8: "Left Wing Back",
    9: "Right Defensive Midfield",
    10: "Center Defensive Midfield",
    11: "Left Defensive Midfield",
    12: "Right Midfield",
    13: "Right Center Midfield",
    14: "Center Midfield",
    15: "Left Center Midfield",
    16: "Left Midfield",
    17: "Right Wing",
    18: "Right Attacking Midfield",
    19: "Center Attacking Midfield",
    20: "Left Attacking Midfield",
    21: "Left Wing",
    22: "Right Center Forward",
    23: "Center Forward",
    24: "Left Center Forward",
    25: "Secondary Striker",
}


def position_slug(name: str) -> str:
    """Nombre de posición -> slug para columnas one-hot (p. ej. 'pos_right_back')."""
    return "pos_" + name.lower().replace(" ", "_")


# --- Rejilla Expected Threat (xT) --------------------------------------------
# Superficie pública de Karun Singh (2018), 12 columnas (x) x 8 filas (y).
# Indexada como XT_GRID[y_bin][x_bin]. Fuente: karun.in/blog/expected-threat.html
XT_GRID: list[list[float]] = [
    [0.006383, 0.007796, 0.008449, 0.009777, 0.011263, 0.012483,
     0.014736, 0.017451, 0.021221, 0.027563, 0.034851, 0.043179],
    [0.007501, 0.008786, 0.009424, 0.010595, 0.012147, 0.013845,
     0.016118, 0.018703, 0.024015, 0.029533, 0.040670, 0.043951],
    [0.008880, 0.009777, 0.010013, 0.011105, 0.012692, 0.014291,
     0.016856, 0.019351, 0.024122, 0.028552, 0.054911, 0.064426],
    [0.009411, 0.010827, 0.010165, 0.011324, 0.012626, 0.014846,
     0.016895, 0.019971, 0.023851, 0.035113, 0.108051, 0.257454],
    [0.009411, 0.010827, 0.010165, 0.011324, 0.012626, 0.014846,
     0.016895, 0.019971, 0.023851, 0.035113, 0.108051, 0.257454],
    [0.008880, 0.009777, 0.010013, 0.011105, 0.012692, 0.014291,
     0.016856, 0.019351, 0.024122, 0.028552, 0.054911, 0.064426],
    [0.007501, 0.008786, 0.009424, 0.010595, 0.012147, 0.013845,
     0.016118, 0.018703, 0.024015, 0.029533, 0.040670, 0.043951],
    [0.006383, 0.007796, 0.008449, 0.009777, 0.011263, 0.012483,
     0.014736, 0.017451, 0.021221, 0.027563, 0.034851, 0.043179],
]
XT_COLS = 12
XT_ROWS = 8

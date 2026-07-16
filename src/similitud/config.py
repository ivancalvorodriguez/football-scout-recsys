"""Constantes y hiperparametros del modelo de similitud.

Todo el pipeline lee de aqui: rutas, listas de features crudas por entidad,
definiciones de features derivadas (ratios/diferencias) e hiperparametros de las
dos formulaciones SLIM. Sigue el estilo de `src/extraccion/config.py`.
"""

from __future__ import annotations

from pathlib import Path

# --- Rutas por defecto -------------------------------------------------------
DEFAULT_DB_PATH = Path("outputs/db/scouting.db")
DEFAULT_MODEL_DIR = Path("outputs/modelo")

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
F2_N_NEIGHBORS = 100
F2_BETA = 1.0        # coeficiente L2 (ridge) del objetivo SLIM
F2_L1 = 0.5          # coeficiente L1 (sparsity) del objetivo SLIM
F2_MAX_ITER = 200
F2_TOL = 1e-4

# --- Hiperparametros Formulacion 5 (distribucional + EASE) -------------------
# Etapa 1 (similitud distribucional). Metodo por tipo de entidad.
F5_METODO_JUGADOR = "mmd"      # kernel mean embedding via Random Fourier Features
F5_METODO_EQUIPO = "sinkhorn"  # OT entropico exacto (P pequeno -> barato)
F5_RFF_DIM = 512               # dimension del embedding RFF para MMD
F5_RFF_SEED = 20240714
F5_SINKHORN_REG = 0.5          # regularizacion entropica (log-domain)
F5_SINKHORN_ITERS = 100
# Etapa 2 (EASE de re-ranking sobre la matriz de similitud entidad-entidad).
F5_EASE_LAMBDA = 50.0

# Recorte (winsorizado) de las features estandarizadas. Los per-90 de partidos
# con muy pocos minutos (p. ej. 0.5') generan z-scores extremos (+-30) que
# distorsionan la seleccion de vecinos y el ajuste; se recortan a +-Z. La
# ponderacion por minutos ya atenua estas observaciones, esto es una salvaguarda.
F_CLIP_Z = 6.0

# Numero de recomendaciones por defecto en el script de prueba.
DEFAULT_TOP_K = 10

"""Constantes del harness de evaluacion (umbrales, seeds, rutas, mapeos).

Los umbrales numericos vienen de `docs/Como_evaluar.pdf` (Decroos & Davis 2019,
SoccerMix 2020, Webber et al. 2010, heuristicas del protocolo). NO son constantes
universales: dependen del tamano del pool, criterio de minutos y liga (ver los
caveats del propio documento). Se usan solo para emitir un veredicto orientativo.
"""

from __future__ import annotations

from pathlib import Path

# --- Rutas -------------------------------------------------------------------
DEFAULT_DB_PATH = Path("outputs/db/scouting.db")
DEFAULT_MODEL_DIR = Path("outputs/modelo")
DEFAULT_OUT_DIR = Path("outputs/evaluacion")

# --- Rejilla de modelos a evaluar --------------------------------------------
FORMULACIONES = ("2", "5")
ENTIDADES = ("jugador", "equipo")
NORMALIZACIONES = ("por_liga", "global")

# --- Parametros de evaluacion ------------------------------------------------
K_LISTA = 10                 # longitud del top-k que se evalua/compara
KS_AUTOSIM = (1, 5, 10)      # top-k que se reportan en auto-similitud
KNN_K = 10                   # vecinos para la clasificacion posicional downstream
RBO_P = 0.9                  # persistencia de RBO (top-weighted, Webber et al.)
BOOTSTRAP_B = 100            # remuestreos para estabilidad
MANTEL_PERMUTACIONES = 999   # permutaciones del test de Mantel
N_MANTEL_MAX = 500           # submuestreo de entidades para el test de Mantel
                             # (con P grande las permutaciones sobre el triangulo
                             #  superior completo son inviables; r se estima sobre
                             #  una submuestra fija por seed)
N_KENDALL_MAX = 150          # submuestreo de entidades para la Kendall tau media
                             # por entidad de la Fase 3 (F2 vs F5): es O(n^2) por
                             #  consulta, asi que se acota el nº de entidades
                             #  sobre las que se rankea y promedia (fijo por seed)
N_ESTABILIDAD_MAX: int | None = None
                             # tope opcional de entidades-consulta cuyo RBO se
                             # promedia en la Fase 2. None = sin tope (exacto). El
                             # bootstrap reconstruye la S completa en cada B; con P
                             # muy grande, fijar este tope estima el RBO medio sobre
                             # una submuestra fija por seed (mismo criterio que
                             # N_MANTEL_MAX) y evita O(B x P) rankings.
PERM_PAREADO = 999           # permutaciones del test pareado (denoising F5)
SEED = 20240720

# --- Umbrales de referencia (orientativos, NO absolutos) ---------------------
PUREZA_MIN = 0.80            # pureza posicional del top-1 (Fase 0)
RBO_ESTABLE = 0.70           # RBO@10 medio que indica top-k estable (Fase 2)
RBO_PREOCUPANTE = 0.50       # por debajo es preocupante
MANTEL_R_CONVERGE = 0.60     # r de Mantel que sugiere misma senal (Fase 3)
ASIMETRIA_ALTA = 0.10        # ||S-S^T||/||S|| por encima del cual conviene simetrizar

# --- Mapeo de posiciones StatsBomb a roles gruesos ---------------------------
# La pureza posicional con las ~24 posiciones finas es demasiado estricta (un
# "Left Center Back" y un "Right Center Back" son el mismo rol). Se agrupan en
# cuatro roles: portero (GK), defensa (DEF), medio (MID) y ataque (ATT).
ROL_GRUESO: dict[str, str] = {
    "Goalkeeper": "GK",
    # Defensas (centrales, laterales, carrileros).
    "Right Center Back": "DEF", "Left Center Back": "DEF", "Center Back": "DEF",
    "Right Back": "DEF", "Left Back": "DEF",
    "Right Wing Back": "DEF", "Left Wing Back": "DEF",
    # Medios (defensivos, centrales, ofensivos, de banda).
    "Right Defensive Midfield": "MID", "Left Defensive Midfield": "MID",
    "Center Defensive Midfield": "MID",
    "Right Center Midfield": "MID", "Left Center Midfield": "MID",
    "Center Midfield": "MID",
    "Right Midfield": "MID", "Left Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Right Attacking Midfield": "MID", "Left Attacking Midfield": "MID",
    # Ataque (extremos y delanteros).
    "Right Wing": "ATT", "Left Wing": "ATT",
    "Center Forward": "ATT",
    "Right Center Forward": "ATT", "Left Center Forward": "ATT",
}
ROL_DESCONOCIDO = "???"

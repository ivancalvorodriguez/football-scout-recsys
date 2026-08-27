"""Constantes del harness de evaluacion (umbrales, seeds, rutas, mapeos).

Los umbrales numericos vienen de `docs/Como_evaluar.pdf` (Decroos & Davis 2019,
SoccerMix 2020, Webber et al. 2010, heuristicas del protocolo). NO son constantes
universales: dependen del tamano del pool, criterio de minutos y liga (ver los
caveats del propio documento). Se usan solo para emitir un veredicto orientativo.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from src.similitud import config as scfg

# --- Rutas -------------------------------------------------------------------
DEFAULT_DB_PATH = Path("outputs/db/scouting.db")
DEFAULT_MODEL_DIR = Path("outputs/modelo")
DEFAULT_OUT_DIR = Path("outputs/evaluacion")

# Base de datos de ENTRENAMIENTO/experimentacion: las cinco grandes 2015/2016 mas
# la Indian Super League 2021/2022. Es la que hace posible la Fase 7 —sin una liga
# ajena a las del modelo servido no hay nada que dejar fuera— y la que conviene
# usar en los barridos.
#
# Vive DELIBERADAMENTE fuera de `outputs/db/conjuntos/`, que es el unico sitio que
# la app enumera (`src.app.conjuntos.CatalogoDatos`): un conjunto ahi sin campo
# `usuario` seria compartido con todas las cuentas y aparecerian en el buscador
# 284 jugadores indios que el modelo servido no conoce. Lo que la app sirve es la
# base; esto es material de laboratorio.
#
# **No es el default de nada** a proposito: `evaluar`, `build` y `barrido` siguen
# apuntando a la base, para que un `build` por descuido no genere artefactos sobre
# un universo distinto del que sirve la app (`src.app.cobertura` lo detectaria,
# pero mas vale no llegar ahi). Se pasa a mano con `--db`.
DB_ENTRENAMIENTO = Path("outputs/db/entrenamiento/scouting.db")

# --- Rejilla de modelos a evaluar --------------------------------------------
FORMULACIONES = ("2", "5")
ENTIDADES = ("jugador", "equipo")
NORMALIZACIONES = ("por_liga", "global")

# Distancias entre observaciones (`src.similitud.distancias`). Son el CUARTO eje
# de la rejilla, al mismo nivel que los tres de arriba: cada una da un artefacto
# distinto, una fila propia en las tablas y una figura propia en las superficies
# 3D — no son un hiperparametro que se afine dentro de un modelo.
#
# En el BARRIDO entra por defecto solo la euclidea, a diferencia de las
# normalizaciones (que entran las dos). No es una asimetria caprichosa: comparar
# las dos normalizaciones cuesta el doble y es el eje experimental declarado del
# TFG, mientras que las cuatro distancias cuestan x4 y una de ellas (`manhattan`
# sobre la F2 de jugador) se va a horas. Se piden explicitamente con
# `--distancias`. Ademas, todo lo acumulado en `outputs/evaluacion/` es euclideo,
# asi que un barrido sin la flag significa lo mismo que antes.
#
# En `evaluar` el default es OTRO (`DISTANCIA_EVALUADA`): ahi no se explora una
# rejilla, se miden los artefactos vigentes de `outputs/modelo/`, y desde el
# 25-8-2026 esos son mahalanobis. Con el default del barrido, un `evaluar` sin
# flags no encontraria ningun modelo que medir.
DISTANCIAS = ("euclidea", "mahalanobis", "coseno", "manhattan")
DISTANCIAS_POR_DEFECTO = ("euclidea",)
DISTANCIA_EVALUADA = (scfg.DISTANCIA_SERVIBLE,)

# Metodos de la etapa 1 de la Formulacion 5 (`src.similitud.distributional`). No
# son un eje de la rejilla como los de arriba: la etapa 1 se elige POR ENTIDAD en
# `similitud.config` (F5_METODO_JUGADOR / F5_METODO_EQUIPO) y el barrido puede
# barrerlos con `--metodo-f5`. Estan aqui, y no en `similitud.config`, por el
# mismo motivo que los otros tres: es el catalogo con el que la linea de comandos
# valida lo que se le pide.
METODOS_F5 = ("mmd", "sinkhorn")

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
N_GENERALIZACION_MAX = 400   # tope de entidades por ambito en la Fase 7. Se
                             # aplica a los DOS lados (dentro y fuera) para que
                             # las cifras sean comparables: el top-1 y la
                             # cobertura dependen de cuantas consultas hay.
                             # Ademas acota el coste, que ahi es una proyeccion
                             # por entidad.
BOOTSTRAP_GENERALIZACION = 25
                             # remuestreos por entidad de la estabilidad fuera de
                             # muestra (Fase 7). Mucho menor que BOOTSTRAP_B
                             # porque aqui se remuestrea POR ENTIDAD y no una vez
                             # para todo el modelo; el remuestreo en si es barato
                             # (`foldin.piezas` no rehace el ajuste), lo que se
                             # acota es el numero de rankings.
N_ESTABILIDAD_GENERALIZACION = 60
                             # entidades cuya estabilidad se promedia. Con las 400
                             # del tope serian 400xB rankings por celda del
                             # barrido; con esta submuestra fija por seed la cifra
                             # sigue siendo representativa y la fase no se dispara.
SEED = 20240720

# --- Umbrales de referencia (orientativos, NO absolutos) ---------------------
PUREZA_MIN = 0.80            # pureza posicional del top-1 (Fase 0)
RBO_ESTABLE = 0.70           # RBO@10 medio que indica top-k estable (Fase 2)
RBO_PREOCUPANTE = 0.50       # por debajo es preocupante
MANTEL_R_CONVERGE = 0.60     # r de Mantel que sugiere misma senal (Fase 3)
ASIMETRIA_ALTA = 0.10        # ||S-S^T||/||S|| por encima del cual conviene simetrizar
BRECHA_ACEPTABLE = 0.10      # caida dentro->fuera de muestra (Fase 7) que se
                             # considera «el criterio viaja». Orientativo y de
                             # cosecha propia: el PDF no cubre esta fase, asi que
                             # se fija en el mismo orden que la propia dispersion
                             # de la proyeccion (ver `generalizacion`).

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


@contextmanager
def config_temporal(valores: dict[str, object]):
    """Fija atributos de `src.similitud.config` y los restaura al salir.

    Todo el pipeline lee los valores en tiempo de llamada, asi que basta con
    fijarlos antes de construir/evaluar. Es estado GLOBAL del proceso: dos
    combinaciones no pueden solaparse dentro del mismo interprete, que es una de
    las razones por las que el barrido paralelo usa procesos y no hilos.
    """
    previos: dict[str, object] = {}
    for clave, valor in valores.items():
        previos[clave] = getattr(scfg, clave)
        setattr(scfg, clave, valor)
    try:
        yield
    finally:
        for clave, valor in previos.items():
            setattr(scfg, clave, valor)

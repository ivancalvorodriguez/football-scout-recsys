"""Reconstruccion de la matriz S y modelos "virtuales" que solo existen para evaluar.

El PDF exige evaluar sobre artefactos que NO estan en `outputs/modelo/`:
- Auto-similitud: cada entidad se desdobla en dos "jugadores virtuales" (mitades
  con sus partidos PARES vs IMPARES) y se comprueba que una mitad recupera a la
  otra. Modelos que no existen -> se construyen aqui.
- Estabilidad: remuestreos bootstrap de las observaciones.

Este modulo mantiene la fidelidad al pipeline real: reutiliza EXACTAMENTE las
piezas de `src.similitud` (`slim`, `distributional`), de modo que reconstruir con
el indice de entidades real reproduce el artefacto guardado (se verifica). La
unica pieza cara (la W MxM de la F2, la etapa distribucional de la F5) se calcula
una vez por (entidad, normalizacion) y se reutiliza para todas las variantes de
agrupacion (real, mitades, bootstrap), porque depende solo de las observaciones,
no de como se agrupen en entidades.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from src.similitud import config as scfg
from src.similitud import data as sdata
from src.similitud import distancias, distributional, slim
from src.similitud.distancias import Espacio
from src.similitud.features import MatrizFeatures, construir as construir_features
from src.similitud.modelo import IndiceEntidades, indexar_entidades


# --------------------------------------------------------------------------- #
# Contexto por (entidad, normalizacion)                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Contexto:
    """Todo lo necesario para reconstruir S con cualquier agrupacion de entidades."""

    entidad: str
    normalizacion: str
    mf: MatrizFeatures
    match_id: np.ndarray        # (M,) partido de cada observacion
    minutos_obs: np.ndarray     # (M,) minutos de cada observacion (equipo -> 1)
    idx_real: IndiceEntidades   # agrupacion real (0..P-1)
    # Geometria con la que se mide (`src.similitud.distancias`). Es parte del
    # contexto y no un argumento suelto porque la W cacheada aqui abajo se ajusta
    # EN ella: dos distancias son dos contextos, no uno con un parametro.
    espacio: Espacio = field(
        default_factory=lambda: Espacio(distancias.POR_DEFECTO))
    _W: tuple | None = None     # cache de la W de la F2 (cols_idx, cols_val)

    @property
    def distancia(self) -> str:
        return self.espacio.nombre

    @property
    def metodo_f5(self) -> str:
        return (
            scfg.F5_METODO_JUGADOR
            if self.entidad == "jugador"
            else scfg.F5_METODO_EQUIPO
        )

    def W(self, progreso=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """W instancia-instancia de la F2 (perezosa; depende solo de mf.X).

        ``progreso`` se reenvia a ``slim_instancia`` para seguir el ajuste (solo
        aplica la PRIMERA vez, que es la cara; despues la W queda cacheada).
        """
        if self._W is None:
            self._W = slim.slim_instancia(
                self.mf.X,
                n_neighbors=scfg.F2_N_NEIGHBORS,
                beta=scfg.F2_BETA,
                l1=scfg.F2_L1,
                max_iter=scfg.F2_MAX_ITER,
                tol=scfg.F2_TOL,
                progreso=progreso,
                espacio=self.espacio,
            )
        return self._W


def crear_contexto(
    db_path,
    entidad: str,
    normalizacion: str,
    excluir_ligas: tuple[str, ...] = (),
    estadisticas=None,
    distancia: str = distancias.POR_DEFECTO,
    espacio: Espacio | None = None,
) -> Contexto:
    """Carga la BD y construye el contexto (mf + metadatos alineados por fila).

    ``excluir_ligas`` deja fuera las observaciones de esas competiciones ANTES de
    estandarizar, que es lo que hace de esto un hold-out de verdad: la liga
    excluida no entra en el ajuste y tampoco en las mu/sd del z-score (Fase 7,
    ver `generalizacion`).

    ``estadisticas`` son mu/sd congeladas de otro contexto, para poder poner
    observaciones nuevas en el espacio de un ajuste anterior. Los dos parametros
    son opcionales y por defecto no hacen nada: sin ellos, esto es el contexto de
    siempre sobre la BD entera.

    ``distancia`` es la geometria con la que se reconstruye la S, y tiene que ser
    la MISMA con la que se ajusto el artefacto que se esta evaluando: las fases
    comparan lo que reconstruye este contexto con la S del modelo. ``espacio``
    permite pasar una geometria ya ajustada (la del estado warm) en vez de
    reestimarla de estos datos, que es lo que hace falta cuando el contexto trae
    observaciones que el ajuste no vio (Fase 7).
    """
    df = sdata.cargar(db_path, entidad)
    if excluir_ligas:
        df = df[~df[sdata.LEAGUE_KEY].astype(str).isin(set(excluir_ligas))]
        df = df.reset_index(drop=True)
    mf = construir_features(
        df, entidad, normalizacion=normalizacion, estadisticas=estadisticas)
    match_id = df["match_id"].to_numpy()
    if entidad == "jugador":
        minutos = df["minutes_played"].to_numpy(dtype=float)
    else:
        minutos = np.ones(len(df), dtype=float)
    return Contexto(
        entidad=entidad,
        normalizacion=normalizacion,
        mf=mf,
        match_id=match_id,
        minutos_obs=minutos,
        idx_real=indexar_entidades(mf),
        espacio=(espacio if espacio is not None
                 else distancias.preparar(mf.X, distancia)),
    )


# --------------------------------------------------------------------------- #
# Reconstruccion de S (fiel a formulacion2/5.construir)                        #
# --------------------------------------------------------------------------- #

def reconstruir_S(
    ctx: Contexto,
    formulacion: str,
    row_entity: np.ndarray,
    n_ent: int,
    weight: np.ndarray | None = None,
    progreso=None,
) -> np.ndarray:
    """Reconstruye la S servible para una agrupacion arbitraria de observaciones.

    Espejo exacto de `formulacion{2,5}.construir` pero parametrizado por
    ``row_entity`` (obs -> entidad) y con ``weight`` opcional (para el bootstrap,
    que perturba las masas). Con el ``row_entity`` real reproduce el artefacto.

    ``progreso`` (opcional) se reenvia al bucle interno caro de cada formulacion
    (ajuste SLIM de la F2 / transporte optimo de la F5) para seguir el avance.
    """
    w = ctx.mf.weight if weight is None else weight
    if formulacion == "2":
        cols_idx, cols_val = ctx.W(progreso)
        return slim.agregar_W_a_entidades(cols_idx, cols_val, row_entity, w, n_ent)
    if formulacion == "5":
        S, _ = reconstruir_S_f5(ctx, row_entity, n_ent, weight=weight, progreso=progreso)
        return S
    raise ValueError(f"formulacion desconocida: {formulacion!r}")


def reconstruir_S_f5(
    ctx: Contexto,
    row_entity: np.ndarray,
    n_ent: int,
    weight: np.ndarray | None = None,
    progreso=None,
) -> tuple[np.ndarray, np.ndarray]:
    """F5 devolviendo tambien la S distribucional PRE-EASE (para la Fase 4).

    Devuelve ``(S_post, S_pre)``: S_post = simetrizada de S_ent @ B (servible),
    S_pre = similitud distribucional cruda (sin denoising). ``progreso`` sigue el
    bucle de la etapa distribucional (solo tiene efecto con el metodo sinkhorn).
    """
    mf = ctx.mf if weight is None else replace(ctx.mf, weight=weight)
    S_pre = distributional.construir_S(mf, row_entity, n_ent, ctx.metodo_f5,
                                       progreso=progreso, espacio=ctx.espacio)
    B = slim.ease(S_pre, lam=scfg.F5_EASE_LAMBDA)
    S_post = S_pre @ B
    S_post = 0.5 * (S_post + S_post.T)
    np.fill_diagonal(S_post, 0.0)
    return S_post, S_pre


def reconstruir_S_cruda(ctx: Contexto, formulacion: str) -> np.ndarray:
    """S entidad-entidad CRUDA (direccional, SIN simetrizar) de la agrupacion real.

    El pipeline servible simetriza como ultimo paso (``0.5*(S+S^T)``), asi que la
    asimetria del artefacto guardado es ~0 por construccion y no informa de nada.
    El sanity check de la Fase 0 (PDF, pp. 6/8-9) quiere justo la asimetria ANTES
    de simetrizar: cuanta senal direccional corrige el pipeline. Aqui se
    reconstruye esa matriz cruda reutilizando las mismas piezas de `src.similitud`:

    - F2: la agregacion de W sin el ``0.5*(S+S^T)`` final (``simetrizar=False``).
    - F5: ``S_pre @ B`` (la S distribucional, simetrica, tras EASE) antes de
      simetrizar; la asimetria la introduce el denoising de EASE, no ``S_pre``.
    """
    row = ctx.idx_real.row_entity
    P = len(ctx.idx_real.ids)
    if formulacion == "2":
        cols_idx, cols_val = ctx.W()
        return slim.agregar_W_a_entidades(
            cols_idx, cols_val, row, ctx.mf.weight, P, simetrizar=False)
    if formulacion == "5":
        S_pre = distributional.construir_S(ctx.mf, row, P, ctx.metodo_f5,
                                           espacio=ctx.espacio)
        R = S_pre @ slim.ease(S_pre, lam=scfg.F5_EASE_LAMBDA)
        np.fill_diagonal(R, 0.0)
        return R
    raise ValueError(f"formulacion desconocida: {formulacion!r}")


# --------------------------------------------------------------------------- #
# Modelo virtual: mitades pares/impares por entidad                            #
# --------------------------------------------------------------------------- #

@dataclass
class SplitVirtual:
    """Desdoblamiento de cada entidad en mitades A (pares) y B (impares)."""

    row_entity: np.ndarray        # (M,) obs -> indice de entidad virtual (0..V-1)
    n_virt: int                   # nº de entidades virtuales
    orig_real: np.ndarray         # (V,) indice de entidad REAL de cada virtual
    half: np.ndarray              # (V,) 0=A(pares) 1=B(impares)
    pares_AB: list[tuple[int, int]]   # (vA, vB) de entidades con ambas mitades
    real_evaluables: np.ndarray       # (nº pares,) indice real de cada par


def split_par_impar(ctx: Contexto) -> SplitVirtual:
    """Desdobla cada entidad ordenando sus partidos y alternando par/impar.

    Para cada entidad se ordenan sus observaciones por ``match_id`` y se asigna la
    posicion 0,2,4,... a la mitad A (pares) y 1,3,5,... a la mitad B (impares).
    Garantiza dos mitades no vacias para toda entidad con >=2 partidos; las de 1
    partido quedan solo como mitad A (distractoras en el pool, no evaluables).

    El PDF pide, en paralelo, un split CRONOLOGICO temporada t vs t+1 (Decroos &
    Davis). Aqui NO se hace: la BD es de una unica temporada, asi que t vs t+1 no
    es aplicable y el split intra-temporada pares/impares es la unica variante
    posible (el propio PDF lo propone para jugadores con pocas temporadas).
    """
    row_real = ctx.idx_real.row_entity
    P = len(ctx.idx_real.ids)
    M = len(row_real)
    row_virt = np.full(M, -1, dtype=np.int64)
    orig_real: list[int] = []
    half: list[int] = []
    pares: list[tuple[int, int]] = []
    reales_eval: list[int] = []

    for e in range(P):
        obs = np.where(row_real == e)[0]
        orden = obs[np.argsort(ctx.match_id[obs], kind="mergesort")]
        es_par = np.arange(len(orden)) % 2 == 0
        vA = len(orig_real)
        orig_real.append(e)
        half.append(0)
        row_virt[orden[es_par]] = vA
        if np.any(~es_par):  # tiene al menos un partido impar -> mitad B
            vB = len(orig_real)
            orig_real.append(e)
            half.append(1)
            row_virt[orden[~es_par]] = vB
            pares.append((vA, vB))
            reales_eval.append(e)

    return SplitVirtual(
        row_entity=row_virt,
        n_virt=len(orig_real),
        orig_real=np.array(orig_real, dtype=np.int64),
        half=np.array(half, dtype=np.int64),
        pares_AB=pares,
        real_evaluables=np.array(reales_eval, dtype=np.int64),
    )


def multiplicidades_bootstrap(
    ctx: Contexto, split: SplitVirtual | None, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicidades bootstrap por entidad (remuestreo de partidos con reemplazo).

    Para cada entidad se remuestrean sus n observaciones con reemplazo; el vector
    de multiplicidades resultante se aplica como multiplicador de la masa por
    observacion. Reutiliza la estructura aprendida (W de la F2, mapa RFF de la F5)
    y perturba la agregacion/embedding: mide la estabilidad del top-k ante la
    composicion de partidos sin re-aprender W (O(1) frente a O(coste de ajuste)).
    """
    row = split.row_entity if split is not None else ctx.idx_real.row_entity
    n_ent = split.n_virt if split is not None else len(ctx.idx_real.ids)
    mult = np.zeros(len(row), dtype=float)
    for e in range(n_ent):
        obs = np.where(row == e)[0]
        if obs.size == 0:
            continue
        sel = rng.integers(0, obs.size, size=obs.size)
        cuenta = np.bincount(sel, minlength=obs.size)
        mult[obs] = cuenta
    return mult

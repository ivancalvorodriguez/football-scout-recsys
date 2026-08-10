"""Formulacion 5 — similitud distribucional + EASE (SLIM) de re-ranking.

Dos etapas (segun `docs/como_usar_slim.pdf`):

- Etapa 1: cada entidad es la NUBE de sus observaciones por-partido; se calcula
  una similitud distribucional S_ent (P x P) que usa TODAS las muestras sin
  promediar las features crudas (jugadores: MMD via kernel mean embedding con
  Random Fourier Features; equipos: OT entropico/Sinkhorn exacto). El input nunca
  se colapsa: la unica "agregacion" es el embedding de la distribucion (media de
  un mapa NO LINEAL) o el transporte optimo entre nubes.
- Etapa 2: EASE (SLIM de forma cerrada, Steck 2019) aprende B reconstruyendo
  S_ent -> similitud entidad-entidad APRENDIDA (denoising/propagacion sobre el
  grafo de similitud distribucional). La puntuacion servible es S_ent @ B (fila i
  = recomendaciones para la entidad i).

Diferencia clave con la Formulacion 2: aqui la unidad que SLIM/EASE reconstruye ya
es la ENTIDAD (via S_ent), no el partido; la similitud aprendida se apoya sobre la
geometria fijada por la etapa distribucional.
"""

from __future__ import annotations

import numpy as np

from . import config, distributional, slim, warm
from .features import MatrizFeatures
from .modelo import (
    ModeloSimilitud,
    features_display,
    indexar_entidades,
    ligas_por_entidad,
)
from .warm import EstadoWarm


def _metodo_por_entidad(entidad: str) -> str:
    return (
        config.F5_METODO_JUGADOR if entidad == "jugador" else config.F5_METODO_EQUIPO
    )


def hiperparametros(entidad: str) -> dict:
    """Identidad del ajuste: con otros valores, un estado previo no vale."""
    metodo = _metodo_por_entidad(entidad)
    return {
        "metodo": metodo,
        "ease_lambda": config.F5_EASE_LAMBDA,
        "rff_dim": config.F5_RFF_DIM if metodo == "mmd" else None,
        "rff_seed": config.F5_RFF_SEED if metodo == "mmd" else None,
        "sinkhorn_reg": config.F5_SINKHORN_REG if metodo == "sinkhorn" else None,
        "sinkhorn_iters": config.F5_SINKHORN_ITERS if metodo == "sinkhorn" else None,
        **warm.hiper_features(),
    }


def construir(
    mf: MatrizFeatures, entidad: str, normalizacion: str = "por_liga"
) -> ModeloSimilitud:
    """Entrena la Formulacion 5 en frio y devuelve el modelo servible."""
    modelo, _ = construir_con_estado(mf, entidad, normalizacion)
    return modelo


def construir_con_estado(
    mf: MatrizFeatures,
    entidad: str,
    normalizacion: str = "por_liga",
    previo: EstadoWarm | None = None,
    progreso=None,
    congelar_kernel: bool = True,
) -> tuple[ModeloSimilitud, EstadoWarm]:
    """Entrena la Formulacion 5 y devuelve tambien el estado reutilizable.

    Con ``previo``, la etapa 1 reaprovecha lo que no ha cambiado:

    - ``mmd``: se CONGELA el ancho del kernel RBF. Sale de la mediana de
      distancias de todo el dataset, asi que recalcularlo moveria el embedding de
      todas las entidades, tambien las intactas; congelandolo, las nuevas entran
      en el mismo espacio y las viejas conservan el suyo. El coste de la etapa es
      despreciable: lo que se gana aqui es ESTABILIDAD, no tiempo. Es la unica
      pieza en la que el resultado NO coincide con un ajuste en frio (que
      estimaria otro ancho); ``congelar_kernel=False`` lo recalcula y recupera la
      equivalencia, a cambio de mover a las entidades que no han jugado.
    - ``sinkhorn``: se reutilizan los costes de transporte optimo de los pares en
      que ninguna de las dos nubes ha cambiado (reutilizacion exacta). Solo se
      calculan las filas y columnas de las entidades tocadas, que es donde esta
      el coste cuadratico de la etapa.

    La etapa 2 (EASE) se resuelve siempre entera: es de forma cerrada y su coste
    (una inversa P x P) es de segundos a esta escala, asi que una actualizacion
    incremental por Woodbury seria mas codigo, mas fragil y no mas rapida.
    """
    idx = indexar_entidades(mf)
    n_ent = len(idx.ids)
    M = mf.X.shape[0]
    metodo = _metodo_por_entidad(entidad)

    hiper = hiperparametros(entidad)
    emp = warm.emparejar(previo, mf, idx.ids, hiper)
    reutiliza = previo is not None and emp.hay_reutilizacion

    sigma = (
        previo.sigma
        if (reutiliza and metodo == "mmd" and congelar_kernel)
        else None
    )
    costos_previos = (
        warm.costos_reutilizables(previo, emp, n_ent)
        if reutiliza and metodo == "sinkhorn"
        else None
    )
    pares_reutilizados = (
        int(np.isfinite(np.triu(costos_previos, k=1)).sum())
        if costos_previos is not None
        else 0
    )

    # Etapa 1: similitud distribucional entidad-entidad (sin colapsar input).
    if metodo == "sinkhorn":
        costos = distributional.costos_sinkhorn(
            mf, idx.row_entity, n_ent, progreso=progreso, previos=costos_previos)
        S_ent = distributional.kernel_desde_costos(costos)
    else:
        costos = None
        if sigma is None:
            sigma = distributional.sigma_kernel(mf.X)
        S_ent = distributional.similitud_mmd(mf, idx.row_entity, n_ent, sigma=sigma)

    # Etapa 2: EASE aprende B; la puntuacion servible es S_ent @ B.
    B = slim.ease(S_ent, lam=config.F5_EASE_LAMBDA)
    S = S_ent @ B
    S = 0.5 * (S + S.T)          # simetrizar para un ranking estable
    np.fill_diagonal(S, 0.0)

    meta = {
        "n_observaciones": M,
        "n_entidades": n_ent,
        "metodo_distribucional": metodo,
        "ease_lambda": config.F5_EASE_LAMBDA,
        "rff_dim": config.F5_RFF_DIM if metodo == "mmd" else None,
        "sinkhorn_reg": config.F5_SINKHORN_REG if metodo == "sinkhorn" else None,
        "normalizacion": normalizacion,
        "ligas_por_entidad": ligas_por_entidad(mf, idx),
        "warm": {
            "aplicado": reutiliza,
            "motivo": emp.motivo,
            "sigma_congelado": bool(reutiliza and metodo == "mmd" and congelar_kernel),
            "pares_ot_reutilizados": pares_reutilizados,
            **emp.resumen(),
        },
        "estadisticas_normalizacion": (
            mf.estadisticas.como_dict() if mf.estadisticas is not None else None
        ),
        "descripcion": (
            "Etapa 1 similitud distribucional (%s) sobre las nubes de "
            "observaciones + etapa 2 EASE (SLIM cerrado) de re-ranking." % metodo
        ),
    }
    modelo = ModeloSimilitud(
        formulacion="5",
        entidad=entidad,
        S=S,
        entity_ids=idx.ids,
        entity_names=idx.names,
        feat_names=mf.feat_names,
        feat_display=features_display(mf, idx),
        meta=meta,
    )
    estado = warm.estado_base(mf, "5", entidad, normalizacion, idx.ids, hiper)
    estado.sigma = float(sigma) if sigma is not None else None
    estado.costos = costos
    return modelo, estado

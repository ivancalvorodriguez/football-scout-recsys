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

from . import config, distributional, slim
from .features import MatrizFeatures
from .modelo import (
    ModeloSimilitud,
    features_display,
    indexar_entidades,
    ligas_por_entidad,
)


def _metodo_por_entidad(entidad: str) -> str:
    return (
        config.F5_METODO_JUGADOR if entidad == "jugador" else config.F5_METODO_EQUIPO
    )


def construir(
    mf: MatrizFeatures, entidad: str, normalizacion: str = "por_liga"
) -> ModeloSimilitud:
    """Entrena la Formulacion 5 y devuelve el modelo servible."""
    idx = indexar_entidades(mf)
    n_ent = len(idx.ids)
    M = mf.X.shape[0]
    metodo = _metodo_por_entidad(entidad)

    # Etapa 1: similitud distribucional entidad-entidad (sin colapsar input).
    S_ent = distributional.construir_S(mf, idx.row_entity, n_ent, metodo)

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
        "descripcion": (
            "Etapa 1 similitud distribucional (%s) sobre las nubes de "
            "observaciones + etapa 2 EASE (SLIM cerrado) de re-ranking." % metodo
        ),
    }
    return ModeloSimilitud(
        formulacion="5",
        entidad=entidad,
        S=S,
        entity_ids=idx.ids,
        entity_names=idx.names,
        feat_names=mf.feat_names,
        feat_display=features_display(mf, idx),
        meta=meta,
    )

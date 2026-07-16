"""Formulacion 2 — SLIM instancia-instancia + agregacion de W a nivel entidad.

Es la formulacion FIEL a la letra de la restriccion (segun `docs/como_usar_slim.pdf`):

- Entra la matriz de instancias X (M x d): una fila por observacion jugador/equipo
  -partido, SIN colapsar. M filas entran, M filas participan en el ajuste.
- SLIM aprende W (M x M) reconstruyendo cada OBSERVACION como combinacion dispersa
  no-negativa de las demas (variante feature-space del PDF). `A` = vecinas de la
  observacion (features en filas), `W_rs` = cuanto contribuye r a reconstruir s.
- La similitud entidad->entidad se obtiene AGREGANDO bloques de W (S_pq =
  sum_{r en p, s en q} m_r m_s W_rs / Z), es decir la agregacion cae sobre los
  PESOS APRENDIDOS, nunca sobre las features de entrada.
- Ponderacion por minutos: masas m en la agregacion (opcion (b) del PDF; en la
  variante feature-space las muestras del minimo-cuadrados son las d features, no
  las observaciones, asi que el sample_weight por observacion no aplica en el
  ajuste). Normalizacion por N_p*N_q para que un jugador con muchos partidos no
  domine. Un jugador con 1 sola observacion entra sin caso especial.

Diferencia clave con la Formulacion 5: aqui SLIM optimiza la reconstruccion de
PARTIDOS (la similitud entidad-entidad es un post-proceso de W); en la 5 SLIM/EASE
opera sobre una similitud entidad-entidad ya calculada.
"""

from __future__ import annotations

import numpy as np

from . import config, slim
from .features import MatrizFeatures
from .modelo import (
    ModeloSimilitud,
    features_display,
    indexar_entidades,
    ligas_por_entidad,
)


def construir(
    mf: MatrizFeatures, entidad: str, normalizacion: str = "por_liga"
) -> ModeloSimilitud:
    """Entrena la Formulacion 2 y devuelve el modelo servible."""
    idx = indexar_entidades(mf)
    n_ent = len(idx.ids)
    M = mf.X.shape[0]

    # (3)->(4) SLIM instancia-instancia: W dispersa por columnas (observaciones).
    cols_idx, cols_val = slim.slim_instancia(
        mf.X,
        n_neighbors=config.F2_N_NEIGHBORS,
        beta=config.F2_BETA,
        l1=config.F2_L1,
        max_iter=config.F2_MAX_ITER,
        tol=config.F2_TOL,
    )
    nnz = int(sum(v.size for v in cols_val))

    # Agregacion sobre W (no sobre el input) -> similitud entidad-entidad.
    S = slim.agregar_W_a_entidades(
        cols_idx, cols_val, idx.row_entity, mf.weight, n_ent
    )

    meta = {
        "n_observaciones": M,
        "n_entidades": n_ent,
        "n_neighbors": config.F2_N_NEIGHBORS,
        "beta": config.F2_BETA,
        "l1": config.F2_L1,
        "nnz_W": nnz,
        "normalizacion": normalizacion,
        "ligas_por_entidad": ligas_por_entidad(mf, idx),
        "descripcion": (
            "SLIM instancia-instancia (W MxM sobre observaciones) + agregacion "
            "ponderada por minutos y normalizada por N_p*N_q sobre W."
        ),
    }
    return ModeloSimilitud(
        formulacion="2",
        entidad=entidad,
        S=S,
        entity_ids=idx.ids,
        entity_names=idx.names,
        feat_names=mf.feat_names,
        feat_display=features_display(mf, idx),
        meta=meta,
    )

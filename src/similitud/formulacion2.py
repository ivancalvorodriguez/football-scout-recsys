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

from . import config, distancias, slim, warm
from .distancias import Espacio
from .features import MatrizFeatures
from .modelo import (
    ModeloSimilitud,
    features_display,
    indexar_entidades,
    ligas_por_entidad,
)
from .warm import EstadoWarm


def hiperparametros() -> dict:
    """Identidad del ajuste: con otros valores, un estado previo no vale."""
    return {
        "n_neighbors": config.F2_N_NEIGHBORS,
        "beta": config.F2_BETA,
        "l1": config.F2_L1,
        "max_iter": config.F2_MAX_ITER,
        "tol": config.F2_TOL,
        **warm.hiper_features(),
    }


def construir(
    mf: MatrizFeatures, entidad: str, normalizacion: str = "por_liga",
    distancia: str = distancias.POR_DEFECTO,
) -> ModeloSimilitud:
    """Entrena la Formulacion 2 en frio y devuelve el modelo servible."""
    modelo, _ = construir_con_estado(mf, entidad, normalizacion, distancia=distancia)
    return modelo


def construir_con_estado(
    mf: MatrizFeatures,
    entidad: str,
    normalizacion: str = "por_liga",
    previo: EstadoWarm | None = None,
    progreso=None,
    congelar_kernel: bool = True,
    distancia: str = distancias.POR_DEFECTO,
) -> tuple[ModeloSimilitud, EstadoWarm]:
    """Entrena la Formulacion 2 y devuelve tambien el estado reutilizable.

    ``congelar_kernel`` no aplica aqui (no hay kernel que congelar): se acepta
    para que quien reentrena pueda despachar a cualquiera de las dos
    formulaciones con los mismos argumentos.

    Con ``previo`` se hace WARM START: cada columna del coordinate descent
    arranca en la solucion que tenia esa misma observacion en el ajuste anterior,
    traducida al espacio de indices de ahora (`warm.inicializador_f2`). El
    objetivo es estrictamente convexo (beta > 0), asi que el optimo no cambia:
    solo se llega antes. Las observaciones nuevas arrancan en frio.

    ``distancia`` es la geometria en la que se eligen los vecinos y se mide el
    residuo del ajuste (`src.similitud.distancias`). Con `mahalanobis`, lo que
    esa geometria aprende de los datos (el blanqueo) se CONGELA igual que las
    mu/sd del z-score: si hay estado previo compatible se reutiliza el suyo, para
    que anadir partidos no mueva a las entidades que no han jugado.
    ``congelar_kernel=False`` lo recalcula, con la misma logica que en la F5.
    """
    idx = indexar_entidades(mf)
    n_ent = len(idx.ids)
    M = mf.X.shape[0]
    distancia = distancias.validar(distancia)

    hiper = hiperparametros()
    emp = warm.emparejar(previo, mf, idx.ids, hiper, distancia=distancia)
    reutiliza = previo is not None and emp.hay_reutilizacion
    inicial = warm.inicializador_f2(previo, emp) if reutiliza else None
    espacio = _espacio_del_ajuste(mf, distancia, previo, reutiliza, congelar_kernel)

    # (3)->(4) SLIM instancia-instancia: W dispersa por columnas (observaciones).
    cols_idx, cols_val = slim.slim_instancia(
        mf.X,
        n_neighbors=config.F2_N_NEIGHBORS,
        beta=config.F2_BETA,
        l1=config.F2_L1,
        max_iter=config.F2_MAX_ITER,
        tol=config.F2_TOL,
        progreso=progreso,
        inicial=inicial,
        espacio=espacio,
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
        "distancia": distancia,
        "ligas_por_entidad": ligas_por_entidad(mf, idx),
        "warm": {"aplicado": inicial is not None, "motivo": emp.motivo, **emp.resumen()},
        "estadisticas_normalizacion": (
            mf.estadisticas.como_dict() if mf.estadisticas is not None else None
        ),
        "descripcion": (
            "SLIM instancia-instancia (W MxM sobre observaciones) + agregacion "
            "ponderada por minutos y normalizada por N_p*N_q sobre W."
        ),
    }
    modelo = ModeloSimilitud(
        formulacion="2",
        entidad=entidad,
        S=S,
        entity_ids=idx.ids,
        entity_names=idx.names,
        feat_names=mf.feat_names,
        feat_display=features_display(mf, idx),
        meta=meta,
    )
    estado = warm.estado_base(mf, "2", entidad, normalizacion, idx.ids, hiper,
                              espacio=espacio)
    estado.w_ptr, estado.w_idx, estado.w_val = warm.comprimir_W(cols_idx, cols_val)
    return modelo, estado


def _espacio_del_ajuste(
    mf: MatrizFeatures, distancia: str, previo: EstadoWarm | None,
    reutiliza: bool, congelar: bool,
) -> Espacio:
    """La geometria con la que se ajusta: la del estado previo, o una nueva.

    Compartida por las dos formulaciones (la F5 la importa de aqui). Reutilizar
    la anterior es lo mismo que congelar el ancho del kernel o las mu/sd del
    z-score: lo que la distancia estima del dataset entero (hoy el blanqueo de
    mahalanobis) se mueve al anadir partidos, y con el se moverian TODAS las
    entidades, tambien las que no han jugado.

    Las distancias que no aprenden nada de los datos —euclidea, coseno,
    manhattan— dan el mismo `Espacio` por los dos caminos, asi que esto solo
    tiene efecto real en mahalanobis.
    """
    if (
        reutiliza and congelar and previo is not None
        and previo.distancia == distancia
        and (previo.espacio_L is not None or distancia != "mahalanobis")
    ):
        return previo.espacio()
    return distancias.preparar(mf.X, distancia)

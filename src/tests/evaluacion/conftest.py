"""Constructores y fixtures propios de las pruebas de `src.evaluacion`.

El harness consume dos cosas que en produccion salen de la BD y de
`outputs/modelo/`: un `ModeloSimilitud` ya entrenado y un `Contexto` de
reconstruccion. Fabricarlos aqui a mano —con una S escrita a mano y unas features
minimas— es lo que permite comprobar cada fase contra un resultado CONOCIDO en vez
de contra lo que devuelva el pipeline ese dia. Las pruebas que si necesitan el
pipeline real viven en `integracion/` y usan la BD sintetica de la suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from src.evaluacion import construccion, registro
from src.extraccion import database as db
from src.similitud.distancias import DISTANCIAS_VALIDAS
from src.similitud.features import MatrizFeatures
from src.similitud.modelo import ModeloSimilitud, indexar_entidades


# --- Piezas del pipeline, fabricadas a mano ---------------------------------

def features_sinteticas(
    X: Sequence[Sequence[float]],
    entity_id: Sequence[int],
    *,
    weight: Sequence[float] | None = None,
    league: Sequence[str] | None = None,
    nombres: dict[int, str] | None = None,
) -> MatrizFeatures:
    """`MatrizFeatures` minima: una fila por observacion, como la real.

    No pasa por `features.construir` a proposito: aqui interesa controlar la X
    exacta (para saber que S deberia salir), no reproducir el z-score.
    """
    X = np.asarray(X, dtype=float)
    entity_id = np.asarray(entity_id, dtype=np.int64)
    nombres = nombres or {}
    return MatrizFeatures(
        X=X,
        feat_names=[f"f{j}" for j in range(X.shape[1])],
        entity_id=entity_id,
        entity_name=np.array([nombres.get(int(e), f"Ent {e}") for e in entity_id]),
        weight=(np.ones(len(entity_id), dtype=float) if weight is None
                else np.asarray(weight, dtype=float)),
        league=(np.array(["1-100"] * len(entity_id)) if league is None
                else np.asarray(league)),
    )


def contexto_sintetico(
    X: Sequence[Sequence[float]],
    entity_id: Sequence[int],
    match_id: Sequence[int],
    *,
    entidad: str = "jugador",
    normalizacion: str = "por_liga",
    minutos: Sequence[float] | None = None,
    weight: Sequence[float] | None = None,
) -> construccion.Contexto:
    """`Contexto` sin tocar la BD (lo que `crear_contexto` monta desde SQLite)."""
    mf = features_sinteticas(X, entity_id, weight=weight)
    return construccion.Contexto(
        entidad=entidad,
        normalizacion=normalizacion,
        mf=mf,
        match_id=np.asarray(match_id, dtype=np.int64),
        minutos_obs=(np.ones(len(entity_id), dtype=float) if minutos is None
                     else np.asarray(minutos, dtype=float)),
        idx_real=indexar_entidades(mf),
    )


def modelo_sintetico(
    S: Sequence[Sequence[float]],
    *,
    formulacion: str = "5",
    entidad: str = "jugador",
    entity_ids: Sequence[int] | None = None,
    entity_names: list[str] | None = None,
    feat_display: Sequence[Sequence[float]] | None = None,
    feat_names: list[str] | None = None,
    meta: dict | None = None,
) -> ModeloSimilitud:
    """Artefacto servible con la S que pida la prueba (diag = 0 como el real)."""
    S = np.asarray(S, dtype=float)
    P = S.shape[0]
    ids = np.arange(P, dtype=np.int64) if entity_ids is None else np.asarray(entity_ids)
    return ModeloSimilitud(
        formulacion=formulacion,
        entidad=entidad,
        S=S,
        entity_ids=ids,
        entity_names=entity_names or [f"Ent {int(i)}" for i in ids],
        feat_names=feat_names or ["f0", "f1"],
        feat_display=(np.zeros((P, 2)) if feat_display is None
                      else np.asarray(feat_display, dtype=float)),
        meta=meta or {"normalizacion": "por_liga"},
    )


# --- Base de datos con posiciones variadas ----------------------------------
# `factories.crear_bd_sintetica` pone a todos los jugadores en "Center Midfield"
# (le basta para el modelo, que descarta la posicion). El harness de evaluacion
# SI la usa como ground truth debil, asi que necesita una BD donde los roles
# gruesos discrepen.

def bd_con_posiciones(
    ruta: Path, filas: Sequence[tuple[int, int, str, float]]
) -> Path:
    """BD con el esquema REAL y solo las columnas que lee `src.evaluacion.datos`.

    `filas` = (player_id, match_id, primary_position_name, minutes_played). Se usa
    el esquema de produccion (`database.init_schema`) para que la prueba se entere
    si cambia; lo demas se deja vacio porque `datos` no lo consulta. Las filas
    padre (jugadores y partidos) se crean igualmente: el esquema real tiene las
    claves ajenas activadas.
    """
    ruta = Path(ruta)
    conn = db.connect(ruta)
    db.init_schema(conn)
    conn.executemany(
        "INSERT OR IGNORE INTO players (player_id, player_name) VALUES (?, ?)",
        [(int(p), f"Jugador {int(p)}") for p, _m, _pos, _min in filas],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO matches (match_id) VALUES (?)",
        [(int(m),) for _p, m, _pos, _min in filas],
    )
    conn.executemany(
        "INSERT INTO player_match_stats "
        "(player_id, match_id, primary_position_name, minutes_played) "
        "VALUES (?, ?, ?, ?)",
        [(int(p), int(m), pos, float(mins)) for p, m, pos, mins in filas],
    )
    conn.commit()
    conn.close()
    return ruta


@pytest.fixture
def bd_roles(tmp_path: Path) -> Path:
    """Cuatro jugadores con rol grueso distinto, uno de ellos con dos posiciones.

    - 1: portero puro (GK).
    - 2: mas minutos de central que de lateral -> DEF por el rol mayoritario.
    - 3: medio (MID) repartido entre dos posiciones que agregan al MISMO rol.
    - 4: posicion que no esta en el mapeo -> ROL_DESCONOCIDO.
    """
    return bd_con_posiciones(tmp_path / "roles.db", [
        (1, 10, "Goalkeeper", 90.0),
        (1, 11, "Goalkeeper", 90.0),
        (2, 10, "Center Back", 80.0),
        (2, 11, "Right Wing", 20.0),
        (3, 10, "Center Midfield", 45.0),
        (3, 11, "Left Midfield", 30.0),
        (4, 10, "Inventada", 60.0),
    ])


# --- Carpeta de barrido sintetica -------------------------------------------

def escribir_barrido(
    barrido_dir: Path,
    metricas: Sequence[dict[str, Any]],
    combinaciones: dict[str, dict[str, Any]],
) -> Path:
    """Carpeta de barrido ya generada: `barrido_metricas.csv` + `combinaciones.json`.

    Es lo que consume `figuras3d` (que no construye ni evalua nada), asi que sus
    pruebas no necesitan pagar un barrido de verdad.
    """
    barrido_dir = Path(barrido_dir)
    barrido_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(metricas)).to_csv(
        barrido_dir / registro.ARCHIVO_METRICAS, index=False)
    (barrido_dir / registro.ARCHIVO).write_text(
        json.dumps({
            "version": registro.VERSION,
            "combinaciones": {n: {"hiperparametros": c}
                              for n, c in combinaciones.items()},
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return barrido_dir


def fila_metrica(
    combinacion: str, modelo: str, metrica: str, valor: float, *, fase: str = "1",
    distancia: str | None = None,
) -> dict[str, Any]:
    """Fila de `barrido_metricas.csv` (el modelo ya descompuesto en sus cuatro ejes).

    `modelo` se acepta con y sin el sufijo de distancia (`F5_equipo_global` o
    `F5_equipo_global_euclidea`): sin el se asume `euclidea` —la unica geometria
    que existia antes de que la distancia fuera un eje— y se completa la etiqueta,
    que es exactamente lo que hace `registro._tipos` al leer una tabla antigua.
    """
    # Se parsea por la DERECHA: la normalizacion lleva guion bajo (`por_liga`),
    # asi que partir por la izquierda la rompe en dos. El ultimo token es la
    # distancia solo si es una de las conocidas; si no, el modelo viene sin ella.
    form, entidad, resto = modelo.lstrip("F").split("_", 2)
    if "_" in resto and resto.rsplit("_", 1)[1] in DISTANCIAS_VALIDAS:
        norm, dist_en_nombre = resto.rsplit("_", 1)
    else:
        norm, dist_en_nombre = resto, None
    dist = distancia or dist_en_nombre or "euclidea"
    return {"combinacion": combinacion, "modelo": f"F{form}_{entidad}_{norm}_{dist}",
            "formulacion": form, "entidad": entidad, "normalizacion": norm,
            "distancia": dist,
            "fase": fase, "metrica": metrica, "valor": valor}


@pytest.fixture
def barrido_2x2(tmp_path: Path) -> Path:
    """Barrido de dos ejes x dos valores POR FORMULACION, tres modelos, dos metricas.

    Es el minimo que da una superficie: dos ejes con mas de un valor cada uno.

    Cada formulacion mueve LOS SUYOS, que es como sale de un barrido de verdad:
    `F2_L1`/`F2_BETA` no entran en la huella de un modelo F5 ni
    `F5_EASE_LAMBDA`/`F5_RFF_DIM` en la de uno F2 (ver `huella.alcance`), asi que
    cada modelo se dibuja sobre su propia pareja de ejes. `F5_METODO_EQUIPO` se
    queda constante para que haya tambien un eje FIJO que declarar en el pie.

    Los dos modelos F5 (las dos normalizaciones) comparten ejes: son los que puede
    superponer la figura comparativa del score.
    """
    combis = {
        "v01": {"F2_L1": 0.25, "F2_BETA": 1.0, "F5_EASE_LAMBDA": 10.0,
                "F5_RFF_DIM": 512, "F5_METODO_EQUIPO": "sinkhorn"},
        "v02": {"F2_L1": 0.5, "F2_BETA": 1.0, "F5_EASE_LAMBDA": 50.0,
                "F5_RFF_DIM": 512, "F5_METODO_EQUIPO": "sinkhorn"},
        "v03": {"F2_L1": 0.25, "F2_BETA": 2.0, "F5_EASE_LAMBDA": 10.0,
                "F5_RFF_DIM": 1024, "F5_METODO_EQUIPO": "sinkhorn"},
        "v04": {"F2_L1": 0.5, "F2_BETA": 2.0, "F5_EASE_LAMBDA": 50.0,
                "F5_RFF_DIM": 1024, "F5_METODO_EQUIPO": "sinkhorn"},
    }
    valores = {"v01": 0.10, "v02": 0.30, "v03": 0.20, "v04": 0.25}
    filas = []
    for combi, base in valores.items():
        for i, modelo in enumerate(("F2_equipo_global", "F5_equipo_global",
                                    "F5_equipo_por_liga")):
            filas.append(fila_metrica(combi, modelo, "top1", base + 0.05 * i))
            filas.append(fila_metrica(combi, modelo, "asimetria",
                                      0.5 - base, fase="0"))
    return escribir_barrido(tmp_path / "barrido", filas, combis)

"""Datos auxiliares derivados de la BD para la evaluacion (solo lectura).

- Rol grueso por jugador (para pureza posicional y clasificacion downstream),
  como el rol donde acumula mas minutos.
- Minutos totales por entidad (para estratificar la auto-similitud en
  cabeza/torso/cola, como pide el PDF por el sesgo de popularidad/MNAR).

No toca el pipeline de `src.similitud`: solo consulta columnas que ese pipeline
descarta (posicion, que "no entra como feature" por diseno).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _leer(sql: str, db_path: Path) -> pd.DataFrame:
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        return pd.read_sql(sql, conn)


def rol_por_jugador(db_path: Path) -> dict[int, str]:
    """Mapa player_id -> rol grueso (GK/DEF/MID/ATT) por minutos acumulados.

    Se elige el rol donde el jugador suma mas minutos a lo largo de sus partidos
    (la posicion "no entra como feature" en el modelo; aqui es solo el ground
    truth debil para los sanity checks y la tarea downstream).
    """
    df = _leer(
        "SELECT player_id, primary_position_name AS pos, "
        "SUM(minutes_played) AS min FROM player_match_stats "
        "GROUP BY player_id, primary_position_name",
        db_path,
    )
    df["rol"] = df["pos"].map(config.ROL_GRUESO).fillna(config.ROL_DESCONOCIDO)
    # Rol con mas minutos por jugador.
    agg = (
        df.groupby(["player_id", "rol"])["min"].sum().reset_index()
    )
    idx = agg.groupby("player_id")["min"].idxmax()
    ganador = agg.loc[idx]
    return {int(p): str(r) for p, r in zip(ganador["player_id"], ganador["rol"])}


def minutos_por_jugador(db_path: Path) -> dict[int, float]:
    """Mapa player_id -> minutos totales (para estratificar por volumen)."""
    df = _leer(
        "SELECT player_id, SUM(minutes_played) AS min "
        "FROM player_match_stats GROUP BY player_id",
        db_path,
    )
    return {int(p): float(m) for p, m in zip(df["player_id"], df["min"])}


def terciles(valores: np.ndarray) -> np.ndarray:
    """Etiqueta cada valor como 0=cola, 1=torso, 2=cabeza por terciles."""
    if valores.size == 0:
        return np.array([], dtype=int)
    q1, q2 = np.percentile(valores, [33.333, 66.667])
    etiquetas = np.zeros(len(valores), dtype=int)
    etiquetas[valores > q1] = 1
    etiquetas[valores > q2] = 2
    return etiquetas


NOMBRE_TERCIL = {0: "cola", 1: "torso", 2: "cabeza"}

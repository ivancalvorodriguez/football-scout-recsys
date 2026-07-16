"""Carga desde SQLite de las observaciones por-partido (solo lectura).

Devuelve una fila por observacion (jugador-partido o equipo-partido) SIN
agregar: cada entidad conserva todas sus filas, como exige el principio central.
Cada fila lleva ademas su liga (competition_id, season_id via `matches`) para
poder estandarizar por competicion, y el nombre de la entidad para servir
consultas por nombre.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# Clave de liga usada para el z-score (competicion + temporada).
LEAGUE_KEY = "league_key"


def _connect(db_path: Path) -> sqlite3.Connection:
    """Abre la BD en modo solo lectura (no se modifica nunca)."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def cargar_jugadores(db_path: Path) -> pd.DataFrame:
    """Una fila por (jugador, partido) con nombre, liga y minutos.

    `entity_id`/`entity_name` homogeneizan la interfaz con `cargar_equipos`.
    """
    sql = """
        SELECT p.*, pl.player_name AS entity_name,
               m.competition_id, m.season_id
        FROM player_match_stats p
        JOIN players pl ON pl.player_id = p.player_id
        JOIN matches m ON m.match_id = p.match_id
    """
    with _connect(db_path) as conn:
        df = pd.read_sql(sql, conn)
    df = df.rename(columns={"player_id": "entity_id"})
    df[LEAGUE_KEY] = (
        df["competition_id"].astype(str) + "-" + df["season_id"].astype(str)
    )
    return df


def cargar_equipos(db_path: Path) -> pd.DataFrame:
    """Una fila por (equipo, partido) con nombre y liga.

    El equipo no tiene `minutes_played` (juega el partido completo): la
    ponderacion por minutos no aplica y se usara masa uniforme aguas arriba.
    """
    sql = """
        SELECT t.*, te.team_name AS entity_name,
               m.competition_id, m.season_id
        FROM team_match_stats t
        JOIN teams te ON te.team_id = t.team_id
        JOIN matches m ON m.match_id = t.match_id
    """
    with _connect(db_path) as conn:
        df = pd.read_sql(sql, conn)
    df = df.rename(columns={"team_id": "entity_id"})
    df[LEAGUE_KEY] = (
        df["competition_id"].astype(str) + "-" + df["season_id"].astype(str)
    )
    return df


def cargar(db_path: Path, entidad: str) -> pd.DataFrame:
    """Despacha por tipo de entidad ('jugador' | 'equipo')."""
    if entidad == "jugador":
        return cargar_jugadores(db_path)
    if entidad == "equipo":
        return cargar_equipos(db_path)
    raise ValueError(f"entidad desconocida: {entidad!r} (usa 'jugador' o 'equipo')")

"""Carga desde SQLite de las observaciones por-partido (solo lectura).

Devuelve una fila por observacion (jugador-partido o equipo-partido) SIN
agregar: cada entidad conserva todas sus filas, como exige el principio central.
Cada fila lleva ademas su liga (competition_id, season_id via `matches`) para
poder estandarizar por competicion, y el nombre de la entidad para servir
consultas por nombre.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from src.extraccion.config import position_slug

from . import config

# Clave de liga usada para el z-score (competicion + temporada).
LEAGUE_KEY = "league_key"


def _connect(db_path: Path) -> sqlite3.Connection:
    """Abre la BD en modo solo lectura (no se modifica nunca)."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _leer(sql: str, db_path: Path) -> pd.DataFrame:
    """Ejecuta una consulta y cierra la conexion.

    `contextlib.closing` es necesario: el context manager propio de una conexion
    sqlite3 hace commit/rollback pero NO la cierra, asi que un `with conn:` a
    secas deja la conexion abierta.
    """
    with closing(_connect(db_path)) as conn:
        return pd.read_sql(sql, conn)


def _fracciones_posicion(db_path: Path) -> pd.DataFrame:
    """Fraccion de minutos por posicion de cada (jugador, partido).

    Pivota `player_match_positions` a las 25 columnas `pos_*` del catalogo
    StatsBomb, cada una con minutos_en_posicion / minutos_totales del jugador en
    ese partido (one-hot "blando": suma 1 salvo redondeo; one-hot puro si no
    cambia de posicion). Devuelve `player_id`, `match_id` y las 25 columnas; un
    jugador-partido ausente de la tabla se resuelve a 0 en el merge de arriba.

    Se carga siempre (es barato): esta capa es un loader agnostico al modelo; que
    la posicion se USE o no lo decide `features` segun `config.USE_POSITION_FEATURES`.
    """
    sql = "SELECT player_id, match_id, position_name, minutes FROM player_match_positions"
    pos = _leer(sql, db_path)
    if pos.empty:
        return pd.DataFrame(columns=["player_id", "match_id", *config.POSITION_FEATURES])
    pos["slug"] = pos["position_name"].map(position_slug)
    total = pos.groupby(["player_id", "match_id"])["minutes"].transform("sum")
    pos["frac"] = np.where(total.to_numpy() > 0.0, pos["minutes"] / total, 0.0)
    ancho = pos.pivot_table(
        index=["player_id", "match_id"], columns="slug", values="frac",
        aggfunc="sum", fill_value=0.0,
    )
    ancho = ancho.reindex(columns=config.POSITION_FEATURES, fill_value=0.0)
    return ancho.reset_index()


def cargar_jugadores(db_path: Path) -> pd.DataFrame:
    """Una fila por (jugador, partido) con nombre, liga, minutos y posicion.

    `entity_id`/`entity_name` homogeneizan la interfaz con `cargar_equipos`. Se
    adjuntan las 25 columnas `pos_*` (fraccion de minutos por posicion): la capa
    de features las incorpora solo si `config.USE_POSITION_FEATURES` esta activo.
    """
    sql = """
        SELECT p.*, pl.player_name AS entity_name,
               m.competition_id, m.season_id
        FROM player_match_stats p
        JOIN players pl ON pl.player_id = p.player_id
        JOIN matches m ON m.match_id = p.match_id
    """
    df = _leer(sql, db_path)
    df = df.merge(_fracciones_posicion(db_path), on=["player_id", "match_id"], how="left")
    # Jugador-partido sin registro de posicion -> sin senal (todo a 0).
    df[config.POSITION_FEATURES] = df[config.POSITION_FEATURES].fillna(0.0)
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
    df = _leer(sql, db_path)
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

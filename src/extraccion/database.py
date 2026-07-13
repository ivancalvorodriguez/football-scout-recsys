"""Almacenamiento en SQLite: esquema (dimensiones + hechos) y escritura con UPSERT.

Modelo estrella:
- Dimensiones: competitions, seasons, teams, players.
- Hechos: matches, player_match_stats (grano jugador-partido),
  team_match_stats (grano equipo-partido), player_match_positions (detalle de
  minutos por posición para cada jugador-partido).

El one-hot de posición NO se persiste: se guarda `primary_position_*` y el
detalle por posición, y el one-hot se genera al construir la matriz de features.
Reprocesar un partido hace UPSERT sobre las claves primarias (sin duplicados).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

# --- Columnas de métricas persistidas (esquema estable y documentado) --------
# Se guardan CONTEOS CRUDOS. Los ratios (%_, per-shot, diferencias G-xG...) son
# derivables de estos conteos y se calculan en la capa de features, junto con la
# normalización por 90', antes del modelo de similitud (ver docs/).
PLAYER_METRIC_COLUMNS: list[str] = [
    # Progresión
    "passes", "passes_completed",
    "progressive_passes", "progressive_carries", "passes_into_final_third", "xt",
    # Creación
    "xa", "passes_into_penalty_area", "deep_completions", "sca",
    # Finalización
    "shots", "np_shots", "goals", "np_goals", "shots_on_target",
    "xg", "npxg", "touches_in_att_pen_area",
    # Duelos (duels_total/duels_won agregan tackles + aéreos + 50/50)
    "take_ons", "take_ons_won", "fouls_drawn", "dispossessed",
    "tackles", "tackles_won", "dribbled_past",
    "duels_total", "duels_won",
    # Juego aéreo
    "aerial_won", "aerial_lost", "aerial_won_off", "aerial_won_def",
    # Defensa
    "interceptions", "ball_recoveries", "padj_def_actions",
    "blocks", "clearances", "pressures", "pressure_regains", "counterpressures",
]

TEAM_METRIC_COLUMNS: list[str] = [
    # Posesión / elaboración
    "possession_pct", "passes", "passes_completed", "ten_plus_pass_sequences",
    # Progresión
    "progressive_passes", "passes_into_final_third", "xt",
    # Creación / amenaza
    "sca", "gca", "passes_into_penalty_area", "open_play_xg", "build_up_attacks",
    # Finalización
    "shots", "np_shots", "goals", "shots_on_target", "xg", "npxg",
    # Presión / recuperación
    "ppda", "high_turnovers", "pressures", "counterpressures",
    # Dominio territorial
    "field_tilt", "absolute_width", "sequence_start_distance",
    # Perfil / estilo
    "direct_speed", "direct_attacks", "passes_per_sequence", "open_play_sequences",
]

_PLAYER_CONTEXT = [
    "player_id", "match_id", "team_id", "minutes_played",
    "primary_position_id", "primary_position_name",
]
_TEAM_CONTEXT = [
    "team_id", "match_id", "opponent_id", "is_home", "goals_for", "goals_against",
]

PLAYER_COLUMNS = _PLAYER_CONTEXT + PLAYER_METRIC_COLUMNS
TEAM_COLUMNS = _TEAM_CONTEXT + TEAM_METRIC_COLUMNS


def _cols_ddl(columns: Iterable[str], real_from: set[str]) -> str:
    parts = []
    for c in columns:
        parts.append(f'"{c}" REAL' if c in real_from else f'"{c}" INTEGER')
    return ", ".join(parts)


def connect(db_path: Path) -> sqlite3.Connection:
    """Abre (creando carpetas) la base de datos y activa claves foráneas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS competitions (
            competition_id INTEGER PRIMARY KEY,
            competition_name TEXT,
            country_name TEXT,
            competition_gender TEXT
        );
        CREATE TABLE IF NOT EXISTS seasons (
            season_id INTEGER PRIMARY KEY,
            season_name TEXT
        );
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT
        );
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT
        );
        -- Un equipo puede jugar en varias competiciones-temporada (liga, copa,
        -- Champions...): relación N:M. Permite saber en qué liga(s) juega un
        -- equipo y contextualizar sus métricas por competición.
        CREATE TABLE IF NOT EXISTS team_competitions (
            team_id INTEGER,
            competition_id INTEGER,
            season_id INTEGER,
            PRIMARY KEY (team_id, competition_id, season_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (competition_id) REFERENCES competitions(competition_id),
            FOREIGN KEY (season_id) REFERENCES seasons(season_id)
        );
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            competition_id INTEGER,
            season_id INTEGER,
            match_date TEXT,
            match_week INTEGER,
            competition_stage TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            FOREIGN KEY (competition_id) REFERENCES competitions(competition_id),
            FOREIGN KEY (season_id) REFERENCES seasons(season_id)
        );
        CREATE TABLE IF NOT EXISTS player_match_positions (
            player_id INTEGER,
            match_id INTEGER,
            position_id INTEGER,
            position_name TEXT,
            minutes REAL,
            PRIMARY KEY (player_id, match_id, position_id)
        );
        """
    )
    # Tablas de hechos con columnas de métricas generadas.
    player_metric_ddl = _cols_ddl(PLAYER_METRIC_COLUMNS, real_from=_PLAYER_REAL)
    team_metric_ddl = _cols_ddl(TEAM_METRIC_COLUMNS, real_from=_TEAM_REAL)
    cur.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS player_match_stats (
            player_id INTEGER,
            match_id INTEGER,
            team_id INTEGER,
            minutes_played REAL,
            primary_position_id INTEGER,
            primary_position_name TEXT,
            {player_metric_ddl},
            PRIMARY KEY (player_id, match_id),
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            FOREIGN KEY (match_id) REFERENCES matches(match_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        );
        CREATE TABLE IF NOT EXISTS team_match_stats (
            team_id INTEGER,
            match_id INTEGER,
            opponent_id INTEGER,
            is_home INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            {team_metric_ddl},
            PRIMARY KEY (team_id, match_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );
        """
    )
    conn.commit()


# Métricas de tipo REAL (el resto de conteos van como INTEGER; NULL permitido).
_PLAYER_REAL = {"xt", "xa", "xg", "npxg", "padj_def_actions"}
_TEAM_REAL = {
    "possession_pct", "xt", "open_play_xg", "xg", "npxg", "ppda", "field_tilt",
    "absolute_width", "sequence_start_distance", "direct_speed", "passes_per_sequence",
}


def _upsert(conn: sqlite3.Connection, table: str, columns: list[str], row: dict[str, Any]) -> None:
    placeholders = ", ".join("?" for _ in columns)
    col_list = ", ".join(f'"{c}"' for c in columns)
    values = [row.get(c) for c in columns]
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})", values
    )


def upsert_competition(conn: sqlite3.Connection, comp: dict[str, Any]) -> None:
    _upsert(conn, "competitions",
            ["competition_id", "competition_name", "country_name", "competition_gender"], comp)


def upsert_season(conn: sqlite3.Connection, season: dict[str, Any]) -> None:
    _upsert(conn, "seasons", ["season_id", "season_name"], season)


def upsert_team(conn: sqlite3.Connection, team_id: int, team_name: str) -> None:
    _upsert(conn, "teams", ["team_id", "team_name"],
            {"team_id": team_id, "team_name": team_name})


def upsert_player(conn: sqlite3.Connection, player_id: int, player_name: str) -> None:
    _upsert(conn, "players", ["player_id", "player_name"],
            {"player_id": player_id, "player_name": player_name})


def upsert_team_competition(
    conn: sqlite3.Connection, team_id: int, competition_id: int, season_id: int
) -> None:
    _upsert(conn, "team_competitions", ["team_id", "competition_id", "season_id"],
            {"team_id": team_id, "competition_id": competition_id, "season_id": season_id})


def upsert_match(conn: sqlite3.Connection, match: dict[str, Any]) -> None:
    _upsert(conn, "matches", [
        "match_id", "competition_id", "season_id", "match_date", "match_week",
        "competition_stage", "home_team_id", "away_team_id", "home_score", "away_score",
    ], match)


def upsert_player_stats(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    _upsert(conn, "player_match_stats", PLAYER_COLUMNS, row)


def upsert_team_stats(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    _upsert(conn, "team_match_stats", TEAM_COLUMNS, row)


def replace_player_positions(
    conn: sqlite3.Connection, player_id: int, match_id: int, positions: list[dict[str, Any]]
) -> None:
    conn.execute(
        "DELETE FROM player_match_positions WHERE player_id=? AND match_id=?",
        (player_id, match_id),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO player_match_positions "
        "(player_id, match_id, position_id, position_name, minutes) VALUES (?,?,?,?,?)",
        [
            (player_id, match_id, p["position_id"], p["position_name"], p["minutes"])
            for p in positions
        ],
    )

"""Punto de entrada: extrae métricas por (jugador, partido) y (equipo, partido).

Uso interactivo (pregunta competición, temporada, % de jornadas y ruta de BD):

    python -m src.extraccion.extract

Uso con argumentos (todo por nombre, no por id):

    python -m src.extraccion.extract --competition "Premier League" --season "2015/2016" \
        --pct 50 --db outputs/db/scouting.db

Los datos se guardan en SQLite (ver `src/extraccion/database.py`). Reprocesar
partidos ya almacenados los actualiza (UPSERT), no los duplica.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import database as db
from .config import DATA_ROOT, DEFAULT_DB_PATH
from .minutes import player_minutes
from .player_stats import _finalize, _new_row, compute_player_stats
from .selection import find_competition, find_season, interactive_selection, select_matchday_subset
from .statsbomb_io import load_events, load_lineups, load_matches
from .team_stats import compute_team_stats


def _padj_factor(possession_pct: float | None) -> float:
    """Factor de ajuste por posesión: infla si defiendes poco, deflacta si mucho.

    padj = acción_defensiva * 0.5 / posesión_rival. Referencia 50%.
    """
    if possession_pct is None:
        return 1.0
    opp = 1.0 - possession_pct
    return (0.5 / opp) if opp > 1e-6 else 1.0


def _build_player_row(
    match_id: int, rec: dict[str, Any], metrics: dict[str, Any], padj_factor: float
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "player_id": rec["player_id"],
        "match_id": match_id,
        "team_id": rec["team_id"],
        "minutes_played": rec["minutes_played"],
        "primary_position_id": rec["primary_position_id"],
        "primary_position_name": rec["primary_position_name"],
    }
    for col in db.PLAYER_METRIC_COLUMNS:
        row[col] = metrics.get(col)
    raw_def = (metrics.get("interceptions", 0) or 0) + (metrics.get("ball_recoveries", 0) or 0)
    row["padj_def_actions"] = round(raw_def * padj_factor, 3)
    return row


def _build_team_row(
    match_id: int, team_id: int, metrics: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "team_id": team_id,
        "match_id": match_id,
        "opponent_id": context["opponent_id"],
        "is_home": context["is_home"],
        "goals_for": context["goals_for"],
        "goals_against": context["goals_against"],
    }
    for col in db.TEAM_METRIC_COLUMNS:
        row[col] = metrics.get(col)
    row["open_play_sequences"] = int(metrics.get("seq_count", 0) or 0)
    return row


def process_match(
    conn, match: dict[str, Any], competition_id: int, season_id: int, data_root: Path
) -> tuple[int, int]:
    """Procesa un partido y lo guarda. Devuelve (n_jugadores, n_equipos)."""
    match_id = int(match["match_id"])
    events = load_events(match_id, data_root)
    lineups = load_lineups(match_id, data_root)

    minutes = player_minutes(lineups, events)
    player_metrics = compute_player_stats(events)
    team_metrics = compute_team_stats(events)
    empty_metrics = _finalize(_new_row())

    home_id = int(match["home_team"]["home_team_id"])
    away_id = int(match["away_team"]["away_team_id"])
    home_name = match["home_team"]["home_team_name"]
    away_name = match["away_team"]["away_team_name"]
    home_score = match.get("home_score")
    away_score = match.get("away_score")

    # --- Dimensiones y partido ----------------------------------------------
    db.upsert_team(conn, home_id, home_name)
    db.upsert_team(conn, away_id, away_name)
    db.upsert_team_competition(conn, home_id, competition_id, season_id)
    db.upsert_team_competition(conn, away_id, competition_id, season_id)
    db.upsert_match(conn, {
        "match_id": match_id,
        "competition_id": competition_id,
        "season_id": season_id,
        "match_date": match.get("match_date"),
        "match_week": match.get("match_week"),
        "competition_stage": (match.get("competition_stage") or {}).get("name"),
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_score": home_score,
        "away_score": away_score,
    })

    # --- Jugadores -----------------------------------------------------------
    possession_by_team = {tid: m.get("possession_pct") for tid, m in team_metrics.items()}
    for rec in minutes:
        db.upsert_player(conn, rec["player_id"], rec["player_name"])
        metrics = player_metrics.get(rec["player_id"], empty_metrics)
        factor = _padj_factor(possession_by_team.get(rec["team_id"]))
        row = _build_player_row(match_id, rec, metrics, factor)
        db.upsert_player_stats(conn, row)
        db.replace_player_positions(conn, rec["player_id"], match_id, rec["positions"])

    # --- Equipos -------------------------------------------------------------
    for tid, metrics in team_metrics.items():
        if tid == home_id:
            context = {"opponent_id": away_id, "is_home": 1,
                       "goals_for": home_score, "goals_against": away_score}
        else:
            context = {"opponent_id": home_id, "is_home": 0,
                       "goals_for": away_score, "goals_against": home_score}
        db.upsert_team_stats(conn, _build_team_row(match_id, tid, metrics, context))

    conn.commit()
    return len(minutes), len(team_metrics)


def _resolve_from_args(args: argparse.Namespace, data_root: Path) -> dict[str, Any] | None:
    comp = find_competition(args.competition, data_root)
    if comp is None:
        print(f'No se encontró la competición "{args.competition}".')
        return None
    season = find_season(comp, args.season)
    if season is None:
        names = ", ".join(str(s["season_name"]) for s in comp["seasons"])
        print(f'No se encontró la temporada "{args.season}". Disponibles: {names}')
        return None
    matches = load_matches(comp["competition_id"], season["season_id"], data_root)
    if not matches:
        print("No hay partidos en disco para esa competición-temporada.")
        return None
    selected = select_matchday_subset(matches, args.pct)
    return {
        "competition": comp,
        "season": season,
        "matches": selected,
        "db_path": Path(args.db),
        "pct": args.pct,
    }


def run(selection: dict[str, Any], data_root: Path) -> None:
    comp = selection["competition"]
    season = selection["season"]
    matches = selection["matches"]
    db_path = selection["db_path"]

    conn = db.connect(db_path)
    db.init_schema(conn)
    db.upsert_competition(conn, {
        "competition_id": comp["competition_id"],
        "competition_name": comp["competition_name"],
        "country_name": comp["country_name"],
        "competition_gender": comp["competition_gender"],
    })
    db.upsert_season(conn, {
        "season_id": season["season_id"], "season_name": season["season_name"]})

    print(f'\nExtrayendo {comp["competition_name"]} — {season["season_name"]} '
          f'({len(matches)} partidos) -> {db_path}')
    total_players = total_teams = 0
    for i, match in enumerate(matches, start=1):
        home = match["home_team"]["home_team_name"]
        away = match["away_team"]["away_team_name"]
        np_, nt_ = process_match(
            conn, match, comp["competition_id"], season["season_id"], data_root)
        total_players += np_
        total_teams += nt_
        print(f'  [{i}/{len(matches)}] match {match["match_id"]}: {home} vs {away} '
              f'— {np_} jugadores')

    conn.close()
    print(f"\nHecho. Filas escritas: {total_players} jugador-partido, "
          f"{total_teams} equipo-partido en {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", help='Nombre de la competición (p. ej. "Premier League")')
    parser.add_argument("--season", help='Nombre de la temporada (p. ej. "2015/2016")')
    parser.add_argument("--pct", type=float, default=100.0,
                        help="Porcentaje de jornadas a procesar (1-100). Por defecto 100.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help="Ruta de la base de datos SQLite de salida.")
    parser.add_argument("--data-root", default=str(DATA_ROOT),
                        help="Raíz de los datos StatsBomb (open-data/data).")
    args = parser.parse_args()
    data_root = Path(args.data_root)

    if args.competition and args.season:
        selection = _resolve_from_args(args, data_root)
        if selection is None:
            return
    else:
        selection = interactive_selection(data_root)

    run(selection, data_root)


if __name__ == "__main__":
    main()

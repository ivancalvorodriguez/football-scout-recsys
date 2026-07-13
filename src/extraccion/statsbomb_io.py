"""Acceso a los ficheros de StatsBomb Open Data (lectura directa de JSON).

Sigue el patrón del proyecto: `json` + rutas `pathlib`, filtrando por
competición/temporada vía `matches/` antes de tocar `events/`. Nunca se cargan
todos los ficheros de eventos a ciegas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA_ROOT


def read_json(path: Path) -> Any:
    """Lee un JSON UTF-8."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_competitions(data_root: Path = DATA_ROOT) -> list[dict[str, Any]]:
    """Lista de filas competición-temporada de `competitions.json`."""
    return read_json(data_root / "competitions.json")


def list_competitions(data_root: Path = DATA_ROOT) -> list[dict[str, Any]]:
    """Competiciones distintas (una entrada por competición, agregando temporadas).

    Devuelve dicts con `competition_id`, `competition_name`, `country_name`,
    `competition_gender` y la lista de temporadas disponibles (con datos de
    partidos en disco), ordenadas por nombre.
    """
    comps = load_competitions(data_root)
    by_id: dict[int, dict[str, Any]] = {}
    for row in comps:
        cid = int(row["competition_id"])
        sid = int(row["season_id"])
        matches_path = data_root / "matches" / str(cid) / f"{sid}.json"
        entry = by_id.setdefault(
            cid,
            {
                "competition_id": cid,
                "competition_name": row.get("competition_name"),
                "country_name": row.get("country_name"),
                "competition_gender": row.get("competition_gender"),
                "seasons": [],
            },
        )
        entry["seasons"].append(
            {
                "season_id": sid,
                "season_name": row.get("season_name"),
                "available": matches_path.exists(),
            }
        )
    result = list(by_id.values())
    for entry in result:
        entry["seasons"].sort(key=lambda s: str(s["season_name"]))
    result.sort(key=lambda c: str(c["competition_name"]))
    return result


def load_matches(
    competition_id: int, season_id: int, data_root: Path = DATA_ROOT
) -> list[dict[str, Any]]:
    """Partidos de una competición-temporada (`matches/<comp>/<season>.json`)."""
    path = data_root / "matches" / str(competition_id) / f"{season_id}.json"
    if not path.exists():
        return []
    return read_json(path)


def load_events(match_id: int, data_root: Path = DATA_ROOT) -> list[dict[str, Any]]:
    """Eventos de un partido (`events/<match_id>.json`)."""
    return read_json(data_root / "events" / f"{match_id}.json")


def load_lineups(match_id: int, data_root: Path = DATA_ROOT) -> list[dict[str, Any]]:
    """Alineaciones de un partido (`lineups/<match_id>.json`)."""
    return read_json(data_root / "lineups" / f"{match_id}.json")

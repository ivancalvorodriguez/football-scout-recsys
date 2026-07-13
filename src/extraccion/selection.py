"""Selección de qué partidos procesar: por nombre de competición y temporada.

Todo se muestra y elige por **nombre** (p. ej. "Premier League"), nunca por id.
La cantidad se expresa como **porcentaje de jornadas** de la liga: se ordena por
`match_week` y se toman las primeras jornadas hasta cubrir el porcentaje pedido.
En torneos sin jornadas limpias (fases KO) se cae a % de partidos por fecha.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import DATA_ROOT, DEFAULT_DB_PATH
from .statsbomb_io import list_competitions


def select_matchday_subset(
    matches: list[dict[str, Any]], pct: float
) -> list[dict[str, Any]]:
    """Subconjunto de partidos según % de jornadas (o de partidos si no hay jornadas).

    `pct` en (0, 100]. Devuelve los partidos ordenados por fecha.
    """
    pct = max(0.0, min(100.0, pct))
    weeks = [m.get("match_week") for m in matches]
    distinct_weeks = sorted({int(w) for w in weeks if w is not None})
    has_matchdays = len(distinct_weeks) > 1 and sum(w is not None for w in weeks) >= 0.5 * len(matches)

    if has_matchdays:
        k = max(1, math.ceil(pct / 100.0 * len(distinct_weeks)))
        keep = set(distinct_weeks[:k])
        selected = [m for m in matches if m.get("match_week") and int(m["match_week"]) in keep]
    else:
        ordered = sorted(matches, key=lambda m: m.get("match_date") or "")
        k = max(1, math.ceil(pct / 100.0 * len(ordered)))
        selected = ordered[:k]

    return sorted(selected, key=lambda m: (m.get("match_date") or "", m.get("match_id")))


# --- Resolución por nombre (para modo con argumentos) ------------------------
def find_competition(name: str, data_root: Path = DATA_ROOT) -> dict[str, Any] | None:
    name_l = name.strip().lower()
    for comp in list_competitions(data_root):
        if str(comp["competition_name"]).lower() == name_l:
            return comp
    return None


def find_season(competition: dict[str, Any], name: str) -> dict[str, Any] | None:
    name_l = name.strip().lower()
    for season in competition["seasons"]:
        if str(season["season_name"]).lower() == name_l:
            return season
    return None


# --- Modo interactivo --------------------------------------------------------
def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    ans = input(f"{prompt}{suffix}: ").strip()
    return ans or (default or "")


def _choose(options: list[str], title: str) -> int:
    print(f"\n{title}")
    for i, label in enumerate(options, start=1):
        print(f"  {i:>3}. {label}")
    while True:
        raw = input("Elige un número: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("  Opción no válida, inténtalo de nuevo.")


def interactive_selection(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Diálogo por consola. Devuelve competición, temporada, partidos y ruta de BD."""
    competitions = [c for c in list_competitions(data_root) if any(s["available"] for s in c["seasons"])]

    labels = [
        f'{c["competition_name"]}  ({c["country_name"]}, {c["competition_gender"]})'
        for c in competitions
    ]
    comp = competitions[_choose(labels, "Competiciones disponibles:")]

    seasons = [s for s in comp["seasons"] if s["available"]]
    s_labels = [str(s["season_name"]) for s in seasons]
    season = seasons[_choose(s_labels, f'Temporadas de {comp["competition_name"]}:')]

    from .statsbomb_io import load_matches  # import local para evitar ciclo

    matches = load_matches(comp["competition_id"], season["season_id"], data_root)
    n_weeks = len({m.get("match_week") for m in matches if m.get("match_week")})
    print(f'\n{comp["competition_name"]} — {season["season_name"]}: '
          f'{len(matches)} partidos, {n_weeks or "s/"} jornadas.')

    while True:
        pct_raw = _ask("Porcentaje de jornadas a procesar (1-100)", "100")
        try:
            pct = float(pct_raw)
            if 0 < pct <= 100:
                break
        except ValueError:
            pass
        print("  Introduce un número entre 1 y 100.")

    selected = select_matchday_subset(matches, pct)
    print(f"  -> Se procesarán {len(selected)} partidos.")

    db_path = Path(_ask("Ruta de la base de datos SQLite", str(DEFAULT_DB_PATH)))

    return {
        "competition": comp,
        "season": season,
        "matches": selected,
        "db_path": db_path,
        "pct": pct,
    }

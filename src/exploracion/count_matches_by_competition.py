"""Count matches by competition and season from the local StatsBomb open-data dump."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DATA_ROOT = Path("open-data/data")
OUTPUT_PATH = Path("outputs/csv/matches_by_competition_season.csv")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_match_counts(data_root: Path) -> pd.DataFrame:
    """Build a dataframe with one row per competition-season and match counts."""
    competitions = pd.DataFrame(read_json(data_root / "competitions.json"))

    rows: list[dict[str, Any]] = []
    matches_root = data_root / "matches"
    for _, comp in competitions.iterrows():
        competition_id = int(comp["competition_id"])
        season_id = int(comp["season_id"])
        matches_path = matches_root / str(competition_id) / f"{season_id}.json"
        if matches_path.exists():
            matches = read_json(matches_path)
            match_count = len(matches)
        else:
            match_count = 0

        rows.append(
            {
                "competition_id": competition_id,
                "competition_name": comp.get("competition_name"),
                "country_name": comp.get("country_name"),
                "season_id": season_id,
                "season_name": comp.get("season_name"),
                "match_count": match_count,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["competition_name", "season_name", "match_count"],
            ascending=[True, True, False],
            kind="stable",
        ).reset_index(drop=True)
    return result


def main() -> None:
    """Generate and print the competition-season match counts."""
    summary = build_match_counts(DATA_ROOT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    if summary.empty:
        print("No se encontraron competiciones o partidos en open-data/data.")
        return

    print(summary.to_string(index=False))
    print(f"\nCSV guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

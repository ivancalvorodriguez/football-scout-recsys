"""Minutos jugados y posiciones por jugador en un partido.

Se derivan de `lineups/<match>.json`: cada jugador trae spells `positions` con
`from`/`to` en reloj de partido continuo ("MM:SS") y `position_id`. Un jugador
puede ocupar varias posiciones (cambios); se guarda el detalle y se marca como
`primary` la de más minutos.
"""

from __future__ import annotations

from typing import Any

from .features import event_seconds, playable_events


def _parse_clock(value: str | None) -> float | None:
    """'MM:SS' -> minutos (float). None si no aplica."""
    if not value:
        return None
    mm, ss = value.split(":")
    return int(mm) + int(ss) / 60.0


def match_end_minute(events: list[dict[str, Any]]) -> float:
    """Minuto del último evento jugable (excluye la tanda de penaltis).

    Incluye el descuento: un titular que juega el partido completo puede quedar
    con ~95-100', no 90'. Es intencionado (más fiel a los minutos reales) y
    consistente entre jugadores; el per-90 aguas abajo divide por este valor.
    """
    end = 0.0
    for e in playable_events(events):
        end = max(end, event_seconds(e) / 60.0)
    return end


def player_minutes(
    lineups: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Un registro por jugador que disputó minutos, con posiciones y minutos.

    Cada registro: team_id, team_name, player_id, player_name, minutes_played,
    primary_position_id, primary_position_name, positions=[{id, name, minutes}].
    """
    end_min = match_end_minute(events)
    records: list[dict[str, Any]] = []

    for team in lineups:
        team_id = team["team_id"]
        team_name = team["team_name"]
        for player in team.get("lineup", []):
            spells = player.get("positions", []) or []
            if not spells:
                continue  # convocado pero no jugó

            minutes_by_pos: dict[int, dict[str, Any]] = {}
            total = 0.0
            for sp in spells:
                start = _parse_clock(sp.get("from")) or 0.0
                stop = _parse_clock(sp.get("to"))
                if stop is None:
                    stop = end_min  # jugó hasta el final
                played = max(0.0, stop - start)
                total += played
                pid = sp.get("position_id")
                pname = sp.get("position")
                if pid is None:
                    continue
                entry = minutes_by_pos.setdefault(
                    pid, {"position_id": pid, "position_name": pname, "minutes": 0.0}
                )
                entry["minutes"] += played

            if total <= 0.0:
                continue

            positions = sorted(
                minutes_by_pos.values(), key=lambda p: p["minutes"], reverse=True
            )
            primary = positions[0]
            records.append(
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": player["player_id"],
                    "player_name": player.get("player_nickname") or player["player_name"],
                    "minutes_played": round(total, 2),
                    "primary_position_id": primary["position_id"],
                    "primary_position_name": primary["position_name"],
                    "positions": [
                        {**p, "minutes": round(p["minutes"], 2)} for p in positions
                    ],
                }
            )
    return records

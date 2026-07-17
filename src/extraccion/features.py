"""Utilidades compartidas sobre eventos StatsBomb: geometría, pases, xT y posesiones.

Convenciones StatsBomb aplicadas de forma consistente (ver `docs/metricas_finales.md`):
- Pase completado = evento Pass **sin** `pass.outcome`.
- Acción progresiva = reduce >= 25% de la distancia restante al centro de la
  portería rival (criterio StatsBomb, no el de FBref/Opta de 10 yardas).
- Los eventos del periodo 5 (tanda de penaltis) se ignoran en todos los cálculos.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from . import config


# --- Geometría ---------------------------------------------------------------
def location(event: dict[str, Any]) -> tuple[float, float] | None:
    """Coordenada (x, y) del evento, o None si no la tiene."""
    loc = event.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None


def dist_to_goal(point: tuple[float, float]) -> float:
    """Distancia euclídea de un punto al centro de la portería rival."""
    return math.hypot(config.GOAL[0] - point[0], config.GOAL[1] - point[1])


def is_progressive(start: tuple[float, float], end: tuple[float, float]) -> bool:
    """True si la acción reduce >= 25% de la distancia restante a portería."""
    d_start = dist_to_goal(start)
    if d_start <= 0:
        return False
    return (d_start - dist_to_goal(end)) >= config.PROGRESSIVE_FRACTION * d_start


def in_penalty_area(point: tuple[float, float]) -> bool:
    """True si el punto está en el área de penalti rival."""
    x, y = point
    return x >= config.PEN_AREA_X and config.PEN_AREA_Y_MIN <= y <= config.PEN_AREA_Y_MAX


def in_final_third(point: tuple[float, float]) -> bool:
    return point[0] >= config.FINAL_THIRD_X


# --- Toques ------------------------------------------------------------------
def is_touch(event: dict[str, Any]) -> bool:
    """True si el evento cuenta como "toque" del jugador/equipo que lo ejecuta.

    Criterio unico para `touches_in_att_pen_area` (jugador) y para
    `touches`/`field_tilt` (equipo): un `Ball Receipt*` fallido no es un toque.
    """
    etype = event["type"]["name"]
    if etype not in config.TOUCH_TYPES:
        return False
    if etype == "Ball Receipt*":
        outcome = event.get("ball_receipt", {}).get("outcome", {})
        if outcome.get("name") == "Incomplete":
            return False
    return True


# --- Pases -------------------------------------------------------------------
def is_pass(event: dict[str, Any]) -> bool:
    return event["type"]["name"] == "Pass"


def pass_completed(event: dict[str, Any]) -> bool:
    """Pase completado: no hay `pass.outcome` (Incomplete/Out/Offside...)."""
    return is_pass(event) and "outcome" not in event.get("pass", {})


def is_set_piece_pass(event: dict[str, Any]) -> bool:
    ptype = event.get("pass", {}).get("type", {})
    return ptype.get("name") in config.SET_PIECE_PASS_TYPES


def pass_end(event: dict[str, Any]) -> tuple[float, float] | None:
    end = event.get("pass", {}).get("end_location")
    if isinstance(end, list) and len(end) >= 2:
        return float(end[0]), float(end[1])
    return None


def carry_end(event: dict[str, Any]) -> tuple[float, float] | None:
    end = event.get("carry", {}).get("end_location")
    if isinstance(end, list) and len(end) >= 2:
        return float(end[0]), float(end[1])
    return None


# --- Expected Threat ---------------------------------------------------------
def _xt_bin(point: tuple[float, float]) -> tuple[int, int]:
    x, y = point
    xb = min(config.XT_COLS - 1, max(0, int(x / config.FIELD_LENGTH * config.XT_COLS)))
    yb = min(config.XT_ROWS - 1, max(0, int(y / config.FIELD_WIDTH * config.XT_ROWS)))
    return xb, yb


def xt_at(point: tuple[float, float]) -> float:
    xb, yb = _xt_bin(point)
    return config.XT_GRID[yb][xb]


def xt_delta(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Amenaza añadida por mover el balón de `start` a `end`."""
    return xt_at(end) - xt_at(start)


# --- Aéreos ------------------------------------------------------------------
def is_aerial_won(event: dict[str, Any]) -> bool:
    """True si el evento lleva el flag `aerial_won`.

    StatsBomb no emite un evento "duelo aereo ganado": marca `aerial_won` en la
    accion con la que el ganador resuelve el salto, y esa accion solo puede ser
    de estos cuatro tipos (ver `AERIAL_WON_KEYS`). El perdedor si recibe un
    evento propio (`Duel` / "Aerial Lost", ver `is_aerial_lost`).
    """
    for key in config.AERIAL_WON_KEYS:
        sub = event.get(key)
        if isinstance(sub, dict) and sub.get("aerial_won"):
            return True
    return False


def is_aerial_lost(event: dict[str, Any]) -> bool:
    return (
        event["type"]["name"] == "Duel"
        and event.get("duel", {}).get("type", {}).get("name") == "Aerial Lost"
    )


# --- Tiempo y periodos -------------------------------------------------------
def event_seconds(event: dict[str, Any]) -> float:
    """Segundos de reloj de partido (minute*60 + second).

    `minute` en StatsBomb es acumulado desde el inicio del partido (no se
    reinicia en cada periodo), asi que el valor es directamente comparable entre
    eventos del mismo partido e incluye el descuento.
    """
    return event.get("minute", 0) * 60 + event.get("second", 0)


def playable_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descarta el periodo 5 (tanda de penaltis)."""
    return [e for e in events if e.get("period", 1) != 5]


# --- Posesiones (secuencias) -------------------------------------------------
class Possession:
    """Agrupa los eventos de un mismo valor `possession`.

    `team_events` son solo los del equipo dueño de la posesión (`possession_team`),
    que es lo relevante para métricas de secuencia (pases por secuencia, 10+, etc.).
    """

    def __init__(self, possession_id: int, events: list[dict[str, Any]]):
        self.id = possession_id
        self.events = events
        self.team_name = events[0]["possession_team"]["name"] if events else None
        self.team_id = events[0]["possession_team"]["id"] if events else None
        self.team_events = [e for e in events if e["team"]["name"] == self.team_name]

    @property
    def play_pattern(self) -> str | None:
        return self.events[0]["play_pattern"]["name"] if self.events else None

    def completed_passes(self) -> list[dict[str, Any]]:
        return [e for e in self.team_events if pass_completed(e)]

    def n_completed_passes(self) -> int:
        return len(self.completed_passes())

    def start_location(self) -> tuple[float, float] | None:
        for e in self.team_events:
            loc = location(e)
            if loc is not None:
                return loc
        return None


def group_possessions(events: list[dict[str, Any]]) -> list[Possession]:
    """Agrupa los eventos jugables por `possession`, en orden de aparición."""
    order: list[int] = []
    buckets: dict[int, list[dict[str, Any]]] = {}
    for e in playable_events(events):
        pid = e.get("possession")
        if pid is None:
            continue
        if pid not in buckets:
            buckets[pid] = []
            order.append(pid)
        buckets[pid].append(e)
    return [Possession(pid, buckets[pid]) for pid in order]


def is_open_play_possession(possession: Possession) -> bool:
    """Posesión de juego abierto (no arranca de córner/falta/saque)."""
    return possession.play_pattern in ("Regular Play", "From Counter", "From Kick Off")

"""Métricas por (jugador, partido) a partir de los eventos de StatsBomb.

Se calculan en **crudo** (conteos y sumas) más los ratios que no dependen de los
minutos (porcentajes). La normalización por 90' y la estandarización se hacen
aguas abajo, sobre la matriz de features (por eso se persiste también
`minutes_played`). Fases cubiertas: Progresión, Creación, Finalización, Duelos,
Juego Aéreo y Defensa. La fase Perfil/Valor global (VAEP/OBV/...) queda fuera de
alcance por requerir modelos/coeficientes no públicos.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import config, features as F


def _new_row() -> dict[str, Any]:
    keys = [
        # Progresión
        "passes", "passes_completed", "progressive_passes", "progressive_carries",
        "passes_into_final_third", "xt",
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
        "interceptions", "ball_recoveries", "blocks", "clearances",
        "pressures", "pressure_regains", "counterpressures",
    ]
    return {k: 0.0 for k in keys}


def compute_player_stats(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Devuelve {player_id: fila de métricas} para un partido."""
    events = F.playable_events(events)
    rows: dict[int, dict[str, Any]] = defaultdict(_new_row)
    by_id: dict[str, dict[str, Any]] = {e["id"]: e for e in events}

    def row_for(event: dict[str, Any]) -> dict[str, Any] | None:
        player = event.get("player")
        return rows[player["id"]] if player else None

    # --- Pasada principal ----------------------------------------------------
    for e in events:
        etype = e["type"]["name"]
        r = row_for(e)
        if r is None:
            continue
        loc = F.location(e)

        if etype == "Pass":
            p = e["pass"]
            r["passes"] += 1
            completed = "outcome" not in p
            if completed:
                r["passes_completed"] += 1
            end = F.pass_end(e)
            set_piece = F.is_set_piece_pass(e)
            if completed and loc and end:
                if not set_piece:
                    if F.is_progressive(loc, end):
                        r["progressive_passes"] += 1
                    if not F.in_final_third(loc) and F.in_final_third(end):
                        r["passes_into_final_third"] += 1
                    if (not F.in_penalty_area(loc)) and F.in_penalty_area(end):
                        r["passes_into_penalty_area"] += 1
                    r["xt"] += max(0.0, F.xt_delta(loc, end))
                    # Deep completion: pase (no centro) que acaba cerca de portería.
                    if not p.get("cross") and F.dist_to_goal(end) <= config.DEEP_COMPLETION_RADIUS:
                        r["deep_completions"] += 1

        elif etype == "Carry":
            end = F.carry_end(e)
            if loc and end:
                if F.is_progressive(loc, end):
                    r["progressive_carries"] += 1
                r["xt"] += max(0.0, F.xt_delta(loc, end))

        elif etype == "Shot":
            s = e["shot"]
            is_pen = s.get("type", {}).get("name") == "Penalty"
            xg = float(s.get("statsbomb_xg", 0.0) or 0.0)
            outcome = s.get("outcome", {}).get("name")
            r["shots"] += 1
            r["xg"] += xg
            if outcome == "Goal":
                r["goals"] += 1
            if not is_pen:
                r["np_shots"] += 1
                r["npxg"] += xg
                if outcome == "Goal":
                    r["np_goals"] += 1
                # `shots_on_target` tambien va aqui dentro: excluye penaltis pese
                # a no llamarse `np_*`. Es el criterio de StatsBomb/FBref y lo que
                # hace homogeneo el `sot_pct` = shots_on_target / np_shots.
                if outcome in config.SHOT_ON_TARGET_OUTCOMES:
                    r["shots_on_target"] += 1

        elif etype == "Dribble":
            r["take_ons"] += 1
            if e.get("dribble", {}).get("outcome", {}).get("name") == "Complete":
                r["take_ons_won"] += 1

        elif etype == "Foul Won":
            r["fouls_drawn"] += 1

        elif etype == "Dispossessed":
            r["dispossessed"] += 1

        elif etype == "Duel":
            dtype = e.get("duel", {}).get("type", {}).get("name")
            outcome = e.get("duel", {}).get("outcome", {}).get("name")
            if dtype == "Tackle":
                r["tackles"] += 1
                r["duels_total"] += 1
                if outcome in config.TACKLE_WON_OUTCOMES:
                    r["tackles_won"] += 1
                    r["duels_won"] += 1

        elif etype == "Dribbled Past":
            r["dribbled_past"] += 1

        elif etype == "Interception":
            r["interceptions"] += 1

        elif etype == "Ball Recovery":
            # Recuperación fallida lleva ball_recovery.recovery_failure=True.
            if not e.get("ball_recovery", {}).get("recovery_failure"):
                r["ball_recoveries"] += 1

        elif etype == "Block":
            r["blocks"] += 1

        elif etype == "Clearance":
            r["clearances"] += 1

        elif etype == "Pressure":
            r["pressures"] += 1

        elif etype == "50/50":
            # Balón dividido (nadie controla): se agrega a los duelos globales.
            fo = e.get("50_50", {}).get("outcome", {}).get("name")
            r["duels_total"] += 1
            if fo in ("Won", "Success To Team"):
                r["duels_won"] += 1

        # Toques en el área rival (fase Finalización); mismo criterio de "toque"
        # que a nivel equipo (`F.is_touch`).
        if F.is_touch(e) and loc and F.in_penalty_area(loc):
            r["touches_in_att_pen_area"] += 1

        # Aéreos (flag disperso en varios tipos) y contadores por zona.
        if F.is_aerial_won(e):
            r["aerial_won"] += 1
            r["duels_total"] += 1
            r["duels_won"] += 1
            if loc and loc[0] >= config.FIELD_LENGTH / 2:
                r["aerial_won_off"] += 1
            else:
                r["aerial_won_def"] += 1
        if F.is_aerial_lost(e):
            r["aerial_lost"] += 1
            r["duels_total"] += 1

        # Counterpress (flag en acciones defensivas).
        if e.get("counterpress"):
            r["counterpressures"] += 1

    # --- xA y SCA (requieren enlazar pase clave -> tiro) ---------------------
    _add_xa_and_sca(events, by_id, rows)

    # --- Pressure regains (recuperación del equipo <= 5 s tras la presión) ---
    _add_pressure_regains(events, rows)

    return {pid: _finalize(r) for pid, r in rows.items()}


def _add_xa_and_sca(
    events: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    rows: dict[int, dict[str, Any]],
) -> None:
    # Índice de posición de cada evento para acotar la secuencia previa al tiro.
    index = {e["id"]: i for i, e in enumerate(events)}
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        xg = float(e["shot"].get("statsbomb_xg", 0.0) or 0.0)
        # xA: el pase clave hereda el xG del tiro que asiste.
        kp_id = e["shot"].get("key_pass_id")
        if kp_id and kp_id in by_id:
            passer = by_id[kp_id].get("player")
            if passer:
                rows[passer["id"]]["xa"] += xg
        # SCA: las 2 últimas acciones ofensivas del equipo antes del tiro.
        _credit_sca(e, events, index, rows)


def _credit_sca(
    shot: dict[str, Any],
    events: list[dict[str, Any]],
    index: dict[str, int],
    rows: dict[int, dict[str, Any]],
) -> None:
    pos = shot.get("possession")
    team = shot["team"]["name"]
    i = index[shot["id"]]
    n = 0
    for j in range(i - 1, -1, -1):
        prev = events[j]
        if prev.get("possession") != pos:
            break
        if prev["team"]["name"] != team:
            continue
        if prev["type"]["name"] not in config.SCA_ACTION_TYPES:
            continue
        if prev["type"]["name"] == "Pass" and "outcome" in prev.get("pass", {}):
            continue  # solo pases completados
        if prev["type"]["name"] == "Dribble" and (
            prev.get("dribble", {}).get("outcome", {}).get("name") != "Complete"
        ):
            continue
        # Se acreditan las 2 acciones ofensivas previas al tiro (criterio FBref),
        # a quien ejecuta cada una; una misma persona puede sumar las dos. Esto es
        # coherente con `team_stats._add_team_sca_gca`, que cuenta acciones (no
        # ejecutantes distintos): así el SCA de equipo == suma del de sus jugadores.
        player = prev.get("player")
        if player:
            rows[player["id"]]["sca"] += 1
        n += 1
        if n >= config.SCA_MAX_ACTIONS:
            break


def _add_pressure_regains(
    events: list[dict[str, Any]], rows: dict[int, dict[str, Any]]
) -> None:
    on_ball = {"Pass", "Carry", "Ball Recovery", "Interception", "Dribble", "Shot"}
    for i, e in enumerate(events):
        if e["type"]["name"] != "Pressure":
            continue
        player = e.get("player")
        if not player:
            continue
        team = e["team"]["name"]
        period = e.get("period")
        t0 = F.event_seconds(e)
        for j in range(i + 1, len(events)):
            nxt = events[j]
            if nxt.get("period") != period:
                break
            if F.event_seconds(nxt) - t0 > config.REGAIN_WINDOW_SECONDS:
                break
            if nxt["team"]["name"] == team and nxt["type"]["name"] in on_ball:
                rows[player["id"]]["pressure_regains"] += 1
                break


def _finalize(r: dict[str, Any]) -> dict[str, Any]:
    """Solo redondea sumas de modelos. Los ratios (pass %, sot %, tackle %,
    duels %, aerial %, npxg/sh, np:G-npxG...) NO se calculan aquí: son
    derivables de los conteos y se computan en la capa de features, junto con la
    normalización por 90', antes del modelo de similitud.
    """
    for k in ("xg", "npxg", "xa", "xt"):
        r[k] = round(r[k], 5)
    return r

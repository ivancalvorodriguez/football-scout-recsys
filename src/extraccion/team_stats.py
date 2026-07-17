"""Métricas por (equipo, partido) a partir de los eventos de StatsBomb.

Cubre las fases de equipo: Posesión/Elaboración, Progresión, Creación/Amenaza,
Finalización, Presión/Recuperación, Dominio territorial y Perfil/Estilo. Las
métricas de secuencia (10+ pases, build-up, direct speed/attacks, amplitud,
distancia de inicio, pases por secuencia) se derivan de las posesiones
(`possession` + `possession_team`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import config, features as F


def _new_row() -> dict[str, Any]:
    keys = [
        "touches", "final_third_touches",
        "passes", "passes_completed",
        "progressive_passes", "passes_into_final_third", "xt",
        "sca", "gca", "passes_into_penalty_area", "open_play_xg",
        "shots", "np_shots", "goals", "shots_on_target", "xg", "npxg",
        "high_turnovers", "pressures", "counterpressures",
        # PPDA (numerador/denominador se acumulan y se dividen al final)
        "ppda_opp_passes", "ppda_def_actions",
        # Secuencias (acumuladores para promedios)
        "ten_plus_pass_sequences", "build_up_attacks", "direct_attacks",
        "seq_count", "seq_passes_sum", "seq_width_sum", "seq_start_dist_sum",
        "seq_speed_sum", "seq_speed_count",
    ]
    return {k: 0.0 for k in keys}


def compute_team_stats(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Devuelve {team_id: fila de métricas} para un partido (dos equipos)."""
    events = F.playable_events(events)
    teams: dict[int, str] = {}
    for e in events:
        t = e.get("team")
        if t:
            teams[t["id"]] = t["name"]

    rows: dict[int, dict[str, Any]] = {tid: _new_row() for tid in teams}
    name_to_id = {name: tid for tid, name in teams.items()}
    index = {e["id"]: i for i, e in enumerate(events)}

    # --- Contadores por evento ----------------------------------------------
    for e in events:
        team = e.get("team")
        if not team:
            continue
        tid = team["id"]
        r = rows[tid]
        etype = e["type"]["name"]
        loc = F.location(e)

        if F.is_touch(e):
            r["touches"] += 1
            if loc and F.in_final_third(loc):
                r["final_third_touches"] += 1

        if etype == "Pass":
            p = e["pass"]
            r["passes"] += 1
            completed = "outcome" not in p
            if completed:
                r["passes_completed"] += 1
            end = F.pass_end(e)
            if completed and loc and end and not F.is_set_piece_pass(e):
                if F.is_progressive(loc, end):
                    r["progressive_passes"] += 1
                if not F.in_final_third(loc) and F.in_final_third(end):
                    r["passes_into_final_third"] += 1
                if (not F.in_penalty_area(loc)) and F.in_penalty_area(end):
                    r["passes_into_penalty_area"] += 1
                r["xt"] += max(0.0, F.xt_delta(loc, end))
            # PPDA (numerador): pases del equipo en su propio 60% (x <= 72).
            if loc and loc[0] <= 72.0:
                r["ppda_opp_passes"] += 1

        elif etype == "Carry":
            end = F.carry_end(e)
            if loc and end:
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
                # Sin penaltis, igual que `player_stats`: `sot_pct` se calcula
                # como shots_on_target/np_shots y ambos deben excluirlos.
                if outcome in config.SHOT_ON_TARGET_OUTCOMES:
                    r["shots_on_target"] += 1
            if s.get("type", {}).get("name") == "Open Play":
                r["open_play_xg"] += xg

        elif etype == "Pressure":
            r["pressures"] += 1

        elif etype in ("Ball Recovery", "Interception"):
            if etype == "Ball Recovery" and e.get("ball_recovery", {}).get("recovery_failure"):
                pass
            elif loc and loc[0] >= config.HIGH_TURNOVER_X:
                r["high_turnovers"] += 1

        if e.get("counterpress"):
            r["counterpressures"] += 1

        # PPDA (denominador): acciones defensivas en el 60% de ataque (x >= 48).
        if loc and loc[0] >= 48.0:
            if etype == "Interception" or etype == "Foul Committed":
                r["ppda_def_actions"] += 1
            elif etype == "Duel" and e.get("duel", {}).get("type", {}).get("name") == "Tackle":
                r["ppda_def_actions"] += 1

    # --- SCA / GCA (2 acciones ofensivas previas a cada tiro/gol) ------------
    _add_team_sca_gca(events, index, rows)

    # --- Métricas de secuencia ----------------------------------------------
    _add_sequence_metrics(events, rows)

    # --- Posesión y ratios finales ------------------------------------------
    total_touches = sum(r["touches"] for r in rows.values()) or 1
    total_ft_touches = sum(r["final_third_touches"] for r in rows.values()) or 1
    # PPDA = pases del RIVAL en su propio 60% / acciones defensivas propias altas.
    # `ppda_opp_passes` de cada fila son sus pases en su 60%; para el equipo T el
    # numerador es el del rival.
    tids = list(rows)
    opp_passes = {
        tid: sum(rows[o]["ppda_opp_passes"] for o in tids if o != tid) for tid in tids
    }
    for tid, r in rows.items():
        # possession_pct, field_tilt, ppda y las medias por secuencia se guardan
        # como valor único: sus insumos son intermedios (no métricas por sí
        # mismos), así que no se reducen a conteos crudos. Los ratios simples
        # (pass %, sot %, xg/shot, G-xG) NO se guardan: son derivables de los
        # conteos y se calculan en la capa de features previa a la similitud.
        r["possession_pct"] = r["touches"] / total_touches
        r["field_tilt"] = r["final_third_touches"] / total_ft_touches
        r["ppda"] = (
            opp_passes[tid] / r["ppda_def_actions"] if r["ppda_def_actions"] else None
        )
        seq = r["seq_count"] or 1
        r["passes_per_sequence"] = r["seq_passes_sum"] / seq
        r["absolute_width"] = round(r["seq_width_sum"] / seq * config.YARD_TO_M, 3)
        r["sequence_start_distance"] = round(
            r["seq_start_dist_sum"] / seq * config.YARD_TO_M, 3
        )
        r["direct_speed"] = round(
            (r["seq_speed_sum"] / r["seq_speed_count"]) if r["seq_speed_count"] else 0.0, 4
        )
        for k in ("xg", "npxg", "xt", "open_play_xg"):
            r[k] = round(r[k], 5)

    return rows


def _add_team_sca_gca(
    events: list[dict[str, Any]], index: dict[str, int], rows: dict[int, dict[str, Any]]
) -> None:
    """SCA/GCA de equipo. Misma definicion que `player_stats._credit_sca` (misma
    lista `config.SCA_ACTION_TYPES` y mismo tope), para que el SCA de equipo sea
    igual a la suma del de sus jugadores.
    """
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        team = e["team"]["name"]
        tid = e["team"]["id"]
        pos = e.get("possession")
        i = index[e["id"]]
        n = 0
        for j in range(i - 1, -1, -1):
            prev = events[j]
            if prev.get("possession") != pos:
                break
            if prev["team"]["name"] != team:
                continue
            pt = prev["type"]["name"]
            if pt not in config.SCA_ACTION_TYPES:
                continue
            if pt == "Pass" and "outcome" in prev.get("pass", {}):
                continue
            if pt == "Dribble" and prev.get("dribble", {}).get("outcome", {}).get("name") != "Complete":
                continue
            n += 1
            if n >= config.SCA_MAX_ACTIONS:
                break
        rows[tid]["sca"] += n
        if e["shot"].get("outcome", {}).get("name") == "Goal":
            rows[tid]["gca"] += n


def _max_x_reached(team_events: list[dict[str, Any]]) -> float:
    best = 0.0
    for e in team_events:
        loc = F.location(e)
        if loc:
            best = max(best, loc[0])
        for end in (F.pass_end(e), F.carry_end(e)):
            if end:
                best = max(best, end[0])
    return best


def _add_sequence_metrics(
    events: list[dict[str, Any]], rows: dict[int, dict[str, Any]]
) -> None:
    for poss in F.group_possessions(events):
        if poss.team_id is None or not F.is_open_play_possession(poss):
            continue
        if poss.team_id not in rows:
            continue
        r = rows[poss.team_id]
        te = poss.team_events
        start = poss.start_location()
        if start is None:
            continue

        n_passes = poss.n_completed_passes()
        ends_in_shot = any(e["type"]["name"] == "Shot" for e in te)
        touch_in_box = any(
            F.location(e) and F.in_penalty_area(F.location(e)) for e in te
        )
        r["seq_count"] += 1
        r["seq_passes_sum"] += n_passes
        r["seq_start_dist_sum"] += start[0]
        r["seq_width_sum"] += 2.0 * max(
            (abs(F.location(e)[1] - config.FIELD_WIDTH / 2) for e in te if F.location(e)),
            default=0.0,
        )

        if n_passes >= 10:
            r["ten_plus_pass_sequences"] += 1
            if ends_in_shot or touch_in_box:
                r["build_up_attacks"] += 1

        # Direct speed y direct attacks
        max_x = _max_x_reached(te)
        progress = max_x - start[0]
        t0 = F.event_seconds(te[0])
        t1 = F.event_seconds(te[-1])
        duration = t1 - t0
        if duration > 0 and progress > 0:
            r["seq_speed_sum"] += (progress * config.YARD_TO_M) / duration
            r["seq_speed_count"] += 1
        if start[0] < config.FIELD_LENGTH / 2 and progress >= 0.5 * (
            config.FIELD_LENGTH - start[0]
        ) and (ends_in_shot or touch_in_box):
            r["direct_attacks"] += 1

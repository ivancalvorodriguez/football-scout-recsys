"""Constructores de datos sinteticos: eventos, alineaciones, arbol open-data y BD.

Ninguna prueba toca `open-data/` (dataset externo, en `.gitignore`, ausente en un
checkout limpio): todo se fabrica aqui. Los eventos siguen el formato real de
StatsBomb verificado en los datos (`type.name`, `team`, `possession`,
`possession_team`, `play_pattern`, `location`, sub-objeto por tipo) pero solo con
los campos que el extractor llega a leer: anadir el resto seria ruido.

Convenciones que respetan los datos reales y de las que dependen las pruebas:
- Coordenadas 120x80, el equipo que ejecuta ataca hacia x=120.
- Pase completado = evento Pass SIN `pass.outcome`.
- `minute` es acumulado desde el inicio del partido (no se reinicia por periodo).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.extraccion import database as db

# --- Identidades reutilizables ----------------------------------------------
EQUIPO_A: dict[str, Any] = {"id": 1, "name": "Equipo A"}
EQUIPO_B: dict[str, Any] = {"id": 2, "name": "Equipo B"}

_contador_ids = itertools.count(1)


def nuevo_id() -> str:
    """Id unico de evento. StatsBomb usa UUID; aqui solo importa que no se repita
    dentro de un partido (`player_stats` y `team_stats` indexan los eventos por id).
    """
    return f"ev-{next(_contador_ids):06d}"


def jugador(player_id: int, nombre: str | None = None) -> dict[str, Any]:
    return {"id": player_id, "name": nombre or f"Jugador {player_id}"}


# --- Eventos -----------------------------------------------------------------
def evento(
    tipo: str,
    *,
    equipo: dict[str, Any] | None = None,
    jug: int | dict[str, Any] | None = None,
    loc: tuple[float, float] | None = None,
    minute: int = 0,
    second: int = 0,
    period: int = 1,
    possession: int = 1,
    possession_team: dict[str, Any] | None = None,
    play_pattern: str = "Regular Play",
    eid: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Evento StatsBomb minimo. `extra` inyecta el sub-objeto del tipo (`pass`...)."""
    equipo = equipo if equipo is not None else EQUIPO_A
    e: dict[str, Any] = {
        "id": eid or nuevo_id(),
        "type": {"name": tipo},
        "team": dict(equipo),
        "possession": possession,
        "possession_team": dict(
            possession_team if possession_team is not None else equipo
        ),
        "play_pattern": {"name": play_pattern},
        "period": period,
        "minute": minute,
        "second": second,
    }
    if jug is not None:
        e["player"] = jug if isinstance(jug, dict) else jugador(jug)
    if loc is not None:
        e["location"] = [float(loc[0]), float(loc[1])]
    e.update(extra)
    return e


def pase(
    desde: tuple[float, float],
    hasta: tuple[float, float],
    *,
    completado: bool = True,
    jug: int | dict[str, Any] | None = 10,
    tipo_pase: str | None = None,
    cross: bool = False,
    aereo_ganado: bool = False,
    **kw: Any,
) -> dict[str, Any]:
    sub: dict[str, Any] = {"end_location": [float(hasta[0]), float(hasta[1])]}
    if not completado:
        sub["outcome"] = {"name": "Incomplete"}
    if tipo_pase is not None:
        sub["type"] = {"name": tipo_pase}
    if cross:
        sub["cross"] = True
    if aereo_ganado:
        sub["aerial_won"] = True
    return evento("Pass", loc=desde, jug=jug, **{"pass": sub}, **kw)


def tiro(
    desde: tuple[float, float],
    *,
    xg: float = 0.1,
    resultado: str = "Saved",
    tipo_tiro: str = "Open Play",
    jug: int | dict[str, Any] | None = 10,
    key_pass_id: str | None = None,
    **kw: Any,
) -> dict[str, Any]:
    sub: dict[str, Any] = {
        "statsbomb_xg": xg,
        "outcome": {"name": resultado},
        "type": {"name": tipo_tiro},
        "end_location": [120.0, 40.0, 1.0],
    }
    if key_pass_id is not None:
        sub["key_pass_id"] = key_pass_id
    return evento("Shot", loc=desde, jug=jug, **{"shot": sub}, **kw)


def conduccion(
    desde: tuple[float, float],
    hasta: tuple[float, float],
    *,
    jug: int | dict[str, Any] | None = 10,
    **kw: Any,
) -> dict[str, Any]:
    return evento(
        "Carry", loc=desde, jug=jug, **{"carry": {"end_location": [float(hasta[0]), float(hasta[1])]}}, **kw
    )


def regate(
    desde: tuple[float, float],
    *,
    completado: bool = True,
    jug: int | dict[str, Any] | None = 10,
    **kw: Any,
) -> dict[str, Any]:
    outcome = "Complete" if completado else "Incomplete"
    return evento("Dribble", loc=desde, jug=jug, dribble={"outcome": {"name": outcome}}, **kw)


def presion(
    desde: tuple[float, float], *, jug: int | dict[str, Any] | None = 10, **kw: Any
) -> dict[str, Any]:
    return evento("Pressure", loc=desde, jug=jug, **kw)


def duelo_tackle(
    desde: tuple[float, float],
    *,
    resultado: str = "Won",
    jug: int | dict[str, Any] | None = 10,
    **kw: Any,
) -> dict[str, Any]:
    return evento(
        "Duel",
        loc=desde,
        jug=jug,
        duel={"type": {"name": "Tackle"}, "outcome": {"name": resultado}},
        **kw,
    )


def duelo_aereo_perdido(
    desde: tuple[float, float], *, jug: int | dict[str, Any] | None = 10, **kw: Any
) -> dict[str, Any]:
    """StatsBomb solo emite evento propio para el PERDEDOR del salto."""
    return evento("Duel", loc=desde, jug=jug, duel={"type": {"name": "Aerial Lost"}}, **kw)


def recepcion(
    desde: tuple[float, float],
    *,
    completada: bool = True,
    jug: int | dict[str, Any] | None = 10,
    **kw: Any,
) -> dict[str, Any]:
    sub: dict[str, Any] = {}
    if not completada:
        sub["outcome"] = {"name": "Incomplete"}
    return evento("Ball Receipt*", loc=desde, jug=jug, ball_receipt=sub, **kw)


def recuperacion(
    desde: tuple[float, float],
    *,
    fallida: bool = False,
    jug: int | dict[str, Any] | None = 10,
    **kw: Any,
) -> dict[str, Any]:
    sub: dict[str, Any] = {}
    if fallida:
        sub["recovery_failure"] = True
    return evento("Ball Recovery", loc=desde, jug=jug, ball_recovery=sub, **kw)


# --- Alineaciones ------------------------------------------------------------
def spell(
    position_id: int, position: str, desde: str = "00:00", hasta: str | None = None
) -> dict[str, Any]:
    """Tramo en una posicion. `hasta=None` = jugo hasta el final del partido."""
    return {"position_id": position_id, "position": position, "from": desde, "to": hasta}


def lineup_jugador(
    player_id: int,
    nombre: str,
    spells: list[dict[str, Any]],
    *,
    apodo: str | None = None,
) -> dict[str, Any]:
    j: dict[str, Any] = {
        "player_id": player_id,
        "player_name": nombre,
        "positions": spells,
    }
    if apodo is not None:
        j["player_nickname"] = apodo
    return j


def lineup_equipo(
    team_id: int, team_name: str, jugadores: list[dict[str, Any]]
) -> dict[str, Any]:
    return {"team_id": team_id, "team_name": team_name, "lineup": jugadores}


# --- Partidos y competiciones ------------------------------------------------
def partido(
    match_id: int,
    *,
    local: dict[str, Any] | None = None,
    visitante: dict[str, Any] | None = None,
    fecha: str = "2020-01-01",
    jornada: int | None = 1,
    goles_local: int = 1,
    goles_visitante: int = 0,
    fase: str = "Regular Season",
) -> dict[str, Any]:
    local = local if local is not None else EQUIPO_A
    visitante = visitante if visitante is not None else EQUIPO_B
    return {
        "match_id": match_id,
        "match_date": fecha,
        "match_week": jornada,
        "competition_stage": {"name": fase},
        "home_team": {"home_team_id": local["id"], "home_team_name": local["name"]},
        "away_team": {"away_team_id": visitante["id"], "away_team_name": visitante["name"]},
        "home_score": goles_local,
        "away_score": goles_visitante,
    }


def fila_competicion(
    competition_id: int,
    season_id: int,
    *,
    competition_name: str = "Liga Sintetica",
    season_name: str = "2019/2020",
    country_name: str = "Pais Sintetico",
    gender: str = "male",
) -> dict[str, Any]:
    return {
        "competition_id": competition_id,
        "season_id": season_id,
        "competition_name": competition_name,
        "season_name": season_name,
        "country_name": country_name,
        "competition_gender": gender,
    }


def escribir_open_data(
    raiz: Path,
    *,
    competiciones: list[dict[str, Any]],
    partidos: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
    eventos: dict[int, list[dict[str, Any]]] | None = None,
    alineaciones: dict[int, list[dict[str, Any]]] | None = None,
) -> Path:
    """Escribe un arbol `open-data/data` sintetico y devuelve su raiz.

    `partidos` se indexa por (competition_id, season_id); `eventos`/`alineaciones`
    por match_id, replicando el layout real que lee `statsbomb_io`.
    """
    raiz = Path(raiz)
    (raiz / "matches").mkdir(parents=True, exist_ok=True)
    (raiz / "events").mkdir(parents=True, exist_ok=True)
    (raiz / "lineups").mkdir(parents=True, exist_ok=True)

    (raiz / "competitions.json").write_text(
        json.dumps(competiciones, ensure_ascii=False), encoding="utf-8"
    )
    for (cid, sid), ms in (partidos or {}).items():
        destino = raiz / "matches" / str(cid)
        destino.mkdir(parents=True, exist_ok=True)
        (destino / f"{sid}.json").write_text(json.dumps(ms, ensure_ascii=False), encoding="utf-8")
    for mid, evs in (eventos or {}).items():
        (raiz / "events" / f"{mid}.json").write_text(
            json.dumps(evs, ensure_ascii=False), encoding="utf-8"
        )
    for mid, lus in (alineaciones or {}).items():
        (raiz / "lineups" / f"{mid}.json").write_text(
            json.dumps(lus, ensure_ascii=False), encoding="utf-8"
        )
    return raiz


def partido_minimo(
    match_id: int = 7001,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Un partido completo y coherente: (match, eventos, alineaciones).

    Equipo A: jugadores 10 y 11 (con eventos) y 12 (portero, sin ningun evento:
    ejercita la ruta de metricas vacias del extractor). Equipo B: 20 y 21. Hay
    acciones de cada fase para que ninguna metrica salga toda a cero.

    La coherencia importa: cada evento lo ejecuta un jugador que esta en la
    alineacion de ese equipo. Un fixture con un jugador de B presionando como si
    fuera de A haria pasar pruebas sobre datos imposibles.
    """
    eventos = [
        # Posesion 1 (Equipo A): pase progresivo -> conduccion -> pase clave -> gol.
        pase((20.0, 40.0), (60.0, 40.0), jug=10, minute=1, second=0),
        conduccion((60.0, 40.0), (85.0, 40.0), jug=11, minute=1, second=5),
        pase((85.0, 40.0), (110.0, 40.0), jug=11, eid="kp-1", minute=1, second=10),
        tiro((110.0, 40.0), xg=0.4, resultado="Goal", jug=10, key_pass_id="kp-1",
             minute=1, second=12),
        # Posesion 2 (de B): A presiona y recupera dentro de la ventana de 5 s.
        presion((50.0, 40.0), jug=11, equipo=EQUIPO_A, possession=2,
                possession_team=EQUIPO_B, minute=5, second=0),
        recuperacion((50.0, 40.0), jug=10, equipo=EQUIPO_A, possession=2,
                     possession_team=EQUIPO_B, minute=5, second=3),
        # Posesion 3 (de B): duelo terrestre y aereo.
        duelo_tackle((30.0, 40.0), resultado="Won", jug=20, equipo=EQUIPO_B,
                     possession=3, possession_team=EQUIPO_B, minute=10, second=0),
        duelo_aereo_perdido((30.0, 40.0), jug=21, equipo=EQUIPO_B, possession=3,
                            possession_team=EQUIPO_B, minute=10, second=1),
    ]
    alineaciones = [
        lineup_equipo(EQUIPO_A["id"], EQUIPO_A["name"], [
            lineup_jugador(10, "Jugador 10", [spell(23, "Center Forward")]),
            lineup_jugador(11, "Jugador 11", [spell(14, "Center Midfield")]),
            lineup_jugador(12, "Portero 12", [spell(1, "Goalkeeper")]),
        ]),
        lineup_equipo(EQUIPO_B["id"], EQUIPO_B["name"], [
            lineup_jugador(20, "Jugador 20", [spell(4, "Center Back")]),
            lineup_jugador(21, "Jugador 21", [spell(10, "Center Defensive Midfield")]),
        ]),
    ]
    return partido(match_id), eventos, alineaciones


# --- Base de datos sintetica -------------------------------------------------
# (competition_id, season_id, nombre competicion, nombre temporada)
LIGAS_SINTETICAS: tuple[tuple[int, int, str, str], ...] = (
    (1, 100, "Liga Uno", "2019/2020"),
    (2, 200, "Liga Dos", "2020/2021"),
)

# Tercera liga, para el flujo incremental: crear la BD con `LIGAS_SINTETICAS` y
# despues con `LIGAS_SINTETICAS + (LIGA_EXTRA,)` reproduce exactamente el caso
# real "la BD ya tenia unas ligas y llegan partidos de otra". Las ligas se
# generan en orden y cada una consume el rng despues de la anterior, asi que las
# filas de las dos primeras salen IDENTICAS en ambas versiones: justo lo que el
# reentrenamiento incremental debe poder reaprovechar.
LIGA_EXTRA: tuple[int, int, str, str] = (3, 300, "Liga Tres", "2021/2022")


def _metricas_jugador(rng: np.random.Generator) -> dict[str, Any]:
    """Fila jugador-partido coherente: los conteos respetan sus invariantes.

    Importa que sean coherentes (exitos <= intentos, duels_total = tackles +
    aereos + 50/50, aerial_won_off + aerial_won_def = aerial_won): la capa de
    features deriva ratios de estos conteos y con datos imposibles las pruebas
    pasarian sobre un fixture que miente.
    """
    passes = int(rng.integers(5, 90))
    passes_completed = int(rng.binomial(passes, 0.82))
    np_shots = int(rng.integers(0, 6))
    penaltis = int(rng.random() < 0.05)
    shots_on_target = int(rng.binomial(np_shots, 0.4))
    np_goals = int(rng.binomial(shots_on_target, 0.3))
    npxg = float(round(float(np.sum(rng.uniform(0.03, 0.35, size=np_shots))), 5))
    take_ons = int(rng.integers(0, 9))
    tackles = int(rng.integers(0, 6))
    tackles_won = int(rng.binomial(tackles, 0.6))
    aerial_won = int(rng.integers(0, 6))
    aerial_lost = int(rng.integers(0, 6))
    aerial_won_off = int(rng.binomial(aerial_won, 0.5))
    cincuenta = int(rng.integers(0, 3))
    cincuenta_won = int(rng.binomial(cincuenta, 0.5))
    interceptions = int(rng.integers(0, 6))
    ball_recoveries = int(rng.integers(0, 10))
    return {
        "passes": passes,
        "passes_completed": passes_completed,
        "progressive_passes": int(rng.binomial(passes_completed, 0.12)),
        "progressive_carries": int(rng.integers(0, 8)),
        "passes_into_final_third": int(rng.binomial(passes_completed, 0.08)),
        "xt": round(float(rng.uniform(0.0, 0.6)), 5),
        "xa": round(float(rng.uniform(0.0, 0.5)), 5),
        "passes_into_penalty_area": int(rng.binomial(passes_completed, 0.03)),
        "deep_completions": int(rng.binomial(passes_completed, 0.02)),
        "sca": int(rng.integers(0, 7)),
        "shots": np_shots + penaltis,
        "np_shots": np_shots,
        "goals": np_goals + penaltis,
        "np_goals": np_goals,
        "shots_on_target": shots_on_target,
        "xg": round(npxg + 0.78 * penaltis, 5),
        "npxg": npxg,
        "touches_in_att_pen_area": int(rng.integers(0, 10)),
        "take_ons": take_ons,
        "take_ons_won": int(rng.binomial(take_ons, 0.5)),
        "fouls_drawn": int(rng.integers(0, 4)),
        "dispossessed": int(rng.integers(0, 5)),
        "tackles": tackles,
        "tackles_won": tackles_won,
        "dribbled_past": int(rng.integers(0, 4)),
        "duels_total": tackles + aerial_won + aerial_lost + cincuenta,
        "duels_won": tackles_won + aerial_won + cincuenta_won,
        "aerial_won": aerial_won,
        "aerial_lost": aerial_lost,
        "aerial_won_off": aerial_won_off,
        "aerial_won_def": aerial_won - aerial_won_off,
        "interceptions": interceptions,
        "ball_recoveries": ball_recoveries,
        "padj_def_actions": round(
            (interceptions + ball_recoveries) * float(rng.uniform(0.8, 1.25)), 3
        ),
        "blocks": int(rng.integers(0, 5)),
        "clearances": int(rng.integers(0, 8)),
        "pressures": int(rng.integers(0, 30)),
        "pressure_regains": int(rng.integers(0, 8)),
        "counterpressures": int(rng.integers(0, 10)),
    }


def _metricas_equipo(
    rng: np.random.Generator, possession_pct: float, field_tilt: float
) -> dict[str, Any]:
    """Fila equipo-partido coherente. `possession_pct`/`field_tilt` se pasan desde
    fuera porque son cuotas: los dos equipos de un partido deben sumar 1.
    """
    passes = int(rng.integers(200, 700))
    passes_completed = int(rng.binomial(passes, 0.83))
    np_shots = int(rng.integers(3, 22))
    penaltis = int(rng.random() < 0.1)
    shots_on_target = int(rng.binomial(np_shots, 0.35))
    npxg = float(round(float(np.sum(rng.uniform(0.03, 0.35, size=np_shots))), 5))
    seq = int(rng.integers(60, 130))
    sca = int(rng.integers(5, 40))
    return {
        "possession_pct": round(possession_pct, 5),
        "passes": passes,
        "passes_completed": passes_completed,
        "ten_plus_pass_sequences": int(rng.integers(0, 25)),
        "progressive_passes": int(rng.binomial(passes_completed, 0.12)),
        "passes_into_final_third": int(rng.binomial(passes_completed, 0.08)),
        "xt": round(float(rng.uniform(0.5, 4.0)), 5),
        "sca": sca,
        "gca": int(rng.binomial(sca, 0.12)),
        "passes_into_penalty_area": int(rng.binomial(passes_completed, 0.02)),
        "open_play_xg": round(npxg * float(rng.uniform(0.7, 1.0)), 5),
        "build_up_attacks": int(rng.integers(0, 12)),
        "shots": np_shots + penaltis,
        "np_shots": np_shots,
        "goals": int(rng.binomial(shots_on_target, 0.3)) + penaltis,
        "shots_on_target": shots_on_target,
        "xg": round(npxg + 0.78 * penaltis, 5),
        "npxg": npxg,
        "ppda": round(float(rng.uniform(5.0, 25.0)), 4),
        "high_turnovers": int(rng.integers(0, 15)),
        "pressures": int(rng.integers(80, 220)),
        "counterpressures": int(rng.integers(10, 60)),
        "field_tilt": round(field_tilt, 5),
        "absolute_width": round(float(rng.uniform(30.0, 60.0)), 3),
        "sequence_start_distance": round(float(rng.uniform(30.0, 60.0)), 3),
        "direct_speed": round(float(rng.uniform(0.3, 2.5)), 4),
        "direct_attacks": int(rng.integers(0, 10)),
        "passes_per_sequence": round(float(rng.uniform(2.0, 6.0)), 4),
        "open_play_sequences": seq,
    }


def crear_bd_sintetica(
    db_path: Path,
    *,
    n_equipos: int = 3,
    n_jugadores: int = 4,
    n_partidos: int = 4,
    seed: int = 7,
    ligas: tuple[tuple[int, int, str, str], ...] = LIGAS_SINTETICAS,
) -> dict[str, Any]:
    """Crea una BD con el esquema REAL (`database.init_schema`) y filas sinteticas.

    Se usa el esquema de produccion a proposito: si el esquema cambia y el
    fixture deja de encajar, la suite debe enterarse.

    Dos ligas para que `normalizacion='por_liga'` y `'global'` sean
    transformaciones distintas (con una sola liga son identicas por construccion,
    ver `comparar._avisar_si_una_sola_liga`). Devuelve un resumen con los ids y
    nombres creados.

    `ligas` permite anadir una mas (ver `LIGA_EXTRA`) para montar el escenario
    incremental sin tocar las filas de las anteriores.
    """
    rng = np.random.default_rng(seed)
    conn = db.connect(Path(db_path))
    db.init_schema(conn)

    equipos: list[tuple[int, str]] = []
    jugadores: list[tuple[int, str]] = []
    partidos: list[int] = []
    claves_liga: list[str] = []
    contador_match = itertools.count(1000)

    for i, (cid, sid, cname, sname) in enumerate(ligas):
        db.upsert_competition(conn, {
            "competition_id": cid,
            "competition_name": cname,
            "country_name": f"Pais {i + 1}",
            "competition_gender": "male",
        })
        db.upsert_season(conn, {"season_id": sid, "season_name": sname})
        claves_liga.append(f"{cid}-{sid}")

        ids_equipos = [100 * (i + 1) + t for t in range(n_equipos)]
        jug_por_equipo: dict[int, list[int]] = {}
        for tid in ids_equipos:
            nombre = f"Equipo {tid}"
            db.upsert_team(conn, tid, nombre)
            db.upsert_team_competition(conn, tid, cid, sid)
            equipos.append((tid, nombre))
            jug_por_equipo[tid] = []
            for p in range(n_jugadores):
                pid = tid * 100 + p
                nombre_j = f"Jugador {pid}"
                db.upsert_player(conn, pid, nombre_j)
                jugadores.append((pid, nombre_j))
                jug_por_equipo[tid].append(pid)

        pares = [
            (a, b)
            for x, a in enumerate(ids_equipos)
            for b in ids_equipos[x + 1:]
        ]
        for j in range(n_partidos):
            local, visitante = pares[j % len(pares)]
            mid = next(contador_match)
            partidos.append(mid)
            gl, gv = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            db.upsert_match(conn, {
                "match_id": mid,
                "competition_id": cid,
                "season_id": sid,
                "match_date": f"2020-0{(j % 9) + 1}-01",
                "match_week": j + 1,
                "competition_stage": "Regular Season",
                "home_team_id": local,
                "away_team_id": visitante,
                "home_score": gl,
                "away_score": gv,
            })
            posesion_local = float(rng.uniform(0.35, 0.65))
            tilt_local = float(rng.uniform(0.3, 0.7))
            contexto = (
                (local, visitante, 1, gl, gv, posesion_local, tilt_local),
                (visitante, local, 0, gv, gl, 1.0 - posesion_local, 1.0 - tilt_local),
            )
            for tid, opp, is_home, gf, ga, pos, tilt in contexto:
                fila_t = _metricas_equipo(rng, pos, tilt)
                fila_t.update({
                    "team_id": tid, "match_id": mid, "opponent_id": opp,
                    "is_home": is_home, "goals_for": gf, "goals_against": ga,
                })
                db.upsert_team_stats(conn, fila_t)
                for pid in jug_por_equipo[tid]:
                    minutos = float(round(rng.uniform(45.0, 96.0), 2))
                    fila_j = _metricas_jugador(rng)
                    fila_j.update({
                        "player_id": pid, "match_id": mid, "team_id": tid,
                        "minutes_played": minutos,
                        "primary_position_id": 14,
                        "primary_position_name": "Center Midfield",
                    })
                    db.upsert_player_stats(conn, fila_j)
                    db.replace_player_positions(conn, pid, mid, [
                        {"position_id": 14, "position_name": "Center Midfield",
                         "minutes": minutos},
                    ])

    conn.commit()
    conn.close()
    return {
        "db_path": Path(db_path),
        "equipos": equipos,
        "jugadores": jugadores,
        "partidos": partidos,
        "ligas": claves_liga,
        "n_obs_jugador": len(partidos) * 2 * n_jugadores,
        "n_obs_equipo": len(partidos) * 2,
    }

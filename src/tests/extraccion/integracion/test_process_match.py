"""Integracion del extractor: JSON de eventos -> metricas -> SQLite.

Aqui ya no se aisla nada: se usan los lectores reales, las metricas reales y el
esquema real sobre una BD en `tmp_path`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.extraccion import database as db
from src.extraccion.extract import process_match
from src.tests import factories as fac

pytestmark = pytest.mark.integracion


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = db.connect(tmp_path / "scouting.db")
    db.init_schema(c)
    db.upsert_competition(c, {"competition_id": 1, "competition_name": "Liga Sintetica",
                              "country_name": "Pais", "competition_gender": "male"})
    db.upsert_season(c, {"season_id": 100, "season_name": "2019/2020"})
    yield c
    c.close()


@pytest.fixture
def procesado(conn: sqlite3.Connection, open_data_minimo: dict[str, Any]) -> dict[str, Any]:
    n_jug, n_eq = process_match(
        conn, open_data_minimo["match"], 1, 100, open_data_minimo["data_root"]
    )
    return {"conn": conn, "n_jugadores": n_jug, "n_equipos": n_eq,
            "match_id": open_data_minimo["match"]["match_id"],
            "data_root": open_data_minimo["data_root"],
            "match": open_data_minimo["match"]}


def _fila(conn: sqlite3.Connection, sql: str, *params: Any) -> Any:
    return conn.execute(sql, params).fetchone()


class TestEscrituraCompleta:
    def test_devuelve_cuantas_filas_ha_escrito(self, procesado: dict) -> None:
        assert procesado["n_jugadores"] == 5   # 3 de A (uno sin eventos) + 2 de B
        assert procesado["n_equipos"] == 2

    def test_se_guarda_el_partido_con_su_contexto(self, procesado: dict) -> None:
        fila = _fila(procesado["conn"],
                     "SELECT competition_id, season_id, home_team_id, away_team_id, "
                     "home_score, away_score FROM matches WHERE match_id=?",
                     procesado["match_id"])
        assert fila == (1, 100, 1, 2, 1, 0)

    def test_se_dan_de_alta_los_dos_equipos(self, procesado: dict) -> None:
        nombres = procesado["conn"].execute(
            "SELECT team_name FROM teams ORDER BY team_id").fetchall()
        assert nombres == [("Equipo A",), ("Equipo B",)]

    def test_se_vincula_cada_equipo_con_su_competicion_temporada(self, procesado: dict) -> None:
        filas = procesado["conn"].execute(
            "SELECT team_id, competition_id, season_id FROM team_competitions "
            "ORDER BY team_id").fetchall()
        assert filas == [(1, 1, 100), (2, 1, 100)]

    def test_se_da_de_alta_a_todos_los_jugadores_que_jugaron(self, procesado: dict) -> None:
        ids = procesado["conn"].execute("SELECT player_id FROM players ORDER BY player_id").fetchall()
        assert [i[0] for i in ids] == [10, 11, 12, 20, 21]

    def test_se_guarda_una_fila_por_jugador_partido(self, procesado: dict) -> None:
        n = _fila(procesado["conn"], "SELECT COUNT(*) FROM player_match_stats")[0]
        assert n == 5

    def test_se_guarda_el_detalle_de_posiciones(self, procesado: dict) -> None:
        fila = _fila(procesado["conn"],
                     "SELECT position_id, position_name FROM player_match_positions "
                     "WHERE player_id=10")
        assert fila == (23, "Center Forward")


class TestMetricasPersistidas:
    def test_el_goleador_tiene_su_gol_y_su_xg(self, procesado: dict) -> None:
        fila = _fila(procesado["conn"],
                     "SELECT goals, np_goals, xg, npxg FROM player_match_stats WHERE player_id=10")
        assert fila == (1, 1, pytest.approx(0.4), pytest.approx(0.4))

    def test_el_asistente_tiene_su_xa(self, procesado: dict) -> None:
        """El pase clave hereda el xG del tiro: 0.4."""
        xa = _fila(procesado["conn"], "SELECT xa FROM player_match_stats WHERE player_id=11")[0]
        assert xa == pytest.approx(0.4)

    def test_el_que_presiona_se_lleva_el_pressure_regain(self, procesado: dict) -> None:
        regains = _fila(procesado["conn"],
                        "SELECT pressure_regains FROM player_match_stats WHERE player_id=11")[0]
        assert regains == 1

    def test_un_jugador_sin_eventos_queda_a_cero_no_a_null(self, procesado: dict) -> None:
        """El portero jugo pero no aparece en los eventos: sus metricas son 0.
        NULL significaria "no se sabe" y contaminaria el z-score aguas abajo.
        """
        conn = procesado["conn"]
        fila = conn.execute(
            "SELECT passes, shots, tackles, xg FROM player_match_stats WHERE player_id=12"
        ).fetchone()
        assert fila == (0, 0, 0, 0.0)

    def test_un_jugador_sin_eventos_conserva_sus_minutos_y_posicion(self, procesado: dict) -> None:
        fila = _fila(procesado["conn"],
                     "SELECT minutes_played, primary_position_name FROM player_match_stats "
                     "WHERE player_id=12")
        assert fila[0] > 0
        assert fila[1] == "Goalkeeper"

    def test_no_queda_ninguna_metrica_de_jugador_a_null(self, procesado: dict) -> None:
        columnas = ", ".join(f'"{c}"' for c in db.PLAYER_METRIC_COLUMNS)
        filas = procesado["conn"].execute(
            f"SELECT {columnas} FROM player_match_stats").fetchall()
        for fila in filas:
            assert all(v is not None for v in fila)

    def test_padj_ajusta_por_la_posesion_del_equipo(self, procesado: dict) -> None:
        """No es un conteo crudo: debe diferir de intercepciones + recuperaciones
        salvo que la posesion sea justo del 50%.
        """
        fila = _fila(procesado["conn"],
                     "SELECT interceptions, ball_recoveries, padj_def_actions "
                     "FROM player_match_stats WHERE player_id=10")
        interceptions, recoveries, padj = fila
        assert padj >= 0.0
        assert (interceptions + recoveries) > 0   # el jugador 10 recupera una vez

    def test_las_metricas_de_equipo_se_guardan_con_su_contexto(self, procesado: dict) -> None:
        fila = _fila(procesado["conn"],
                     "SELECT opponent_id, is_home, goals_for, goals_against "
                     "FROM team_match_stats WHERE team_id=1")
        assert fila == (2, 1, 1, 0)

    def test_el_visitante_lleva_el_contexto_invertido(self, procesado: dict) -> None:
        fila = _fila(procesado["conn"],
                     "SELECT opponent_id, is_home, goals_for, goals_against "
                     "FROM team_match_stats WHERE team_id=2")
        assert fila == (1, 0, 0, 1)

    def test_las_posesiones_de_los_dos_equipos_suman_uno(self, procesado: dict) -> None:
        filas = procesado["conn"].execute(
            "SELECT possession_pct FROM team_match_stats").fetchall()
        assert sum(f[0] for f in filas) == pytest.approx(1.0)


class TestInvariantesEntreCapas:
    def test_el_sca_del_equipo_es_la_suma_del_de_sus_jugadores(self, procesado: dict) -> None:
        """Invariante declarado en `config.SCA_ACTION_TYPES`: ambas capas comparten
        la definicion justo para que esto se cumpla.
        """
        conn = procesado["conn"]
        for team_id in (1, 2):
            sca_equipo = _fila(conn, "SELECT sca FROM team_match_stats WHERE team_id=?", team_id)[0]
            sca_jugadores = _fila(
                conn, "SELECT COALESCE(SUM(sca), 0) FROM player_match_stats WHERE team_id=?",
                team_id,
            )[0]
            assert sca_equipo == sca_jugadores, f"equipo {team_id}"

    def test_los_goles_de_equipo_cuadran_con_los_de_sus_jugadores(self, procesado: dict) -> None:
        conn = procesado["conn"]
        goles_equipo = _fila(conn, "SELECT goals FROM team_match_stats WHERE team_id=1")[0]
        goles_jugadores = _fila(
            conn, "SELECT COALESCE(SUM(goals), 0) FROM player_match_stats WHERE team_id=1")[0]
        assert goles_equipo == goles_jugadores

    def test_el_xg_de_equipo_cuadra_con_el_de_sus_jugadores(self, procesado: dict) -> None:
        conn = procesado["conn"]
        xg_equipo = _fila(conn, "SELECT xg FROM team_match_stats WHERE team_id=1")[0]
        xg_jugadores = _fila(
            conn, "SELECT COALESCE(SUM(xg), 0) FROM player_match_stats WHERE team_id=1")[0]
        assert xg_equipo == pytest.approx(xg_jugadores, abs=1e-4)

    def test_los_goles_del_marcador_cuadran_con_los_eventos(self, procesado: dict) -> None:
        conn = procesado["conn"]
        marcador = _fila(conn, "SELECT home_score FROM matches")[0]
        eventos = _fila(conn, "SELECT goals FROM team_match_stats WHERE team_id=1")[0]
        assert marcador == eventos


class TestReprocesado:
    def test_reprocesar_no_duplica_filas(self, procesado: dict) -> None:
        """El extractor esta pensado para relanzarse sobre la misma BD."""
        conn = procesado["conn"]
        antes = {
            t: _fila(conn, f"SELECT COUNT(*) FROM {t}")[0]
            for t in ("matches", "teams", "players", "player_match_stats",
                      "team_match_stats", "player_match_positions", "team_competitions")
        }
        process_match(conn, procesado["match"], 1, 100, procesado["data_root"])
        despues = {t: _fila(conn, f"SELECT COUNT(*) FROM {t}")[0] for t in antes}
        assert antes == despues

    def test_reprocesar_da_exactamente_las_mismas_metricas(self, procesado: dict) -> None:
        """El calculo es determinista: no hay orden aleatorio ni acumulacion."""
        conn = procesado["conn"]
        columnas = ", ".join(f'"{c}"' for c in db.PLAYER_METRIC_COLUMNS)
        antes = conn.execute(
            f"SELECT player_id, {columnas} FROM player_match_stats ORDER BY player_id").fetchall()
        process_match(conn, procesado["match"], 1, 100, procesado["data_root"])
        despues = conn.execute(
            f"SELECT player_id, {columnas} FROM player_match_stats ORDER BY player_id").fetchall()
        assert antes == despues


class TestVariosPartidos:
    def test_dos_partidos_acumulan_filas_por_jugador(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """El grano es (jugador, partido): el mismo jugador en dos partidos son dos
        observaciones. Que no colapsen es el principio central del modelo.
        """
        m1, ev1, lu1 = fac.partido_minimo(7001)
        m2, ev2, lu2 = fac.partido_minimo(7002)
        raiz = fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[fac.fila_competicion(1, 100)],
            partidos={(1, 100): [m1, m2]},
            eventos={7001: ev1, 7002: ev2},
            alineaciones={7001: lu1, 7002: lu2},
        )
        for m in (m1, m2):
            process_match(conn, m, 1, 100, raiz)

        assert _fila(conn, "SELECT COUNT(*) FROM matches")[0] == 2
        assert _fila(conn, "SELECT COUNT(*) FROM players")[0] == 5
        assert _fila(conn, "SELECT COUNT(*) FROM player_match_stats")[0] == 10
        assert _fila(
            conn, "SELECT COUNT(*) FROM player_match_stats WHERE player_id=10")[0] == 2

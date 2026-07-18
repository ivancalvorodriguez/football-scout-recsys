"""Extremo a extremo del extractor: `python -m src.extraccion.extract`.

Se lanza el modulo como subproceso, igual que lo documenta CLAUDE.md, contra un
arbol open-data sintetico y una BD en `tmp_path`. Es la unica prueba que
comprueba que el punto de entrada esta realmente cableado (argparse, rutas,
apertura de la BD, bucle de partidos).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.tests import factories as fac
from src.tests.conftest import ejecutar_modulo

pytestmark = pytest.mark.e2e


@pytest.fixture
def dataset(tmp_path: Path) -> dict:
    """Una competicion-temporada con dos partidos en dos jornadas."""
    m1, ev1, lu1 = fac.partido_minimo(7001)
    m2, ev2, lu2 = fac.partido_minimo(7002)
    m1["match_week"], m1["match_date"] = 1, "2020-01-01"
    m2["match_week"], m2["match_date"] = 2, "2020-01-08"
    raiz = fac.escribir_open_data(
        tmp_path / "open-data" / "data",
        competiciones=[fac.fila_competicion(
            1, 100, competition_name="Liga Sintetica", season_name="2019/2020")],
        partidos={(1, 100): [m1, m2]},
        eventos={7001: ev1, 7002: ev2},
        alineaciones={7001: lu1, 7002: lu2},
    )
    return {"data_root": raiz, "db": tmp_path / "outputs" / "db" / "scouting.db"}


def _extraer(dataset: dict, *args: str):
    return ejecutar_modulo(
        "src.extraccion.extract",
        "--competition", "Liga Sintetica",
        "--season", "2019/2020",
        "--db", str(dataset["db"]),
        "--data-root", str(dataset["data_root"]),
        *args,
    )


class TestEjecucionCompleta:
    def test_termina_bien_y_crea_la_base_de_datos(self, dataset: dict) -> None:
        res = _extraer(dataset, "--pct", "100")
        assert res.returncode == 0, res.stderr
        assert dataset["db"].exists()

    def test_informa_de_lo_que_ha_escrito(self, dataset: dict) -> None:
        res = _extraer(dataset, "--pct", "100")
        assert "2 partidos" in res.stdout
        assert "10 jugador-partido" in res.stdout   # 5 jugadores x 2 partidos
        assert "4 equipo-partido" in res.stdout

    def test_los_datos_quedan_completos_en_la_bd(self, dataset: dict) -> None:
        _extraer(dataset, "--pct", "100")
        conn = sqlite3.connect(dataset["db"])
        try:
            cuenta = lambda t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            assert cuenta("competitions") == 1
            assert cuenta("seasons") == 1
            assert cuenta("matches") == 2
            assert cuenta("teams") == 2
            assert cuenta("players") == 5
            assert cuenta("player_match_stats") == 10
            assert cuenta("team_match_stats") == 4
        finally:
            conn.close()

    def test_las_metricas_llegan_calculadas_hasta_la_bd(self, dataset: dict) -> None:
        """No basta con que haya filas: deben traer el calculo real."""
        _extraer(dataset, "--pct", "100")
        conn = sqlite3.connect(dataset["db"])
        try:
            goles, xg = conn.execute(
                "SELECT SUM(goals), SUM(xg) FROM player_match_stats WHERE player_id=10"
            ).fetchone()
            assert goles == 2                      # un gol en cada partido
            assert xg == pytest.approx(0.8)
        finally:
            conn.close()

    def test_el_porcentaje_de_jornadas_limita_los_partidos(self, dataset: dict) -> None:
        res = _extraer(dataset, "--pct", "50")
        assert res.returncode == 0, res.stderr
        conn = sqlite3.connect(dataset["db"])
        try:
            assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
        finally:
            conn.close()

    def test_relanzarlo_no_duplica_nada(self, dataset: dict) -> None:
        _extraer(dataset, "--pct", "100")
        _extraer(dataset, "--pct", "100")
        conn = sqlite3.connect(dataset["db"])
        try:
            assert conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0] == 10
            assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
        finally:
            conn.close()


class TestErroresDeUso:
    def test_una_competicion_inexistente_avisa_y_no_crea_bd(self, dataset: dict) -> None:
        res = ejecutar_modulo(
            "src.extraccion.extract",
            "--competition", "Liga Que No Existe",
            "--season", "2019/2020",
            "--db", str(dataset["db"]),
            "--data-root", str(dataset["data_root"]),
        )
        assert res.returncode == 0          # no es un fallo del programa
        assert "No se encontró la competición" in res.stdout
        assert not dataset["db"].exists()   # no deja una BD vacia a medias

    def test_una_temporada_inexistente_sugiere_las_validas(self, dataset: dict) -> None:
        res = ejecutar_modulo(
            "src.extraccion.extract",
            "--competition", "Liga Sintetica",
            "--season", "1999/2000",
            "--db", str(dataset["db"]),
            "--data-root", str(dataset["data_root"]),
        )
        assert "No se encontró la temporada" in res.stdout
        assert "2019/2020" in res.stdout

    def test_la_ayuda_documenta_las_opciones(self) -> None:
        res = ejecutar_modulo("src.extraccion.extract", "--help")
        assert res.returncode == 0
        for opcion in ("--competition", "--season", "--pct", "--db", "--data-root"):
            assert opcion in res.stdout

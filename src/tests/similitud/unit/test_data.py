"""Carga de observaciones por-partido desde SQLite (solo lectura)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.extraccion import database as db
from src.similitud import data
from src.similitud.data import LEAGUE_KEY


class TestCargarJugadores:
    def test_una_fila_por_jugador_partido(self, resumen_bd: dict[str, Any]) -> None:
        """El principio central: NO se agrega por entidad. M filas entran, M salen."""
        df = data.cargar_jugadores(resumen_bd["db_path"])
        assert len(df) == resumen_bd["n_obs_jugador"]

    def test_cada_entidad_conserva_todas_sus_observaciones(self, resumen_bd: dict) -> None:
        df = data.cargar_jugadores(resumen_bd["db_path"])
        assert len(df) > df["entity_id"].nunique()

    def test_el_id_se_expone_como_entity_id(self, resumen_bd: dict) -> None:
        """`entity_id`/`entity_name` homogeneizan la interfaz con `cargar_equipos`."""
        df = data.cargar_jugadores(resumen_bd["db_path"])
        assert "entity_id" in df.columns
        assert "player_id" not in df.columns

    def test_trae_el_nombre_del_jugador(self, resumen_bd: dict) -> None:
        df = data.cargar_jugadores(resumen_bd["db_path"])
        nombres = set(df["entity_name"])
        assert nombres == {n for _, n in resumen_bd["jugadores"]}

    def test_trae_los_minutos_para_el_per90(self, resumen_bd: dict) -> None:
        df = data.cargar_jugadores(resumen_bd["db_path"])
        assert (df["minutes_played"] > 0).all()

    def test_trae_todas_las_columnas_de_metricas(self, resumen_bd: dict) -> None:
        df = data.cargar_jugadores(resumen_bd["db_path"])
        assert set(db.PLAYER_METRIC_COLUMNS) <= set(df.columns)

    def test_la_clave_de_liga_junta_competicion_y_temporada(self, resumen_bd: dict) -> None:
        """El z-score por liga se agrupa por competicion-temporada, no por
        competicion: la Premier de 2015 no es la de 2003.
        """
        df = data.cargar_jugadores(resumen_bd["db_path"])
        assert set(df[LEAGUE_KEY]) == set(resumen_bd["ligas"])

    def test_la_clave_de_liga_se_deriva_del_partido(self, resumen_bd: dict) -> None:
        df = data.cargar_jugadores(resumen_bd["db_path"])
        esperado = df["competition_id"].astype(str) + "-" + df["season_id"].astype(str)
        assert (df[LEAGUE_KEY] == esperado).all()


class TestCargarEquipos:
    def test_una_fila_por_equipo_partido(self, resumen_bd: dict) -> None:
        df = data.cargar_equipos(resumen_bd["db_path"])
        assert len(df) == resumen_bd["n_obs_equipo"]

    def test_el_id_se_expone_como_entity_id(self, resumen_bd: dict) -> None:
        df = data.cargar_equipos(resumen_bd["db_path"])
        assert "entity_id" in df.columns
        assert "team_id" not in df.columns

    def test_trae_el_nombre_del_equipo(self, resumen_bd: dict) -> None:
        df = data.cargar_equipos(resumen_bd["db_path"])
        assert set(df["entity_name"]) == {n for _, n in resumen_bd["equipos"]}

    def test_trae_todas_las_columnas_de_metricas(self, resumen_bd: dict) -> None:
        df = data.cargar_equipos(resumen_bd["db_path"])
        assert set(db.TEAM_METRIC_COLUMNS) <= set(df.columns)

    def test_el_equipo_no_tiene_minutos(self, resumen_bd: dict) -> None:
        """Juega el partido completo: la ponderacion por minutos no aplica."""
        df = data.cargar_equipos(resumen_bd["db_path"])
        assert "minutes_played" not in df.columns

    def test_la_clave_de_liga(self, resumen_bd: dict) -> None:
        df = data.cargar_equipos(resumen_bd["db_path"])
        assert set(df[LEAGUE_KEY]) == set(resumen_bd["ligas"])


class TestDespacho:
    def test_jugador(self, bd_sintetica: Path) -> None:
        assert len(data.cargar(bd_sintetica, "jugador")) > 0

    def test_equipo(self, bd_sintetica: Path) -> None:
        assert len(data.cargar(bd_sintetica, "equipo")) > 0

    def test_las_dos_entidades_dan_universos_distintos(self, bd_sintetica: Path) -> None:
        jug = data.cargar(bd_sintetica, "jugador")
        eq = data.cargar(bd_sintetica, "equipo")
        assert len(jug) != len(eq)

    def test_una_entidad_desconocida_falla_pronto(self, bd_sintetica: Path) -> None:
        with pytest.raises(ValueError, match="entidad desconocida"):
            data.cargar(bd_sintetica, "arbitro")


class TestSoloLectura:
    def test_la_conexion_es_de_solo_lectura(self, bd_sintetica: Path) -> None:
        """`src/similitud` nunca debe modificar la BD que produce el extractor."""
        conn = data._connect(bd_sintetica)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("DELETE FROM players")
        finally:
            conn.close()

    def test_no_se_puede_crear_una_bd_que_no_existe(self, tmp_path: Path) -> None:
        """En modo `ro`, sqlite falla en vez de crear un fichero vacio y devolver
        cero filas en silencio.
        """
        with pytest.raises(sqlite3.OperationalError):
            data._connect(tmp_path / "no-existe.db").execute("SELECT 1")

    def test_cargar_no_deja_la_conexion_abierta(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        """El context manager propio de sqlite3 hace commit/rollback pero NO cierra
        la conexion: por eso `_leer` usa `contextlib.closing`. Si la fuga volviese,
        Windows mantendria el fichero bloqueado y no se podria borrar.
        """
        copia = tmp_path / "copia.db"
        copia.write_bytes(bd_sintetica.read_bytes())
        for _ in range(20):
            data.cargar_jugadores(copia)
            data.cargar_equipos(copia)
        copia.unlink()   # falla con PermissionError si queda algun handle abierto
        assert not copia.exists()

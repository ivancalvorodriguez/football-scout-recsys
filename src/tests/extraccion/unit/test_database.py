"""Esquema SQLite y escritura con UPSERT."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.extraccion import database as db


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = db.connect(tmp_path / "sub" / "dir" / "scouting.db")
    db.init_schema(c)
    yield c
    c.close()


def _tablas(conn: sqlite3.Connection) -> set[str]:
    filas = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {f[0] for f in filas}


def _columnas(conn: sqlite3.Connection, tabla: str) -> dict[str, str]:
    return {f[1]: f[2] for f in conn.execute(f"PRAGMA table_info({tabla})").fetchall()}


class TestConexion:
    def test_crea_los_directorios_que_falten(self, tmp_path: Path) -> None:
        """`outputs/db/` no esta versionado: la primera ejecucion debe crearlo."""
        ruta = tmp_path / "no" / "existe" / "scouting.db"
        conn = db.connect(ruta)
        conn.close()
        assert ruta.exists()

    def test_las_claves_foraneas_estan_activas(self, conn: sqlite3.Connection) -> None:
        """SQLite las desactiva por defecto en cada conexion."""
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


class TestEsquema:
    def test_estan_las_dimensiones_y_los_hechos(self, conn: sqlite3.Connection) -> None:
        esperadas = {
            "competitions", "seasons", "teams", "players", "team_competitions",
            "matches", "player_match_stats", "team_match_stats",
            "player_match_positions",
        }
        assert esperadas <= _tablas(conn)

    def test_init_schema_es_idempotente(self, conn: sqlite3.Connection) -> None:
        """Se llama en cada ejecucion del extractor sobre la misma BD."""
        db.init_schema(conn)
        db.init_schema(conn)
        assert "player_match_stats" in _tablas(conn)

    def test_estan_todas_las_columnas_de_metricas_de_jugador(self, conn: sqlite3.Connection) -> None:
        assert set(db.PLAYER_COLUMNS) <= set(_columnas(conn, "player_match_stats"))

    def test_estan_todas_las_columnas_de_metricas_de_equipo(self, conn: sqlite3.Connection) -> None:
        assert set(db.TEAM_COLUMNS) <= set(_columnas(conn, "team_match_stats"))

    def test_las_columnas_son_contexto_mas_metricas(self) -> None:
        assert db.PLAYER_COLUMNS == db._PLAYER_CONTEXT + db.PLAYER_METRIC_COLUMNS
        assert db.TEAM_COLUMNS == db._TEAM_CONTEXT + db.TEAM_METRIC_COLUMNS

    def test_no_hay_columnas_de_metricas_repetidas(self) -> None:
        assert len(set(db.PLAYER_METRIC_COLUMNS)) == len(db.PLAYER_METRIC_COLUMNS)
        assert len(set(db.TEAM_METRIC_COLUMNS)) == len(db.TEAM_METRIC_COLUMNS)

    def test_las_metricas_de_modelo_son_reales_y_los_conteos_enteros(
        self, conn: sqlite3.Connection
    ) -> None:
        """xG/xT/xA no son conteos: guardarlos como INTEGER los truncaria a 0."""
        cols = _columnas(conn, "player_match_stats")
        for real in db._PLAYER_REAL:
            assert cols[real] == "REAL", f"{real} deberia ser REAL"
        assert cols["passes"] == "INTEGER"
        assert cols["goals"] == "INTEGER"

    def test_las_tasas_de_equipo_son_reales(self, conn: sqlite3.Connection) -> None:
        cols = _columnas(conn, "team_match_stats")
        for real in db._TEAM_REAL:
            assert cols[real] == "REAL", f"{real} deberia ser REAL"

    def test_el_one_hot_de_posicion_no_se_persiste(self, conn: sqlite3.Connection) -> None:
        """Decision de diseño: se guarda `primary_position_*` y el detalle; el
        one-hot se genera al construir la matriz de features.
        """
        cols = _columnas(conn, "player_match_stats")
        assert not [c for c in cols if c.startswith("pos_")]
        assert "primary_position_id" in cols


class TestUpsertDimensiones:
    def test_upsert_competition(self, conn: sqlite3.Connection) -> None:
        db.upsert_competition(conn, {
            "competition_id": 1, "competition_name": "La Liga",
            "country_name": "Spain", "competition_gender": "male",
        })
        assert conn.execute("SELECT competition_name FROM competitions").fetchone()[0] == "La Liga"

    def test_reinsertar_una_dimension_no_la_duplica(self, conn: sqlite3.Connection) -> None:
        db.upsert_team(conn, 1, "Barcelona")
        db.upsert_team(conn, 1, "Barcelona")
        assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 1

    def test_reinsertar_una_dimension_actualiza_sus_datos(self, conn: sqlite3.Connection) -> None:
        db.upsert_player(conn, 10, "Nombre Viejo")
        db.upsert_player(conn, 10, "Nombre Nuevo")
        assert conn.execute("SELECT player_name FROM players").fetchone()[0] == "Nombre Nuevo"

    def test_un_equipo_puede_jugar_varias_competiciones(self, conn: sqlite3.Connection) -> None:
        """Relacion N:M: liga + Champions en la misma temporada."""
        db.upsert_team(conn, 1, "Barcelona")
        db.upsert_competition(conn, {"competition_id": 11, "competition_name": "La Liga",
                                     "country_name": "Spain", "competition_gender": "male"})
        db.upsert_competition(conn, {"competition_id": 16, "competition_name": "Champions",
                                     "country_name": "Europe", "competition_gender": "male"})
        db.upsert_season(conn, {"season_id": 90, "season_name": "2019/2020"})
        db.upsert_team_competition(conn, 1, 11, 90)
        db.upsert_team_competition(conn, 1, 16, 90)
        db.upsert_team_competition(conn, 1, 11, 90)   # repetido: no duplica
        assert conn.execute("SELECT COUNT(*) FROM team_competitions").fetchone()[0] == 2


class TestUpsertHechos:
    @pytest.fixture
    def conn(self, conn: sqlite3.Connection) -> sqlite3.Connection:
        """Las tablas de hechos tienen claves foraneas a las dimensiones: hay que
        darlas de alta antes, igual que hace `extract.process_match`.
        """
        db.upsert_competition(conn, {"competition_id": 1, "competition_name": "Liga",
                                     "country_name": "Pais", "competition_gender": "male"})
        db.upsert_season(conn, {"season_id": 100, "season_name": "2019/2020"})
        db.upsert_team(conn, 1, "Equipo A")
        db.upsert_team(conn, 2, "Equipo B")
        db.upsert_player(conn, 10, "Jugador 10")
        db.upsert_match(conn, {
            "match_id": 1, "competition_id": 1, "season_id": 100,
            "match_date": "2020-01-01", "match_week": 1,
            "competition_stage": "Regular Season",
            "home_team_id": 1, "away_team_id": 2, "home_score": 0, "away_score": 0,
        })
        return conn

    def _fila_jugador(self, **cambios) -> dict:
        fila = {c: 0 for c in db.PLAYER_COLUMNS}
        fila.update({"player_id": 10, "match_id": 1, "team_id": 1, "minutes_played": 90.0})
        fila.update(cambios)
        return fila

    def test_reprocesar_un_partido_actualiza_sin_duplicar(self, conn: sqlite3.Connection) -> None:
        """La clave primaria es (player_id, match_id): reprocesar es idempotente."""
        db.upsert_player_stats(conn, self._fila_jugador(goals=1))
        db.upsert_player_stats(conn, self._fila_jugador(goals=2))
        filas = conn.execute("SELECT goals FROM player_match_stats").fetchall()
        assert filas == [(2,)]

    def test_una_columna_ausente_se_guarda_como_null(self, conn: sqlite3.Connection) -> None:
        fila = self._fila_jugador()
        del fila["xg"]
        db.upsert_player_stats(conn, fila)
        assert conn.execute("SELECT xg FROM player_match_stats").fetchone()[0] is None

    def test_upsert_team_stats(self, conn: sqlite3.Connection) -> None:
        fila = {c: 0 for c in db.TEAM_COLUMNS}
        fila.update({"team_id": 1, "match_id": 1, "opponent_id": 2, "is_home": 1,
                     "possession_pct": 0.55})
        db.upsert_team_stats(conn, fila)
        db.upsert_team_stats(conn, fila)
        assert conn.execute("SELECT COUNT(*) FROM team_match_stats").fetchone()[0] == 1

    def test_upsert_match(self, conn: sqlite3.Connection) -> None:
        db.upsert_match(conn, {
            "match_id": 7, "competition_id": 1, "season_id": 100,
            "match_date": "2020-01-01", "match_week": 3,
            "competition_stage": "Regular Season",
            "home_team_id": 1, "away_team_id": 2, "home_score": 2, "away_score": 1,
        })
        fila = conn.execute(
            "SELECT match_week, home_score, competition_stage FROM matches WHERE match_id=7"
        ).fetchone()
        assert fila == (3, 2, "Regular Season")


class TestPosiciones:
    def test_se_guardan_todas_las_posiciones_del_jugador(self, conn: sqlite3.Connection) -> None:
        db.replace_player_positions(conn, 10, 1, [
            {"position_id": 23, "position_name": "Center Forward", "minutes": 60.0},
            {"position_id": 14, "position_name": "Center Midfield", "minutes": 30.0},
        ])
        assert conn.execute("SELECT COUNT(*) FROM player_match_positions").fetchone()[0] == 2

    def test_reprocesar_reemplaza_las_posiciones_anteriores(self, conn: sqlite3.Connection) -> None:
        """Un UPSERT a secas dejaria huerfana una posicion que ya no aplica."""
        db.replace_player_positions(conn, 10, 1, [
            {"position_id": 23, "position_name": "Center Forward", "minutes": 60.0},
            {"position_id": 14, "position_name": "Center Midfield", "minutes": 30.0},
        ])
        db.replace_player_positions(conn, 10, 1, [
            {"position_id": 23, "position_name": "Center Forward", "minutes": 90.0},
        ])
        filas = conn.execute(
            "SELECT position_id, minutes FROM player_match_positions"
        ).fetchall()
        assert filas == [(23, 90.0)]

    def test_solo_se_reemplazan_las_del_jugador_partido_indicado(
        self, conn: sqlite3.Connection
    ) -> None:
        db.replace_player_positions(conn, 10, 1, [
            {"position_id": 23, "position_name": "Center Forward", "minutes": 90.0}])
        db.replace_player_positions(conn, 11, 1, [
            {"position_id": 14, "position_name": "Center Midfield", "minutes": 90.0}])
        db.replace_player_positions(conn, 10, 2, [
            {"position_id": 4, "position_name": "Center Back", "minutes": 90.0}])
        db.replace_player_positions(conn, 10, 1, [
            {"position_id": 1, "position_name": "Goalkeeper", "minutes": 90.0}])
        filas = conn.execute(
            "SELECT player_id, match_id, position_id FROM player_match_positions "
            "ORDER BY player_id, match_id"
        ).fetchall()
        assert filas == [(10, 1, 1), (10, 2, 4), (11, 1, 14)]

    def test_una_lista_vacia_deja_al_jugador_sin_posiciones(self, conn: sqlite3.Connection) -> None:
        db.replace_player_positions(conn, 10, 1, [
            {"position_id": 23, "position_name": "Center Forward", "minutes": 90.0}])
        db.replace_player_positions(conn, 10, 1, [])
        assert conn.execute("SELECT COUNT(*) FROM player_match_positions").fetchone()[0] == 0

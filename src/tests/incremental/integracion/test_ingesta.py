"""Ingesta de un paquete sobre una BD real (esquema de produccion, en tmp_path).

Lo que se comprueba es que la ingesta NO es un camino paralelo: escribe las
mismas tablas que `src.extraccion.extract`, con UPSERT, y sabe decir que entidades
son nuevas — que es lo que despues justifica reentrenar.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.incremental import ingesta, paquete as pq
from src.tests import factories
from src.tests.incremental.conftest import escribir_paquete

pytestmark = pytest.mark.integracion


def _contar(db_path: Path, tabla: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]


def _ingerir(raiz: Path, db_path: Path) -> dict:
    paquete = pq.leer(raiz)
    assert not paquete.errores, paquete.errores
    return ingesta.ingerir(paquete, paquete.partidos, db_path)


def test_ingesta_sobre_bd_vacia_crea_el_esquema(tmp_path):
    p = escribir_paquete(tmp_path / "paquete")
    db_path = tmp_path / "nueva.db"
    informe = _ingerir(p["raiz"], db_path)

    assert db_path.exists()
    assert informe["partidos_nuevos"] == [p["match_id"]]
    assert _contar(db_path, "player_match_stats") == 5   # 5 jugadores con minutos
    assert _contar(db_path, "team_match_stats") == 2


def test_ingesta_sobre_bd_existente_conserva_lo_anterior(tmp_path):
    db_path = tmp_path / "scouting.db"
    resumen = factories.crear_bd_sintetica(db_path)
    antes_jugadores = _contar(db_path, "players")
    antes_partidos = _contar(db_path, "matches")

    p = escribir_paquete(tmp_path / "paquete", competition_id=9, season_id=900,
                         competition_name="Liga Nueva")
    informe = _ingerir(p["raiz"], db_path)

    assert _contar(db_path, "matches") == antes_partidos + 1
    assert _contar(db_path, "players") == antes_jugadores + 5
    assert len(informe["jugadores_nuevos"]) == 5
    assert len(informe["equipos_nuevos"]) == 2
    assert informe["ligas_nuevas"] == [{"competition_id": 9, "season_id": 900}]
    assert informe["totales_bd"]["partidos"] == len(resumen["partidos"]) + 1


def test_reingerir_el_mismo_partido_no_duplica(tmp_path):
    p = escribir_paquete(tmp_path / "paquete")
    db_path = tmp_path / "scouting.db"
    primero = _ingerir(p["raiz"], db_path)
    filas = _contar(db_path, "player_match_stats")

    segundo = _ingerir(p["raiz"], db_path)

    assert _contar(db_path, "player_match_stats") == filas
    assert _contar(db_path, "matches") == 1
    assert primero["partidos_nuevos"] == [p["match_id"]]
    assert segundo["partidos_nuevos"] == []          # ya estaba
    assert segundo["partidos_actualizados"] == [p["match_id"]]
    assert segundo["jugadores_nuevos"] == []


def test_las_metricas_son_las_del_motor_de_extraccion(tmp_path):
    """La ingesta llama a `extract.process_match`: no hay una segunda definicion."""
    from src.extraccion.extract import process_match
    from src.extraccion import database as db

    p = escribir_paquete(tmp_path / "paquete")

    por_ingesta = tmp_path / "a.db"
    _ingerir(p["raiz"], por_ingesta)

    por_extraccion = tmp_path / "b.db"
    conn = db.connect(por_extraccion)
    db.init_schema(conn)
    # Las dimensiones primero, como hace `extract.run`: `team_competitions` tiene
    # clave foranea contra competitions/seasons.
    db.upsert_competition(conn, factories.fila_competicion(1, 100))
    db.upsert_season(conn, {"season_id": 100, "season_name": "2019/2020"})
    process_match(conn, p["match"], 1, 100, p["raiz"])
    conn.close()

    def filas(ruta):
        with sqlite3.connect(ruta) as conn:
            return conn.execute(
                "SELECT * FROM player_match_stats ORDER BY player_id").fetchall()

    assert filas(por_ingesta) == filas(por_extraccion)


def test_la_copia_de_seguridad_es_una_copia_fiel(tmp_path):
    db_path = tmp_path / "scouting.db"
    factories.crear_bd_sintetica(db_path)
    copia = ingesta.copia_seguridad(db_path, dir_copias=tmp_path / "copias")

    assert copia is not None and copia.exists()
    assert copia.read_bytes() == db_path.read_bytes()


def test_sin_bd_previa_no_hay_copia_que_hacer(tmp_path):
    assert ingesta.copia_seguridad(tmp_path / "no-existe.db", tmp_path / "c") is None


def test_avisa_si_un_team_id_ya_existe_con_otro_nombre(tmp_path):
    """Senal de que el paquete inventa ids: el UPSERT pisaria al equipo equivocado."""
    db_path = tmp_path / "scouting.db"
    factories.crear_bd_sintetica(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO teams VALUES (?, ?)",
                     (factories.EQUIPO_A["id"], "Otro Club Distinto"))

    p = escribir_paquete(tmp_path / "paquete")
    avisos = ingesta._choques(ingesta._foto(db_path), pq.leer(p["raiz"]))

    assert any("Otro Club Distinto" in a for a in avisos)


def test_foto_de_bd_inexistente_esta_vacia(tmp_path):
    foto = ingesta._foto(tmp_path / "nada.db")
    assert foto["partidos"] == set() and foto["jugadores"] == {}


def test_foto_de_bd_sin_esquema_esta_vacia(tmp_path):
    ruta = tmp_path / "vacia.db"
    sqlite3.connect(ruta).close()
    assert ingesta._foto(ruta)["partidos"] == set()


def test_el_informe_es_json_serializable(tmp_path):
    p = escribir_paquete(tmp_path / "paquete")
    informe = _ingerir(p["raiz"], tmp_path / "s.db")
    assert json.loads(json.dumps(informe, ensure_ascii=False))["partidos_procesados"] == 1

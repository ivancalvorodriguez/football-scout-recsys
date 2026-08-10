"""El formato de entrada: que se acepta, que se rechaza y con que mensaje.

Estas pruebas son la red del contrato documentado en `docs/incremental.md`: si
alguien afloja una comprobacion, el paquete roto entraria en la BD produciendo
metricas mudas (todo a cero) en vez de un error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.incremental import paquete as pq
from src.tests.incremental.conftest import escribir_paquete, reescribir


def _validar_unico(raiz: Path) -> tuple[pq.Paquete, list[pq.Problema]]:
    paquete = pq.leer(raiz)
    assert paquete.partidos, "el paquete deberia traer un partido"
    return paquete, pq.validar(paquete, paquete.partidos[0])


def _codigos(problemas: list[pq.Problema]) -> str:
    return " | ".join(p.mensaje for p in problemas)


# --- Paquete correcto ---------------------------------------------------------
def test_paquete_valido_no_tiene_problemas(paquete_valido):
    paquete, problemas = _validar_unico(paquete_valido["raiz"])
    assert paquete.problemas == []
    assert problemas == []
    assert paquete.partidos[0].match_id == paquete_valido["match_id"]


def test_resuelve_la_competicion_de_cada_partido(paquete_valido):
    paquete = pq.leer(paquete_valido["raiz"])
    comp = paquete.partidos[0].competicion
    assert comp["competition_name"] == "Liga Sintetica"
    assert comp["season_name"] == "2019/2020"
    assert len(paquete.competiciones()) == 1


# --- Filtros ------------------------------------------------------------------
def test_filtra_por_match_id(tmp_path):
    escribir_paquete(tmp_path / "p", match_id=7001)
    assert pq.leer(tmp_path / "p", ids=[7001]).partidos
    fallido = pq.leer(tmp_path / "p", ids=[9999])
    assert fallido.partidos == []
    assert "9999" in _codigos(fallido.errores)


def test_filtra_por_nombre_de_competicion_y_temporada(paquete_valido):
    raiz = paquete_valido["raiz"]
    # El nombre no distingue mayusculas ni espacios sobrantes.
    assert pq.leer(raiz, competicion="  liga sintetica ").partidos
    assert pq.leer(raiz, temporada="2019/2020").partidos
    otro = pq.leer(raiz, competicion="Premier League")
    assert otro.partidos == []
    assert "Premier League" in _codigos(otro.errores)


# --- Estructura ---------------------------------------------------------------
def test_directorio_inexistente_es_error(tmp_path):
    paquete = pq.leer(tmp_path / "no-existe")
    assert paquete.errores and "no existe" in _codigos(paquete.errores)


def test_sin_competitions_json_es_error(paquete_valido):
    (paquete_valido["raiz"] / "competitions.json").unlink()
    paquete = pq.leer(paquete_valido["raiz"])
    assert any("competitions.json" in p.donde for p in paquete.errores)


def test_matches_de_competicion_no_declarada_es_error(paquete_valido):
    reescribir(paquete_valido["raiz"] / "competitions.json", [])
    paquete = pq.leer(paquete_valido["raiz"])
    assert "no esta declarado en competitions.json" in _codigos(paquete.errores)


def test_sin_carpeta_matches_es_error(paquete_valido):
    for fichero in (paquete_valido["raiz"] / "matches").rglob("*.json"):
        fichero.unlink()
    paquete = pq.leer(paquete_valido["raiz"])
    assert "matches/" in _codigos(paquete.errores) or paquete.errores


def test_partido_sin_equipos_es_error(paquete_valido):
    ruta = paquete_valido["raiz"] / "matches" / "1" / "100.json"
    partidos = json.loads(ruta.read_text(encoding="utf-8"))
    del partidos[0]["home_team"]
    reescribir(ruta, partidos)
    paquete = pq.leer(paquete_valido["raiz"])
    assert "home_team.home_team_id" in _codigos(paquete.errores)


def test_match_id_duplicado_es_error(paquete_valido):
    ruta = paquete_valido["raiz"] / "matches" / "1" / "100.json"
    partidos = json.loads(ruta.read_text(encoding="utf-8"))
    reescribir(ruta, partidos + partidos)
    paquete = pq.leer(paquete_valido["raiz"])
    assert "duplicado" in _codigos(paquete.errores)


def test_json_ilegible_es_error(paquete_valido):
    (paquete_valido["raiz"] / "competitions.json").write_text("{no json", encoding="utf-8")
    paquete = pq.leer(paquete_valido["raiz"])
    assert "ilegible" in _codigos(paquete.errores)


# --- Eventos ------------------------------------------------------------------
def test_sin_fichero_de_eventos_es_error(paquete_valido):
    (paquete_valido["raiz"] / "events" / "7001.json").unlink()
    _, problemas = _validar_unico(paquete_valido["raiz"])
    assert "no existe" in _codigos(problemas)


@pytest.mark.parametrize("campo", ["id", "type", "team", "period", "minute", "second"])
def test_evento_sin_campo_obligatorio_es_error(paquete_valido, campo):
    ruta = paquete_valido["raiz"] / "events" / "7001.json"
    eventos = json.loads(ruta.read_text(encoding="utf-8"))
    del eventos[0][campo]
    reescribir(ruta, eventos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    assert any(p.nivel == pq.NIVEL_ERROR for p in problemas), _codigos(problemas)


def test_ids_de_evento_repetidos_son_error(paquete_valido):
    """El id enlaza el pase clave con su tiro: repetirlo corrompe xA y SCA."""
    ruta = paquete_valido["raiz"] / "events" / "7001.json"
    eventos = json.loads(ruta.read_text(encoding="utf-8"))
    eventos[1]["id"] = eventos[0]["id"]
    reescribir(ruta, eventos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    assert "repetido" in _codigos(problemas)


def test_partido_con_reloj_a_cero_es_error(paquete_valido):
    """Sin minutos no se acumula tiempo jugado y el partido saldria vacio."""
    ruta = paquete_valido["raiz"] / "events" / "7001.json"
    eventos = json.loads(ruta.read_text(encoding="utf-8"))
    for ev in eventos:
        ev["minute"] = 0
    reescribir(ruta, eventos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    assert "minuto 0" in _codigos(problemas)


def test_coordenadas_fuera_del_campo_son_aviso(paquete_valido):
    ruta = paquete_valido["raiz"] / "events" / "7001.json"
    eventos = json.loads(ruta.read_text(encoding="utf-8"))
    eventos[0]["location"] = [500.0, 40.0]
    reescribir(ruta, eventos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    fuera = [p for p in problemas if "fuera del campo" in p.mensaje]
    assert fuera and all(p.nivel == pq.NIVEL_AVISO for p in fuera)


def test_tiro_sin_xg_es_aviso(paquete_valido):
    ruta = paquete_valido["raiz"] / "events" / "7001.json"
    eventos = json.loads(ruta.read_text(encoding="utf-8"))
    for ev in eventos:
        if ev["type"]["name"] == "Shot":
            del ev["shot"]["statsbomb_xg"]
    reescribir(ruta, eventos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    sin_xg = [p for p in problemas if "statsbomb_xg" in p.mensaje]
    assert sin_xg and all(p.nivel == pq.NIVEL_AVISO for p in sin_xg)


# --- Alineaciones -------------------------------------------------------------
def test_sin_fichero_de_alineaciones_es_error(paquete_valido):
    (paquete_valido["raiz"] / "lineups" / "7001.json").unlink()
    _, problemas = _validar_unico(paquete_valido["raiz"])
    assert "no existe" in _codigos(problemas)


def test_equipos_de_alineacion_distintos_a_los_del_partido_es_error(paquete_valido):
    ruta = paquete_valido["raiz"] / "lineups" / "7001.json"
    equipos = json.loads(ruta.read_text(encoding="utf-8"))
    equipos[0]["team_id"] = 999
    reescribir(ruta, equipos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    assert "no coinciden" in _codigos(problemas)


def test_nadie_con_minutos_es_error(paquete_valido):
    ruta = paquete_valido["raiz"] / "lineups" / "7001.json"
    equipos = json.loads(ruta.read_text(encoding="utf-8"))
    for equipo in equipos:
        for jug in equipo["lineup"]:
            jug["positions"] = []
    reescribir(ruta, equipos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    assert "ninguna fila jugador-partido" in _codigos(problemas)


def test_posicion_fuera_del_catalogo_es_aviso(paquete_valido):
    ruta = paquete_valido["raiz"] / "lineups" / "7001.json"
    equipos = json.loads(ruta.read_text(encoding="utf-8"))
    equipos[0]["lineup"][0]["positions"][0]["position"] = "Libero"
    reescribir(ruta, equipos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    raras = [p for p in problemas if "catalogo" in p.mensaje]
    assert raras and all(p.nivel == pq.NIVEL_AVISO for p in raras)
    assert "Libero" in _codigos(raras)


def test_spell_sin_reloj_es_error(paquete_valido):
    ruta = paquete_valido["raiz"] / "lineups" / "7001.json"
    equipos = json.loads(ruta.read_text(encoding="utf-8"))
    del equipos[0]["lineup"][0]["positions"][0]["from"]
    reescribir(ruta, equipos)
    _, problemas = _validar_unico(paquete_valido["raiz"])
    assert "from" in _codigos(problemas)

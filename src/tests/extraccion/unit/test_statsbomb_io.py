"""Lectura de los ficheros de StatsBomb Open Data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extraccion import statsbomb_io as io
from src.tests import factories as fac


class TestReadJson:
    def test_lee_utf8(self, tmp_path: Path) -> None:
        """Hay nombres con caracteres fuera de latin-1 (p. ej. la 'g' turca)."""
        ruta = tmp_path / "x.json"
        ruta.write_text(json.dumps({"n": "Caglar Söyüncü"}), encoding="utf-8")
        assert io.read_json(ruta) == {"n": "Caglar Söyüncü"}

    def test_un_fichero_inexistente_falla(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            io.read_json(tmp_path / "no-existe.json")


class TestCompetitions:
    def test_load_competitions_devuelve_las_filas_crudas(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(tmp_path / "data", competiciones=[
            fac.fila_competicion(1, 100), fac.fila_competicion(1, 101),
        ])
        assert len(io.load_competitions(raiz)) == 2

    def test_list_competitions_agrupa_las_temporadas_por_competicion(self, tmp_path: Path) -> None:
        """`competitions.json` trae una fila por competicion-temporada; el selector
        necesita una entrada por competicion con sus temporadas dentro.
        """
        raiz = fac.escribir_open_data(tmp_path / "data", competiciones=[
            fac.fila_competicion(1, 100, competition_name="La Liga", season_name="2019/2020"),
            fac.fila_competicion(1, 101, competition_name="La Liga", season_name="2020/2021"),
            fac.fila_competicion(2, 200, competition_name="Premier League", season_name="2015/2016"),
        ])
        comps = io.list_competitions(raiz)
        assert len(comps) == 2
        la_liga = next(c for c in comps if c["competition_name"] == "La Liga")
        assert len(la_liga["seasons"]) == 2

    def test_marca_que_temporadas_tienen_partidos_en_disco(self, tmp_path: Path) -> None:
        """El dataset lista temporadas cuyo fichero de partidos no esta descargado."""
        raiz = fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[
                fac.fila_competicion(1, 100, season_name="2019/2020"),
                fac.fila_competicion(1, 101, season_name="2020/2021"),
            ],
            partidos={(1, 100): [fac.partido(1)]},   # solo la primera existe
        )
        (comp,) = io.list_competitions(raiz)
        por_temporada = {s["season_id"]: s["available"] for s in comp["seasons"]}
        assert por_temporada == {100: True, 101: False}

    def test_las_competiciones_salen_ordenadas_por_nombre(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(tmp_path / "data", competiciones=[
            fac.fila_competicion(1, 100, competition_name="Premier League"),
            fac.fila_competicion(2, 200, competition_name="Bundesliga"),
            fac.fila_competicion(3, 300, competition_name="La Liga"),
        ])
        nombres = [c["competition_name"] for c in io.list_competitions(raiz)]
        assert nombres == ["Bundesliga", "La Liga", "Premier League"]

    def test_las_temporadas_salen_ordenadas_por_nombre(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(tmp_path / "data", competiciones=[
            fac.fila_competicion(1, 102, season_name="2020/2021"),
            fac.fila_competicion(1, 100, season_name="2018/2019"),
            fac.fila_competicion(1, 101, season_name="2019/2020"),
        ])
        (comp,) = io.list_competitions(raiz)
        nombres = [s["season_name"] for s in comp["seasons"]]
        assert nombres == ["2018/2019", "2019/2020", "2020/2021"]

    def test_se_conservan_los_metadatos_de_la_competicion(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(tmp_path / "data", competiciones=[
            fac.fila_competicion(1, 100, country_name="Spain", gender="female"),
        ])
        (comp,) = io.list_competitions(raiz)
        assert comp["country_name"] == "Spain"
        assert comp["competition_gender"] == "female"


class TestMatchesEventosYAlineaciones:
    def test_load_matches_lee_la_competicion_temporada(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[fac.fila_competicion(1, 100)],
            partidos={(1, 100): [fac.partido(1), fac.partido(2)]},
        )
        assert len(io.load_matches(1, 100, raiz)) == 2

    def test_una_temporada_sin_fichero_devuelve_lista_vacia(self, tmp_path: Path) -> None:
        """No es un error: el dataset no trae todas las temporadas que lista."""
        raiz = fac.escribir_open_data(tmp_path / "data", competiciones=[])
        assert io.load_matches(99, 999, raiz) == []

    def test_load_events_y_load_lineups(self, tmp_path: Path) -> None:
        match, eventos, alineaciones = fac.partido_minimo()
        mid = match["match_id"]
        raiz = fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[fac.fila_competicion(1, 100)],
            partidos={(1, 100): [match]},
            eventos={mid: eventos},
            alineaciones={mid: alineaciones},
        )
        assert len(io.load_events(mid, raiz)) == len(eventos)
        assert len(io.load_lineups(mid, raiz)) == 2

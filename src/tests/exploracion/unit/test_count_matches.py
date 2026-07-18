"""Recuento de partidos por competicion-temporada."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.exploracion.count_matches_by_competition import OUTPUT_PATH, build_match_counts
from src.tests import factories as fac


class TestBuildMatchCounts:
    def test_una_fila_por_competicion_temporada(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[fac.fila_competicion(1, 100), fac.fila_competicion(1, 101)],
            partidos={(1, 100): [fac.partido(1), fac.partido(2)]},
        )
        df = build_match_counts(raiz)
        assert len(df) == 2

    def test_cuenta_los_partidos_de_cada_temporada(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[
                fac.fila_competicion(1, 100, season_name="2019/2020"),
                fac.fila_competicion(1, 101, season_name="2020/2021"),
            ],
            partidos={(1, 100): [fac.partido(i) for i in range(3)]},
        )
        df = build_match_counts(raiz)
        por_temporada = dict(zip(df["season_id"], df["match_count"]))
        assert por_temporada == {100: 3, 101: 0}

    def test_una_temporada_sin_fichero_cuenta_cero(self, tmp_path: Path) -> None:
        """El dataset lista temporadas cuyos partidos no estan descargados: es
        justo lo que este script sirve para ver.
        """
        raiz = fac.escribir_open_data(
            tmp_path / "data", competiciones=[fac.fila_competicion(9, 900)])
        assert build_match_counts(raiz)["match_count"].tolist() == [0]

    def test_lleva_los_metadatos_de_la_competicion(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[fac.fila_competicion(
                11, 90, competition_name="La Liga", country_name="Spain",
                season_name="2019/2020")],
            partidos={(11, 90): [fac.partido(1)]},
        )
        fila = build_match_counts(raiz).iloc[0]
        assert fila["competition_id"] == 11
        assert fila["competition_name"] == "La Liga"
        assert fila["country_name"] == "Spain"
        assert fila["season_id"] == 90
        assert fila["season_name"] == "2019/2020"
        assert fila["match_count"] == 1

    def test_va_ordenado_por_competicion_y_temporada(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[
                fac.fila_competicion(2, 200, competition_name="Premier League",
                                     season_name="2015/2016"),
                fac.fila_competicion(1, 101, competition_name="La Liga",
                                     season_name="2020/2021"),
                fac.fila_competicion(1, 100, competition_name="La Liga",
                                     season_name="2019/2020"),
            ],
        )
        df = build_match_counts(raiz)
        assert df["competition_name"].tolist() == ["La Liga", "La Liga", "Premier League"]
        assert df["season_name"].tolist() == ["2019/2020", "2020/2021", "2015/2016"]

    def test_las_columnas_son_las_esperadas(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(
            tmp_path / "data", competiciones=[fac.fila_competicion(1, 100)])
        assert build_match_counts(raiz).columns.tolist() == [
            "competition_id", "competition_name", "country_name",
            "season_id", "season_name", "match_count",
        ]

    def test_sin_competiciones_devuelve_un_dataframe_vacio(self, tmp_path: Path) -> None:
        raiz = fac.escribir_open_data(tmp_path / "data", competiciones=[])
        assert build_match_counts(raiz).empty


class TestSalida:
    def test_el_csv_va_a_outputs(self) -> None:
        """`outputs/` esta en .gitignore: es una salida derivada, no fuente."""
        assert OUTPUT_PATH == Path("outputs/csv/matches_by_competition_season.csv")

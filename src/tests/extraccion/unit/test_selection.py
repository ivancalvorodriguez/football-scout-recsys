"""Seleccion de que partidos procesar (por nombre y por % de jornadas)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from src.extraccion import selection as sel
from src.extraccion.config import DEFAULT_DB_PATH
from src.tests import factories as fac


def _liga(n_jornadas: int) -> list[dict[str, Any]]:
    """Una liga limpia: un partido por jornada, en fechas crecientes."""
    return [
        fac.partido(i, jornada=i, fecha=f"2020-01-{i:02d}")
        for i in range(1, n_jornadas + 1)
    ]


class TestSubconjuntoPorJornadas:
    def test_el_porcentaje_se_aplica_sobre_las_jornadas(self) -> None:
        seleccion = sel.select_matchday_subset(_liga(10), 50)
        assert [m["match_week"] for m in seleccion] == [1, 2, 3, 4, 5]

    def test_el_cien_por_cien_coge_todo(self) -> None:
        assert len(sel.select_matchday_subset(_liga(10), 100)) == 10

    def test_se_redondea_hacia_arriba(self) -> None:
        """25% de 10 jornadas = 2.5 -> 3: es preferible pasarse a quedarse corto."""
        assert len(sel.select_matchday_subset(_liga(10), 25)) == 3

    def test_un_porcentaje_minusculo_deja_al_menos_una_jornada(self) -> None:
        assert len(sel.select_matchday_subset(_liga(10), 0.1)) == 1

    def test_se_cogen_las_primeras_jornadas(self) -> None:
        """Coger las primeras (y no una muestra aleatoria) hace el subconjunto
        reproducible entre ejecuciones.
        """
        seleccion = sel.select_matchday_subset(_liga(20), 15)
        assert max(m["match_week"] for m in seleccion) == 3

    def test_varios_partidos_por_jornada_entran_todos(self) -> None:
        partidos = [
            fac.partido(1, jornada=1, fecha="2020-01-01"),
            fac.partido(2, jornada=1, fecha="2020-01-02"),
            fac.partido(3, jornada=2, fecha="2020-01-08"),
            fac.partido(4, jornada=2, fecha="2020-01-09"),
        ]
        seleccion = sel.select_matchday_subset(partidos, 50)
        assert [m["match_id"] for m in seleccion] == [1, 2]

    def test_el_resultado_va_ordenado_por_fecha(self) -> None:
        partidos = [
            fac.partido(3, jornada=1, fecha="2020-01-03"),
            fac.partido(1, jornada=1, fecha="2020-01-01"),
            fac.partido(2, jornada=2, fecha="2020-01-02"),
        ]
        seleccion = sel.select_matchday_subset(partidos, 100)
        assert [m["match_id"] for m in seleccion] == [1, 2, 3]

    def test_el_porcentaje_se_recorta_al_rango_valido(self) -> None:
        assert len(sel.select_matchday_subset(_liga(4), 500)) == 4


class TestSubconjuntoSinJornadas:
    def test_sin_jornadas_se_cae_a_porcentaje_de_partidos_por_fecha(self) -> None:
        """Fases KO: `match_week` no describe una jornada de liga."""
        partidos = [
            fac.partido(i, jornada=None, fecha=f"2020-01-{i:02d}") for i in range(1, 11)
        ]
        seleccion = sel.select_matchday_subset(partidos, 30)
        assert [m["match_id"] for m in seleccion] == [1, 2, 3]

    def test_una_sola_jornada_distinta_no_es_una_liga_por_jornadas(self) -> None:
        partidos = [fac.partido(i, jornada=1, fecha=f"2020-01-{i:02d}") for i in range(1, 5)]
        seleccion = sel.select_matchday_subset(partidos, 50)
        assert len(seleccion) == 2   # 50% de los partidos, no de las jornadas

    def test_si_faltan_jornadas_en_mas_de_la_mitad_se_usa_la_fecha(self) -> None:
        """Datos incompletos: fiarse de `match_week` seleccionaria casi nada."""
        partidos = [
            fac.partido(1, jornada=1, fecha="2020-01-01"),
            fac.partido(2, jornada=2, fecha="2020-01-02"),
            fac.partido(3, jornada=None, fecha="2020-01-03"),
            fac.partido(4, jornada=None, fecha="2020-01-04"),
            fac.partido(5, jornada=None, fecha="2020-01-05"),
        ]
        seleccion = sel.select_matchday_subset(partidos, 40)
        assert [m["match_id"] for m in seleccion] == [1, 2]


class TestResolucionPorNombre:
    @pytest.fixture
    def raiz(self, tmp_path: Path) -> Path:
        return fac.escribir_open_data(tmp_path / "data", competiciones=[
            fac.fila_competicion(1, 100, competition_name="Premier League",
                                 season_name="2015/2016"),
            fac.fila_competicion(1, 101, competition_name="Premier League",
                                 season_name="2016/2017"),
        ])

    def test_encuentra_la_competicion_por_nombre_exacto(self, raiz: Path) -> None:
        comp = sel.find_competition("Premier League", raiz)
        assert comp is not None
        assert comp["competition_id"] == 1

    def test_el_nombre_no_distingue_mayusculas_ni_espacios_sobrantes(self, raiz: Path) -> None:
        assert sel.find_competition("  premier league  ", raiz) is not None

    def test_una_competicion_inexistente_devuelve_none(self, raiz: Path) -> None:
        assert sel.find_competition("Liga Inventada", raiz) is None

    def test_el_nombre_parcial_no_vale_para_competiciones(self, raiz: Path) -> None:
        """A diferencia de las consultas del modelo, aqui se exige nombre exacto."""
        assert sel.find_competition("Premier", raiz) is None

    def test_encuentra_la_temporada_por_nombre(self, raiz: Path) -> None:
        comp = sel.find_competition("Premier League", raiz)
        season = sel.find_season(comp, "2016/2017")
        assert season is not None
        assert season["season_id"] == 101

    def test_una_temporada_inexistente_devuelve_none(self, raiz: Path) -> None:
        comp = sel.find_competition("Premier League", raiz)
        assert sel.find_season(comp, "1999/2000") is None


class TestModoInteractivo:
    @pytest.fixture
    def raiz(self, tmp_path: Path) -> Path:
        return fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[
                fac.fila_competicion(1, 100, competition_name="Premier League",
                                     season_name="2015/2016"),
                # Sin fichero de partidos: no debe ofrecerse.
                fac.fila_competicion(2, 200, competition_name="Liga Sin Datos",
                                     season_name="2019/2020"),
            ],
            partidos={(1, 100): _liga(10)},
        )

    def _respuestas(self, monkeypatch: pytest.MonkeyPatch, valores: list[str]) -> None:
        it: Iterator[str] = iter(valores)
        monkeypatch.setattr("builtins.input", lambda *_: next(it))

    def test_dialogo_completo(self, raiz: Path, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture) -> None:
        self._respuestas(monkeypatch, ["1", "1", "50", "outputs/db/prueba.db"])
        seleccion = sel.interactive_selection(raiz)
        assert seleccion["competition"]["competition_name"] == "Premier League"
        assert seleccion["season"]["season_name"] == "2015/2016"
        assert seleccion["pct"] == 50.0
        assert len(seleccion["matches"]) == 5
        assert seleccion["db_path"] == Path("outputs/db/prueba.db")

    def test_solo_se_ofrecen_competiciones_con_datos_en_disco(
        self, raiz: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        self._respuestas(monkeypatch, ["1", "1", "100", ""])
        sel.interactive_selection(raiz)
        salida = capsys.readouterr().out
        assert "Premier League" in salida
        assert "Liga Sin Datos" not in salida

    def test_los_valores_por_defecto_se_aplican_al_pulsar_enter(
        self, raiz: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._respuestas(monkeypatch, ["1", "1", "", ""])
        seleccion = sel.interactive_selection(raiz)
        assert seleccion["pct"] == 100.0
        assert seleccion["db_path"] == DEFAULT_DB_PATH

    def test_una_opcion_invalida_se_vuelve_a_pedir(
        self, raiz: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        self._respuestas(monkeypatch, ["0", "99", "no-es-un-numero", "1", "1", "100", ""])
        seleccion = sel.interactive_selection(raiz)
        assert seleccion["competition"]["competition_name"] == "Premier League"
        assert "no válida" in capsys.readouterr().out

    @pytest.mark.parametrize("pct_malo", ["0", "101", "-5", "hola"])
    def test_un_porcentaje_invalido_se_vuelve_a_pedir(
        self, raiz: Path, monkeypatch: pytest.MonkeyPatch, pct_malo: str,
        capsys: pytest.CaptureFixture,
    ) -> None:
        self._respuestas(monkeypatch, ["1", "1", pct_malo, "100", ""])
        seleccion = sel.interactive_selection(raiz)
        assert seleccion["pct"] == 100.0

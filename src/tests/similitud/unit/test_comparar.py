"""Funciones puras del comparador de normalizaciones (sin E/S ni figuras)."""

from __future__ import annotations

import numpy as np
import pytest

from src.similitud import comparar
from src.similitud.modelo import ModeloSimilitud


def _comp(idx: int, nombre: str, jaccard: float = 0.0, tops=None) -> comparar._Comparacion:
    return comparar._Comparacion(
        idx=idx, nombre=nombre, tops=tops or {"por_liga": [], "global": []}, jaccard=jaccard,
    )


def _modelo_con_ligas(ligas: dict[str, list[str]]) -> ModeloSimilitud:
    n = len(ligas)
    return ModeloSimilitud(
        formulacion="2", entidad="jugador", S=np.zeros((n, n)),
        entity_ids=np.arange(n), entity_names=[f"E{i}" for i in range(n)],
        feat_names=["f0"], feat_display=np.zeros((n, 1)),
        meta={"ligas_por_entidad": ligas},
    )


class TestJaccard:
    def test_tops_identicos(self) -> None:
        assert comparar._jaccard(["a", "b"], ["b", "a"]) == 1.0

    def test_el_orden_no_importa(self) -> None:
        """Jaccard mide QUE candidatos coinciden, no en que posicion."""
        assert comparar._jaccard(["a", "b", "c"], ["c", "b", "a"]) == 1.0

    def test_sin_solape(self) -> None:
        assert comparar._jaccard(["a"], ["b"]) == 0.0

    def test_solape_parcial(self) -> None:
        assert comparar._jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)

    def test_dos_tops_vacios_no_dividen_por_cero(self) -> None:
        assert comparar._jaccard([], []) == 0.0

    def test_uno_vacio(self) -> None:
        assert comparar._jaccard(["a"], []) == 0.0


class TestRankComun:
    def test_devuelve_las_posiciones_en_cada_modo(self) -> None:
        comun = comparar._rank_comun(["a", "b", "c"], ["c", "a", "b"])
        assert comun == {"a": (1, 2), "b": (2, 3), "c": (3, 1)}

    def test_solo_incluye_los_que_estan_en_ambos(self) -> None:
        assert comparar._rank_comun(["a", "b"], ["b", "z"]) == {"b": (2, 1)}

    def test_sin_comunes(self) -> None:
        assert comparar._rank_comun(["a"], ["b"]) == {}

    def test_los_rangos_empiezan_en_uno(self) -> None:
        assert comparar._rank_comun(["a"], ["a"]) == {"a": (1, 1)}


class TestEtiquetas:
    def test_usa_el_nombre_cuando_es_unico(self) -> None:
        assert comparar._etiquetas([_comp(0, "Messi"), _comp(1, "Pedri")]) == ["Messi", "Pedri"]

    def test_desambigua_los_nombres_repetidos_con_el_indice(self) -> None:
        """Dos entidades distintas pueden compartir nombre: sin el indice, dos
        barras del grafico serian indistinguibles.
        """
        etiquetas = comparar._etiquetas([_comp(3, "Danilo"), _comp(9, "Danilo")])
        assert etiquetas == ["Danilo [#3]", "Danilo [#9]"]

    def test_solo_desambigua_los_repetidos(self) -> None:
        etiquetas = comparar._etiquetas(
            [_comp(3, "Danilo"), _comp(9, "Danilo"), _comp(5, "Messi")])
        assert etiquetas == ["Danilo [#3]", "Danilo [#9]", "Messi"]

    def test_sin_comparaciones(self) -> None:
        assert comparar._etiquetas([]) == []


class TestComparacion:
    def test_candidatos_extrae_los_nombres_de_un_modo(self) -> None:
        c = _comp(0, "Messi", tops={"por_liga": [("A", 0.5), ("B", 0.3)], "global": []})
        assert c.candidatos("por_liga") == ["A", "B"]

    def test_la_identidad_es_el_indice_no_el_nombre(self) -> None:
        """Dos consultas distintas ("Messi", "Lionel Messi") pueden resolver a la
        misma entidad: el nombre es solo etiqueta de presentacion.
        """
        assert _comp(7, "Messi").idx == 7


class TestLigasDelModelo:
    def test_recoge_todas_las_ligas_del_meta(self) -> None:
        modelo = _modelo_con_ligas({"1": ["11-90"], "2": ["16-90", "11-90"]})
        assert comparar._ligas_del_modelo(modelo) == {"11-90", "16-90"}

    def test_un_meta_sin_ligas_no_rompe(self) -> None:
        modelo = _modelo_con_ligas({})
        modelo.meta = {}
        assert comparar._ligas_del_modelo(modelo) == set()


class TestAvisoUnaSolaLiga:
    def test_avisa_si_solo_hay_una_liga(self, capsys: pytest.CaptureFixture) -> None:
        """Con una sola liga-temporada, el z-score por liga y el global se calculan
        sobre las mismas filas: son la MISMA transformacion y el Jaccard sale 1.00
        por construccion. Es una comparacion vacia, no un hallazgo.
        """
        comparar._avisar_si_una_sola_liga(_modelo_con_ligas({"1": ["11-90"], "2": ["11-90"]}))
        salida = capsys.readouterr().out
        assert "[AVISO]" in salida
        assert "11-90" in salida

    def test_no_avisa_con_varias_ligas(self, capsys: pytest.CaptureFixture) -> None:
        comparar._avisar_si_una_sola_liga(_modelo_con_ligas({"1": ["11-90"], "2": ["16-90"]}))
        assert capsys.readouterr().out == ""

    def test_sin_ligas_avisa_sin_romper(self, capsys: pytest.CaptureFixture) -> None:
        comparar._avisar_si_una_sola_liga(_modelo_con_ligas({}))
        assert "[AVISO]" in capsys.readouterr().out

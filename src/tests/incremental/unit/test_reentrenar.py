"""Que combinaciones reentrena el CLI segun sus flags.

Es la pieza que separa las dos formas de pedirlo: el producto cartesiano de
`--entidad/--formulacion/--normalizacion` (la del TFG, para comparar) y
`--servibles`, la pareja que sirve la app —una formulacion distinta por
entidad—, que no se puede expresar con las tres flags.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.incremental.reentrenar import _combinaciones
from src.similitud.config import DISTANCIA_SERVIBLE, MODELOS_SERVIBLES


def _args(**kw) -> SimpleNamespace:
    base = {"servibles": False, "entidad": "ambas", "formulacion": "ambas",
            "normalizacion": "ambas", "distancia": "euclidea"}
    return SimpleNamespace(**{**base, **kw})


class TestProductoCartesiano:
    def test_todo_ambas_son_las_ocho_combinaciones(self) -> None:
        assert len(_combinaciones(_args())) == 2 * 2 * 2

    def test_fijar_una_flag_la_recorta(self) -> None:
        combinaciones = _combinaciones(_args(entidad="equipo", formulacion="2"))
        assert combinaciones == [("equipo", "2", "por_liga", "euclidea"),
                                 ("equipo", "2", "global", "euclidea")]

    def test_la_distancia_multiplica_como_las_demas(self) -> None:
        """Es un eje mas del producto cartesiano, no un ajuste dentro de un modelo."""
        assert len(_combinaciones(_args(distancia="todas"))) == 2 * 2 * 2 * 4

    def test_por_defecto_solo_la_euclidea(self) -> None:
        """Sin pedirlo, reentrenar sigue significando lo de siempre."""
        assert {c[3] for c in _combinaciones(_args())} == {"euclidea"}


class TestServibles:
    def test_una_por_entidad(self) -> None:
        combinaciones = _combinaciones(_args(servibles=True))
        assert len(combinaciones) == len(MODELOS_SERVIBLES)
        assert {c[0] for c in combinaciones} == {"jugador", "equipo"}

    def test_cada_entidad_con_su_receta(self) -> None:
        for entidad, formulacion, normalizacion, _d in _combinaciones(
                _args(servibles=True)):
            assert (formulacion, normalizacion) == MODELOS_SERVIBLES[entidad]

    def test_lo_servible_es_siempre_euclideo(self) -> None:
        """La app no elige geometria, igual que no elige formulacion: sirve la
        euclidea, que es la de los artefactos de fabrica."""
        assert {c[3] for c in _combinaciones(
            _args(servibles=True, distancia="todas"))} == {DISTANCIA_SERVIBLE}

    def test_manda_sobre_las_otras_flags(self) -> None:
        """La app no las pasa, pero un `--servibles --entidad jugador` a mano no
        puede acabar reentrenando media pareja."""
        assert _combinaciones(_args(servibles=True, entidad="jugador")) == \
            _combinaciones(_args(servibles=True))

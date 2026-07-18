"""Minutos jugados y posiciones por jugador, derivados de `lineups` + eventos.

`minutes_played` es el denominador de todo el per-90 aguas abajo: si se calcula
mal, todas las metricas de similitud se escalan mal.
"""

from __future__ import annotations

import pytest

from src.extraccion import minutes as M
from src.tests import factories as fac


class TestParseClock:
    @pytest.mark.parametrize(
        ("texto", "esperado"),
        [("00:00", 0.0), ("45:30", 45.5), ("90:00", 90.0), ("120:15", 120.25)],
    )
    def test_convierte_mm_ss_a_minutos(self, texto: str, esperado: float) -> None:
        assert M._parse_clock(texto) == pytest.approx(esperado)

    def test_none_y_vacio_no_aplican(self) -> None:
        """`to=None` significa "jugo hasta el final", no "jugo 0 minutos"."""
        assert M._parse_clock(None) is None
        assert M._parse_clock("") is None


class TestMatchEndMinute:
    def test_es_el_minuto_del_ultimo_evento_jugable(self) -> None:
        eventos = [
            fac.pase((10, 40), (30, 40), minute=0, second=10),
            fac.pase((10, 40), (30, 40), minute=94, second=30),
        ]
        assert M.match_end_minute(eventos) == pytest.approx(94.5)

    def test_incluye_el_descuento(self) -> None:
        """Intencionado: un titular que juega todo queda con ~95', no con 90'."""
        eventos = [fac.pase((10, 40), (30, 40), minute=96, second=0, period=2)]
        assert M.match_end_minute(eventos) == pytest.approx(96.0)

    def test_la_tanda_de_penaltis_no_alarga_el_partido(self) -> None:
        eventos = [
            fac.pase((10, 40), (30, 40), minute=120, second=0, period=4),
            fac.tiro((108, 40), minute=135, second=0, period=5),
        ]
        assert M.match_end_minute(eventos) == pytest.approx(120.0)

    def test_sin_eventos_el_partido_dura_cero(self) -> None:
        assert M.match_end_minute([]) == 0.0


class TestPlayerMinutes:
    EVENTOS = [fac.pase((10, 40), (30, 40), minute=94, second=30)]

    def _alineacion(self, jugadores: list[dict]) -> list[dict]:
        return [fac.lineup_equipo(1, "Equipo A", jugadores)]

    def test_un_titular_sin_hora_de_salida_juega_hasta_el_final(self) -> None:
        lineups = self._alineacion([
            fac.lineup_jugador(10, "Titular", [fac.spell(23, "Center Forward")]),
        ])
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert rec["minutes_played"] == pytest.approx(94.5)

    def test_un_jugador_sustituido_juega_hasta_su_hora_de_salida(self) -> None:
        lineups = self._alineacion([
            fac.lineup_jugador(10, "Sustituido",
                               [fac.spell(23, "Center Forward", "00:00", "60:00")]),
        ])
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert rec["minutes_played"] == pytest.approx(60.0)

    def test_un_suplente_juega_desde_su_entrada_hasta_el_final(self) -> None:
        lineups = self._alineacion([
            fac.lineup_jugador(11, "Suplente",
                               [fac.spell(23, "Center Forward", "60:00", None)]),
        ])
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert rec["minutes_played"] == pytest.approx(34.5)

    def test_un_convocado_que_no_juega_no_aparece(self) -> None:
        """Sin `positions` no disputo minutos: no debe generar fila (seria un
        jugador-partido con 0' y todas las metricas a cero).
        """
        lineups = self._alineacion([fac.lineup_jugador(12, "Banquillo", [])])
        assert M.player_minutes(lineups, self.EVENTOS) == []

    def test_un_spell_de_duracion_cero_no_genera_fila(self) -> None:
        lineups = self._alineacion([
            fac.lineup_jugador(13, "Entra y sale",
                               [fac.spell(23, "Center Forward", "90:00", "90:00")]),
        ])
        assert M.player_minutes(lineups, self.EVENTOS) == []

    def test_los_contextos_de_equipo_se_copian_a_cada_fila(self) -> None:
        lineups = self._alineacion([
            fac.lineup_jugador(10, "Titular", [fac.spell(23, "Center Forward")]),
        ])
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert rec["team_id"] == 1
        assert rec["team_name"] == "Equipo A"
        assert rec["player_id"] == 10

    def test_se_prefiere_el_apodo_al_nombre_completo(self) -> None:
        """StatsBomb trae 'Lionel Andres Messi Cuccittini' y el apodo 'Lionel Messi'."""
        lineups = self._alineacion([
            fac.lineup_jugador(10, "Nombre Completo Largo",
                               [fac.spell(23, "Center Forward")], apodo="Apodo"),
        ])
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert rec["player_name"] == "Apodo"

    def test_sin_apodo_se_usa_el_nombre_completo(self) -> None:
        lineups = self._alineacion([
            fac.lineup_jugador(10, "Nombre Completo", [fac.spell(23, "Center Forward")]),
        ])
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert rec["player_name"] == "Nombre Completo"


class TestPosicionesMultiples:
    EVENTOS = [fac.pase((10, 40), (30, 40), minute=90, second=0)]

    def test_la_posicion_principal_es_la_de_mas_minutos(self) -> None:
        lineups = [fac.lineup_equipo(1, "Equipo A", [
            fac.lineup_jugador(10, "Polivalente", [
                fac.spell(14, "Center Midfield", "00:00", "30:00"),   # 30'
                fac.spell(23, "Center Forward", "30:00", None),       # 60'
            ]),
        ])]
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert rec["primary_position_id"] == 23
        assert rec["primary_position_name"] == "Center Forward"

    def test_los_minutos_totales_suman_todos_los_tramos(self) -> None:
        lineups = [fac.lineup_equipo(1, "Equipo A", [
            fac.lineup_jugador(10, "Polivalente", [
                fac.spell(14, "Center Midfield", "00:00", "30:00"),
                fac.spell(23, "Center Forward", "30:00", None),
            ]),
        ])]
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert rec["minutes_played"] == pytest.approx(90.0)

    def test_el_detalle_por_posicion_va_ordenado_de_mas_a_menos_minutos(self) -> None:
        lineups = [fac.lineup_equipo(1, "Equipo A", [
            fac.lineup_jugador(10, "Polivalente", [
                fac.spell(14, "Center Midfield", "00:00", "30:00"),
                fac.spell(23, "Center Forward", "30:00", None),
            ]),
        ])]
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert [p["position_id"] for p in rec["positions"]] == [23, 14]
        assert [p["minutes"] for p in rec["positions"]] == [60.0, 30.0]

    def test_volver_a_una_posicion_acumula_sus_minutos(self) -> None:
        """El jugador ocupa 14, se va a 23 y vuelve a 14: 14 suma 20+40 = 60'."""
        lineups = [fac.lineup_equipo(1, "Equipo A", [
            fac.lineup_jugador(10, "Ida y vuelta", [
                fac.spell(14, "Center Midfield", "00:00", "20:00"),
                fac.spell(23, "Center Forward", "20:00", "50:00"),
                fac.spell(14, "Center Midfield", "50:00", None),
            ]),
        ])]
        (rec,) = M.player_minutes(lineups, self.EVENTOS)
        assert len(rec["positions"]) == 2
        assert rec["primary_position_id"] == 14
        posiciones = {p["position_id"]: p["minutes"] for p in rec["positions"]}
        assert posiciones == {14: pytest.approx(60.0), 23: pytest.approx(30.0)}


class TestVariosEquipos:
    def test_se_procesan_los_dos_equipos_del_partido(self) -> None:
        eventos = [fac.pase((10, 40), (30, 40), minute=90, second=0)]
        lineups = [
            fac.lineup_equipo(1, "Equipo A", [
                fac.lineup_jugador(10, "A1", [fac.spell(23, "Center Forward")]),
            ]),
            fac.lineup_equipo(2, "Equipo B", [
                fac.lineup_jugador(20, "B1", [fac.spell(4, "Center Back")]),
                fac.lineup_jugador(21, "B2", [fac.spell(1, "Goalkeeper")]),
            ]),
        ]
        recs = M.player_minutes(lineups, eventos)
        assert len(recs) == 3
        assert {r["team_id"] for r in recs} == {1, 2}
        assert {r["player_id"] for r in recs} == {10, 20, 21}

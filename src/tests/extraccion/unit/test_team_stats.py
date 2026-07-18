"""Metricas por (equipo, partido).

Ademas de los conteos, cubre lo que solo existe a nivel equipo: cuotas
(possession_pct, field_tilt), PPDA y las metricas derivadas de secuencias.
"""

from __future__ import annotations

import pytest

from src.extraccion import config
from src.extraccion.team_stats import _max_x_reached, compute_team_stats
from src.tests import factories as fac

A = fac.EQUIPO_A["id"]
B = fac.EQUIPO_B["id"]


def stats(eventos: list[dict]) -> dict[int, dict]:
    return compute_team_stats(eventos)


class TestCuotas:
    def test_la_posesion_es_la_cuota_de_toques(self) -> None:
        eventos = [
            *[fac.recepcion((50, 40), jug=10, equipo=fac.EQUIPO_A) for _ in range(3)],
            fac.recepcion((50, 40), jug=20, equipo=fac.EQUIPO_B),
        ]
        r = stats(eventos)
        assert r[A]["possession_pct"] == pytest.approx(0.75)
        assert r[B]["possession_pct"] == pytest.approx(0.25)

    def test_las_posesiones_de_los_dos_equipos_suman_uno(self) -> None:
        eventos = [
            fac.recepcion((50, 40), jug=10, equipo=fac.EQUIPO_A),
            fac.conduccion((50, 40), (60, 40), jug=20, equipo=fac.EQUIPO_B),
            fac.recepcion((70, 40), jug=20, equipo=fac.EQUIPO_B),
        ]
        r = stats(eventos)
        assert r[A]["possession_pct"] + r[B]["possession_pct"] == pytest.approx(1.0)

    def test_el_field_tilt_es_la_cuota_de_toques_en_el_ultimo_tercio(self) -> None:
        """Mide dominio territorial: donde se juega, no cuanto se tiene el balon."""
        eventos = [
            fac.recepcion((90, 40), jug=10, equipo=fac.EQUIPO_A),   # ultimo tercio
            fac.recepcion((90, 40), jug=10, equipo=fac.EQUIPO_A),
            fac.recepcion((30, 40), jug=10, equipo=fac.EQUIPO_A),   # no cuenta
            fac.recepcion((90, 40), jug=20, equipo=fac.EQUIPO_B),
        ]
        r = stats(eventos)
        assert r[A]["field_tilt"] == pytest.approx(2 / 3)
        assert r[B]["field_tilt"] == pytest.approx(1 / 3)

    def test_sin_toques_las_cuotas_no_dividen_por_cero(self) -> None:
        eventos = [fac.presion((50, 40), jug=10, equipo=fac.EQUIPO_A)]
        r = stats(eventos)
        assert r[A]["possession_pct"] == 0.0
        assert r[A]["field_tilt"] == 0.0


class TestPPDA:
    def test_ppda_son_los_pases_del_rival_entre_las_acciones_defensivas_propias(self) -> None:
        eventos = [
            *[fac.pase((30, 40), (40, 40), jug=20, equipo=fac.EQUIPO_B) for _ in range(10)],
            fac.evento("Interception", jug=10, loc=(50, 40), equipo=fac.EQUIPO_A),
            fac.evento("Interception", jug=11, loc=(50, 40), equipo=fac.EQUIPO_A),
        ]
        r = stats(eventos)
        assert r[A]["ppda"] == pytest.approx(5.0)   # 10 pases de B / 2 acciones de A

    def test_sin_acciones_defensivas_el_ppda_no_esta_definido(self) -> None:
        """None, no 0: "no presiono nunca" no es "presiono infinito"."""
        eventos = [fac.pase((30, 40), (40, 40), jug=20, equipo=fac.EQUIPO_B)]
        r = stats(eventos)
        assert r[B]["ppda"] is None

    def test_el_numerador_solo_cuenta_pases_en_el_propio_60_por_ciento(self) -> None:
        eventos = [
            fac.pase((72, 40), (80, 40), jug=20, equipo=fac.EQUIPO_B),   # cuenta
            fac.pase((73, 40), (80, 40), jug=20, equipo=fac.EQUIPO_B),   # no cuenta
            fac.evento("Interception", jug=10, loc=(50, 40), equipo=fac.EQUIPO_A),
        ]
        assert stats(eventos)[A]["ppda"] == pytest.approx(1.0)

    def test_el_denominador_solo_cuenta_acciones_en_el_60_por_ciento_de_ataque(self) -> None:
        eventos = [
            fac.pase((30, 40), (40, 40), jug=20, equipo=fac.EQUIPO_B),
            fac.evento("Interception", jug=10, loc=(48, 40), equipo=fac.EQUIPO_A),   # cuenta
            fac.evento("Interception", jug=11, loc=(47, 40), equipo=fac.EQUIPO_A),   # no
        ]
        assert stats(eventos)[A]["ppda"] == pytest.approx(1.0)

    @pytest.mark.parametrize("tipo", ["Interception", "Foul Committed"])
    def test_acciones_que_cuentan_como_presion_defensiva(self, tipo: str) -> None:
        eventos = [
            fac.pase((30, 40), (40, 40), jug=20, equipo=fac.EQUIPO_B),
            fac.evento(tipo, jug=10, loc=(50, 40), equipo=fac.EQUIPO_A),
        ]
        assert stats(eventos)[A]["ppda"] == pytest.approx(1.0)

    def test_el_tackle_cuenta_como_accion_defensiva(self) -> None:
        eventos = [
            fac.pase((30, 40), (40, 40), jug=20, equipo=fac.EQUIPO_B),
            fac.duelo_tackle((50, 40), jug=10, equipo=fac.EQUIPO_A),
        ]
        assert stats(eventos)[A]["ppda"] == pytest.approx(1.0)

    def test_el_evento_pressure_no_entra_en_el_denominador_del_ppda(self) -> None:
        """PPDA es una definicion cerrada (Opta): tackles, intercepciones y faltas.
        Las presiones son una metrica aparte, propia de StatsBomb.
        """
        eventos = [
            fac.pase((30, 40), (40, 40), jug=20, equipo=fac.EQUIPO_B),
            fac.presion((50, 40), jug=10, equipo=fac.EQUIPO_A),
        ]
        assert stats(eventos)[A]["ppda"] is None


class TestConteos:
    def test_pases_progresivos_y_al_ultimo_tercio(self) -> None:
        r = stats([fac.pase((20, 40), (90, 40), jug=10)])[A]
        assert r["passes"] == 1
        assert r["passes_completed"] == 1
        assert r["progressive_passes"] == 1
        assert r["passes_into_final_third"] == 1

    def test_el_balon_parado_se_excluye_de_la_progresion(self) -> None:
        r = stats([fac.pase((120, 0), (110, 40), jug=10, tipo_pase="Corner")])[A]
        assert r["passes"] == 1
        assert r["progressive_passes"] == 0
        assert r["passes_into_penalty_area"] == 0

    def test_el_xt_de_equipo_suma_pases_y_conducciones(self) -> None:
        r = stats([
            fac.pase((20, 40), (60, 40), jug=10),
            fac.conduccion((60, 40), (100, 40), jug=11),
        ])[A]
        assert r["xt"] > 0.0

    def test_tiros_y_penaltis(self) -> None:
        r = stats([
            fac.tiro((100, 40), xg=0.2, resultado="Goal", jug=10),
            fac.tiro((108, 40), xg=0.78, resultado="Goal", tipo_tiro="Penalty", jug=11),
        ])[A]
        assert r["shots"] == 2
        assert r["np_shots"] == 1
        assert r["goals"] == 2
        assert r["xg"] == pytest.approx(0.98)
        assert r["npxg"] == pytest.approx(0.2)

    def test_el_xg_de_juego_abierto_excluye_el_balon_parado(self) -> None:
        r = stats([
            fac.tiro((100, 40), xg=0.2, jug=10, tipo_tiro="Open Play"),
            fac.tiro((100, 40), xg=0.05, jug=11, tipo_tiro="Free Kick"),
            fac.tiro((108, 40), xg=0.78, jug=11, tipo_tiro="Penalty"),
        ])[A]
        assert r["open_play_xg"] == pytest.approx(0.2)
        assert r["xg"] == pytest.approx(1.03)

    def test_high_turnover_es_recuperar_cerca_de_la_porteria_rival(self) -> None:
        eventos = [
            fac.recuperacion((80, 40), jug=10),                       # a <40 m: cuenta
            fac.recuperacion((70, 40), jug=10),                       # lejos: no
            fac.evento("Interception", jug=11, loc=(80, 40)),         # cuenta
        ]
        assert stats(eventos)[A]["high_turnovers"] == 2

    def test_una_recuperacion_fallida_no_es_high_turnover(self) -> None:
        r = stats([fac.recuperacion((80, 40), jug=10, fallida=True)])[A]
        assert r["high_turnovers"] == 0

    def test_presiones_y_contrapresiones(self) -> None:
        r = stats([
            fac.presion((60, 40), jug=10),
            fac.presion((60, 40), jug=11, counterpress=True),
        ])[A]
        assert r["pressures"] == 2
        assert r["counterpressures"] == 1


class TestSCAyGCA:
    def test_sca_cuenta_las_dos_acciones_previas_al_tiro(self) -> None:
        eventos = [
            fac.pase((20, 40), (40, 40), jug=1),
            fac.pase((40, 40), (60, 40), jug=2),
            fac.pase((60, 40), (85, 40), jug=3),
            fac.tiro((85, 40), jug=4),
        ]
        assert stats(eventos)[A]["sca"] == 2

    def test_gca_solo_cuenta_si_el_tiro_es_gol(self) -> None:
        eventos = [
            fac.pase((40, 40), (60, 40), jug=2),
            fac.pase((60, 40), (85, 40), jug=3),
            fac.tiro((85, 40), jug=4, resultado="Goal"),
        ]
        r = stats(eventos)[A]
        assert r["sca"] == 2
        assert r["gca"] == 2

    def test_un_tiro_parado_no_suma_gca(self) -> None:
        eventos = [
            fac.pase((60, 40), (85, 40), jug=3),
            fac.tiro((85, 40), jug=4, resultado="Saved"),
        ]
        r = stats(eventos)[A]
        assert r["sca"] == 1
        assert r["gca"] == 0

    def test_un_tiro_sin_acciones_previas_no_suma_sca(self) -> None:
        assert stats([fac.tiro((85, 40), jug=4)])[A]["sca"] == 0


class TestSecuencias:
    def _secuencia_larga(self) -> list[dict]:
        """Posesion de juego abierto de A: 10 pases completados y tiro final."""
        eventos = [
            fac.pase((20 + i, 40), (21 + i, 40), jug=10, possession=1, minute=0, second=i)
            for i in range(10)
        ]
        eventos.append(fac.tiro((85, 40), jug=10, possession=1, minute=0, second=11))
        return eventos

    def test_secuencia_de_10_o_mas_pases(self) -> None:
        r = stats(self._secuencia_larga())[A]
        assert r["ten_plus_pass_sequences"] == 1

    def test_nueve_pases_no_llegan_al_umbral(self) -> None:
        eventos = [
            fac.pase((20 + i, 40), (21 + i, 40), jug=10, possession=1, minute=0, second=i)
            for i in range(9)
        ]
        assert stats(eventos)[A]["ten_plus_pass_sequences"] == 0

    def test_build_up_attack_es_una_secuencia_larga_que_acaba_en_peligro(self) -> None:
        assert stats(self._secuencia_larga())[A]["build_up_attacks"] == 1

    def test_una_secuencia_larga_sin_peligro_no_es_build_up_attack(self) -> None:
        eventos = [
            fac.pase((20 + i, 40), (21 + i, 40), jug=10, possession=1, minute=0, second=i)
            for i in range(10)
        ]
        r = stats(eventos)[A]
        assert r["ten_plus_pass_sequences"] == 1
        assert r["build_up_attacks"] == 0

    def test_los_pases_fallados_no_cuentan_en_la_secuencia(self) -> None:
        eventos = [
            fac.pase((20, 40), (30, 40), jug=10, possession=1),
            fac.pase((30, 40), (40, 40), jug=10, possession=1, completado=False),
        ]
        assert stats(eventos)[A]["passes_per_sequence"] == pytest.approx(1.0)

    def test_pases_por_secuencia_promedia_sobre_las_secuencias(self) -> None:
        eventos = [
            fac.pase((20, 40), (30, 40), jug=10, possession=1),
            fac.pase((30, 40), (40, 40), jug=10, possession=1),
            fac.pase((20, 40), (30, 40), jug=10, possession=2),
        ]
        assert stats(eventos)[A]["passes_per_sequence"] == pytest.approx(1.5)

    def test_el_balon_parado_no_genera_secuencia(self) -> None:
        """`build_up`, `direct speed` y compania describen juego abierto."""
        eventos = [
            fac.pase((20, 40), (30, 40), jug=10, possession=1, play_pattern="From Corner"),
        ]
        r = stats(eventos)[A]
        assert r["seq_count"] == 0
        assert r["passes_per_sequence"] == 0.0

    @pytest.mark.parametrize("patron", ["Regular Play", "From Counter", "From Kick Off"])
    def test_los_patrones_de_juego_abierto_si_generan_secuencia(self, patron: str) -> None:
        eventos = [fac.pase((20, 40), (30, 40), jug=10, possession=1, play_pattern=patron)]
        assert stats(eventos)[A]["seq_count"] == 1

    def test_la_distancia_de_inicio_se_expresa_en_metros(self) -> None:
        """Se mide desde la propia linea de gol; la BD guarda metros, no yardas."""
        eventos = [fac.pase((20, 40), (30, 40), jug=10, possession=1)]
        esperado = 20.0 * config.YARD_TO_M
        assert stats(eventos)[A]["sequence_start_distance"] == pytest.approx(esperado, abs=1e-3)

    def test_la_amplitud_es_el_doble_de_la_maxima_desviacion_al_eje(self) -> None:
        eventos = [
            fac.pase((20, 40), (30, 40), jug=10, possession=1),
            fac.pase((30, 20), (40, 20), jug=10, possession=1),   # 20 de desviacion
        ]
        esperado = 2 * 20.0 * config.YARD_TO_M
        assert stats(eventos)[A]["absolute_width"] == pytest.approx(esperado, abs=1e-3)

    def test_una_secuencia_por_el_centro_tiene_amplitud_cero(self) -> None:
        eventos = [fac.pase((20, 40), (30, 40), jug=10, possession=1)]
        assert stats(eventos)[A]["absolute_width"] == pytest.approx(0.0)

    def test_direct_speed_es_el_avance_por_segundo_en_metros(self) -> None:
        eventos = [
            fac.pase((20, 40), (30, 40), jug=10, possession=1, minute=0, second=0),
            fac.conduccion((30, 40), (70, 40), jug=10, possession=1, minute=0, second=10),
        ]
        # Avance: de x=20 a x=70 (max alcanzado) en 10 s.
        esperado = (70.0 - 20.0) * config.YARD_TO_M / 10.0
        assert stats(eventos)[A]["direct_speed"] == pytest.approx(esperado, abs=1e-4)

    def test_una_secuencia_instantanea_no_aporta_velocidad(self) -> None:
        """Duracion 0: dividir daria infinito."""
        eventos = [
            fac.pase((20, 40), (30, 40), jug=10, possession=1, minute=0, second=0),
            fac.pase((30, 40), (70, 40), jug=10, possession=1, minute=0, second=0),
        ]
        assert stats(eventos)[A]["direct_speed"] == 0.0

    def test_direct_attack_arranca_atras_avanza_mucho_y_acaba_en_peligro(self) -> None:
        eventos = [
            fac.pase((20, 40), (60, 40), jug=10, possession=1, minute=0, second=0),
            fac.tiro((110, 40), jug=11, possession=1, minute=0, second=8),
        ]
        assert stats(eventos)[A]["direct_attacks"] == 1

    def test_una_secuencia_que_arranca_pasado_el_medio_campo_no_es_direct_attack(self) -> None:
        eventos = [
            fac.pase((70, 40), (90, 40), jug=10, possession=1, minute=0, second=0),
            fac.tiro((110, 40), jug=11, possession=1, minute=0, second=4),
        ]
        assert stats(eventos)[A]["direct_attacks"] == 0

    def test_un_direct_attack_sin_peligro_no_cuenta(self) -> None:
        eventos = [
            fac.pase((20, 40), (60, 40), jug=10, possession=1, minute=0, second=0),
            fac.pase((60, 40), (90, 40), jug=11, possession=1, minute=0, second=8),
        ]
        assert stats(eventos)[A]["direct_attacks"] == 0

    def test_un_toque_en_el_area_basta_como_peligro(self) -> None:
        eventos = [
            fac.pase((20, 40), (60, 40), jug=10, possession=1, minute=0, second=0),
            fac.recepcion((110, 40), jug=11, possession=1, minute=0, second=8),
        ]
        assert stats(eventos)[A]["direct_attacks"] == 1

    def test_una_posesion_sin_coordenadas_no_genera_secuencia(self) -> None:
        assert stats([fac.evento("Half Start", jug=10, possession=1)])[A]["seq_count"] == 0

    def test_los_eventos_del_rival_no_entran_en_la_secuencia(self) -> None:
        """La presion de B durante la posesion de A no es un pase de A."""
        eventos = [
            fac.pase((20, 40), (30, 40), jug=10, possession=1,
                     equipo=fac.EQUIPO_A, possession_team=fac.EQUIPO_A),
            fac.pase((30, 40), (40, 40), jug=20, possession=1,
                     equipo=fac.EQUIPO_B, possession_team=fac.EQUIPO_A),
        ]
        r = stats(eventos)
        assert r[A]["seq_count"] == 1
        assert r[A]["passes_per_sequence"] == pytest.approx(1.0)
        assert r[B]["seq_count"] == 0


class TestMaxXReached:
    def test_considera_el_destino_del_pase_no_solo_su_origen(self) -> None:
        eventos = [fac.pase((20, 40), (95, 40), jug=10)]
        assert _max_x_reached(eventos) == pytest.approx(95.0)

    def test_considera_el_destino_de_la_conduccion(self) -> None:
        eventos = [fac.conduccion((20, 40), (60, 40), jug=10)]
        assert _max_x_reached(eventos) == pytest.approx(60.0)

    def test_sin_coordenadas_devuelve_cero(self) -> None:
        assert _max_x_reached([fac.evento("Half Start")]) == 0.0


class TestEstructura:
    def test_se_devuelve_una_fila_por_equipo(self) -> None:
        eventos = [
            fac.pase((20, 40), (30, 40), jug=10, equipo=fac.EQUIPO_A),
            fac.pase((20, 40), (30, 40), jug=20, equipo=fac.EQUIPO_B),
        ]
        assert set(stats(eventos)) == {A, B}

    def test_la_tanda_de_penaltis_se_ignora(self) -> None:
        eventos = [
            fac.tiro((100, 40), xg=0.2, resultado="Goal", jug=10, period=1),
            fac.tiro((108, 40), xg=0.78, resultado="Goal", jug=10, period=5),
        ]
        assert stats(eventos)[A]["shots"] == 1

    def test_los_valores_de_modelo_se_redondean_a_cinco_decimales(self) -> None:
        eventos = [fac.pase((20, 40), (60, 40), jug=10), fac.tiro((100, 40), xg=0.123456789, jug=10)]
        r = stats(eventos)[A]
        for clave in ("xg", "npxg", "xt", "open_play_xg"):
            assert r[clave] == round(r[clave], 5)

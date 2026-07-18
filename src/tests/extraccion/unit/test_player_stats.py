"""Metricas por (jugador, partido).

Cada prueba monta el minimo de eventos que activa una metrica y comprueba que
las demas no se contaminan. Las fases cubiertas son las de
`docs/metricas_finales.md`: Progresion, Creacion, Finalizacion, Duelos, Aereo y
Defensa.
"""

from __future__ import annotations

import pytest

from src.extraccion import config
from src.extraccion.player_stats import compute_player_stats
from src.tests import factories as fac


def stats(eventos: list[dict]) -> dict[int, dict]:
    return compute_player_stats(eventos)


class TestProgresion:
    def test_un_pase_completado_suma_a_intentados_y_completados(self) -> None:
        r = stats([fac.pase((20, 40), (60, 40), jug=10)])[10]
        assert r["passes"] == 1
        assert r["passes_completed"] == 1

    def test_un_pase_fallado_suma_a_intentados_pero_no_a_completados(self) -> None:
        r = stats([fac.pase((20, 40), (60, 40), jug=10, completado=False)])[10]
        assert r["passes"] == 1
        assert r["passes_completed"] == 0

    def test_un_pase_fallado_no_progresa_aunque_apunte_adelante(self) -> None:
        """Si el balon no llega al companero, no ha progresado nada."""
        r = stats([fac.pase((20, 40), (100, 40), jug=10, completado=False)])[10]
        assert r["progressive_passes"] == 0
        assert r["xt"] == 0.0

    def test_pase_progresivo(self) -> None:
        r = stats([fac.pase((20, 40), (60, 40), jug=10)])[10]
        assert r["progressive_passes"] == 1

    def test_pase_lateral_no_progresivo(self) -> None:
        r = stats([fac.pase((60, 20), (60, 60), jug=10)])[10]
        assert r["progressive_passes"] == 0

    def test_pase_al_ultimo_tercio_solo_si_lo_cruza(self) -> None:
        entra = stats([fac.pase((60, 40), (90, 40), jug=10)])[10]
        assert entra["passes_into_final_third"] == 1
        # Ya estaba en el ultimo tercio: no lo "mete" ahi.
        dentro = stats([fac.pase((85, 40), (95, 40), jug=10)])[10]
        assert dentro["passes_into_final_third"] == 0

    def test_pase_al_area_solo_si_la_cruza(self) -> None:
        entra = stats([fac.pase((90, 40), (110, 40), jug=10)])[10]
        assert entra["passes_into_penalty_area"] == 1
        dentro = stats([fac.pase((105, 30), (110, 40), jug=10)])[10]
        assert dentro["passes_into_penalty_area"] == 0

    def test_el_balon_parado_no_cuenta_como_progresion_ni_creacion(self) -> None:
        """Un corner al area no mide la capacidad de progresar del jugador."""
        r = stats([fac.pase((120, 0), (110, 40), jug=10, tipo_pase="Corner")])[10]
        assert r["passes"] == 1              # el intento si se cuenta
        assert r["passes_completed"] == 1
        assert r["progressive_passes"] == 0
        assert r["passes_into_penalty_area"] == 0
        assert r["deep_completions"] == 0
        assert r["xt"] == 0.0

    def test_conduccion_progresiva(self) -> None:
        r = stats([fac.conduccion((20, 40), (60, 40), jug=10)])[10]
        assert r["progressive_carries"] == 1

    def test_conduccion_corta_no_progresiva(self) -> None:
        r = stats([fac.conduccion((20, 40), (25, 40), jug=10)])[10]
        assert r["progressive_carries"] == 0

    def test_el_xt_solo_acumula_la_amenaza_ganada(self) -> None:
        """Se usa max(0, delta): un pase atras no resta amenaza acumulada."""
        r = stats([fac.pase((110, 40), (20, 40), jug=10)])[10]
        assert r["xt"] == 0.0

    def test_el_xt_suma_pases_y_conducciones(self) -> None:
        r = stats([
            fac.pase((20, 40), (60, 40), jug=10),
            fac.conduccion((60, 40), (100, 40), jug=10),
        ])[10]
        assert r["xt"] > 0.0

    def test_el_xt_se_redondea_a_cinco_decimales(self) -> None:
        r = stats([fac.pase((20, 40), (60, 40), jug=10)])[10]
        assert r["xt"] == round(r["xt"], 5)


class TestCreacion:
    def test_deep_completion(self) -> None:
        """Pase (no centro) que termina a <= 20 m de la porteria rival."""
        r = stats([fac.pase((90, 40), (105, 40), jug=10)])[10]
        assert r["deep_completions"] == 1

    def test_un_centro_no_es_deep_completion(self) -> None:
        """Criterio Wyscout: los centros se excluyen a proposito."""
        r = stats([fac.pase((90, 5), (105, 40), jug=10, cross=True)])[10]
        assert r["deep_completions"] == 0
        assert r["passes_into_penalty_area"] == 1   # pero si entra al area

    def test_un_pase_lejano_no_es_deep_completion(self) -> None:
        r = stats([fac.pase((20, 40), (60, 40), jug=10)])[10]
        assert r["deep_completions"] == 0

    def test_el_pase_clave_hereda_el_xg_del_tiro_que_asiste(self) -> None:
        eventos = [
            fac.pase((85, 40), (110, 40), jug=11, eid="kp-1"),
            fac.tiro((110, 40), xg=0.4, jug=10, key_pass_id="kp-1"),
        ]
        r = stats(eventos)
        assert r[11]["xa"] == pytest.approx(0.4)
        assert r[10]["xa"] == 0.0   # el rematador no se auto-asiste

    def test_sin_pase_clave_no_hay_xa(self) -> None:
        r = stats([fac.tiro((110, 40), xg=0.4, jug=10)])
        assert r[10]["xa"] == 0.0

    def test_un_key_pass_id_que_no_existe_no_rompe(self) -> None:
        r = stats([fac.tiro((110, 40), xg=0.4, jug=10, key_pass_id="no-existe")])
        assert r[10]["xa"] == 0.0


class TestShotCreatingActions:
    def test_se_acreditan_las_dos_ultimas_acciones_del_equipo(self) -> None:
        eventos = [
            fac.pase((20, 40), (40, 40), jug=1),
            fac.pase((40, 40), (60, 40), jug=2),
            fac.pase((60, 40), (85, 40), jug=3),
            fac.tiro((85, 40), jug=4),
        ]
        r = stats(eventos)
        assert r[3]["sca"] == 1   # ultima accion antes del tiro
        assert r[2]["sca"] == 1   # penultima
        assert r[1]["sca"] == 0   # la tercera ya no se acredita (criterio FBref)

    def test_el_tope_es_config_sca_max_actions(self) -> None:
        eventos = [fac.pase((20 + 5 * i, 40), (25 + 5 * i, 40), jug=i) for i in range(5)]
        eventos.append(fac.tiro((85, 40), jug=99))
        r = stats(eventos)
        acreditados = sum(1 for pid in range(5) if r[pid]["sca"] > 0)
        assert acreditados == config.SCA_MAX_ACTIONS

    def test_la_secuencia_no_cruza_la_frontera_de_posesion(self) -> None:
        """Lo que paso en la posesion anterior no creo este tiro."""
        eventos = [
            fac.pase((20, 40), (40, 40), jug=1, possession=1),
            fac.tiro((85, 40), jug=4, possession=2),
        ]
        assert stats(eventos)[1]["sca"] == 0

    def test_las_acciones_del_rival_no_crean_el_tiro(self) -> None:
        eventos = [
            fac.pase((20, 40), (40, 40), jug=20, equipo=fac.EQUIPO_B,
                     possession_team=fac.EQUIPO_A),
            fac.pase((40, 40), (60, 40), jug=2, equipo=fac.EQUIPO_A,
                     possession_team=fac.EQUIPO_A),
            fac.tiro((85, 40), jug=4, equipo=fac.EQUIPO_A, possession_team=fac.EQUIPO_A),
        ]
        r = stats(eventos)
        assert r[20]["sca"] == 0
        assert r[2]["sca"] == 1

    def test_un_pase_fallado_no_crea_el_tiro(self) -> None:
        eventos = [
            fac.pase((20, 40), (40, 40), jug=1),
            fac.pase((40, 40), (60, 40), jug=2, completado=False),
            fac.tiro((85, 40), jug=4),
        ]
        r = stats(eventos)
        assert r[2]["sca"] == 0
        assert r[1]["sca"] == 1   # se sigue buscando hacia atras

    def test_un_regate_fallado_no_crea_el_tiro(self) -> None:
        eventos = [
            fac.regate((60, 40), jug=2, completado=False),
            fac.tiro((85, 40), jug=4),
        ]
        assert stats(eventos)[2]["sca"] == 0

    def test_un_regate_completado_si_crea_el_tiro(self) -> None:
        eventos = [
            fac.regate((60, 40), jug=2, completado=True),
            fac.tiro((85, 40), jug=4),
        ]
        assert stats(eventos)[2]["sca"] == 1

    def test_un_tiro_previo_rechazado_crea_el_siguiente(self) -> None:
        """Rebote: Shot esta en SCA_ACTION_TYPES a proposito."""
        eventos = [
            fac.tiro((100, 40), jug=2, resultado="Blocked"),
            fac.tiro((105, 40), jug=4),
        ]
        assert stats(eventos)[2]["sca"] == 1

    def test_un_mismo_jugador_puede_sumar_las_dos_acciones(self) -> None:
        eventos = [
            fac.conduccion((60, 40), (80, 40), jug=7),
            fac.pase((80, 40), (85, 40), jug=7),
            fac.tiro((85, 40), jug=4),
        ]
        assert stats(eventos)[7]["sca"] == 2


class TestFinalizacion:
    def test_un_tiro_normal_cuenta_en_todas_las_metricas_sin_penalti(self) -> None:
        r = stats([fac.tiro((100, 40), xg=0.25, resultado="Saved", jug=10)])[10]
        assert r["shots"] == 1
        assert r["np_shots"] == 1
        assert r["xg"] == pytest.approx(0.25)
        assert r["npxg"] == pytest.approx(0.25)
        assert r["shots_on_target"] == 1
        assert r["goals"] == 0

    def test_un_gol_cuenta_como_gol_y_como_tiro_a_puerta(self) -> None:
        r = stats([fac.tiro((100, 40), xg=0.25, resultado="Goal", jug=10)])[10]
        assert r["goals"] == 1
        assert r["np_goals"] == 1
        assert r["shots_on_target"] == 1

    @pytest.mark.parametrize("resultado", ["Off T", "Blocked", "Wayward", "Post"])
    def test_los_tiros_no_a_puerta_no_suman_sot(self, resultado: str) -> None:
        r = stats([fac.tiro((100, 40), resultado=resultado, jug=10)])[10]
        assert r["shots_on_target"] == 0
        assert r["np_shots"] == 1

    def test_el_penalti_cuenta_como_tiro_y_gol_pero_no_como_np(self) -> None:
        """np:G-xG mide finalizacion en juego; el penalti la contaminaria."""
        r = stats([
            fac.tiro((108, 40), xg=0.78, resultado="Goal", tipo_tiro="Penalty", jug=10)
        ])[10]
        assert r["shots"] == 1
        assert r["goals"] == 1
        assert r["xg"] == pytest.approx(0.78)
        assert r["np_shots"] == 0
        assert r["np_goals"] == 0
        assert r["npxg"] == 0.0
        assert r["shots_on_target"] == 0

    def test_un_tiro_sin_xg_no_rompe(self) -> None:
        e = fac.tiro((100, 40), jug=10)
        e["shot"]["statsbomb_xg"] = None
        assert stats([e])[10]["xg"] == 0.0

    def test_el_xg_se_acumula_entre_tiros(self) -> None:
        r = stats([
            fac.tiro((100, 40), xg=0.1, jug=10),
            fac.tiro((110, 40), xg=0.25, jug=10),
        ])[10]
        assert r["shots"] == 2
        assert r["xg"] == pytest.approx(0.35)

    def test_toques_en_el_area_rival(self) -> None:
        r = stats([
            fac.recepcion((110, 40), jug=10),   # dentro
            fac.recepcion((90, 40), jug=10),    # fuera
        ])[10]
        assert r["touches_in_att_pen_area"] == 1

    def test_una_recepcion_fallida_no_es_un_toque_en_el_area(self) -> None:
        r = stats([fac.recepcion((110, 40), jug=10, completada=False)])[10]
        assert r["touches_in_att_pen_area"] == 0

    def test_un_tiro_dentro_del_area_es_tambien_un_toque(self) -> None:
        r = stats([fac.tiro((110, 40), jug=10)])[10]
        assert r["touches_in_att_pen_area"] == 1


class TestDuelos:
    def test_regate_intentado_y_completado(self) -> None:
        r = stats([
            fac.regate((60, 40), jug=10, completado=True),
            fac.regate((60, 40), jug=10, completado=False),
        ])[10]
        assert r["take_ons"] == 2
        assert r["take_ons_won"] == 1

    def test_falta_recibida_y_perdida_de_balon(self) -> None:
        r = stats([
            fac.evento("Foul Won", jug=10, loc=(60, 40)),
            fac.evento("Dispossessed", jug=10, loc=(60, 40)),
        ])[10]
        assert r["fouls_drawn"] == 1
        assert r["dispossessed"] == 1

    @pytest.mark.parametrize("resultado", sorted(config.TACKLE_WON_OUTCOMES))
    def test_los_tackles_ganados_suman_a_duelos_ganados(self, resultado: str) -> None:
        r = stats([fac.duelo_tackle((40, 40), resultado=resultado, jug=10)])[10]
        assert r["tackles"] == 1
        assert r["tackles_won"] == 1
        assert r["duels_total"] == 1
        assert r["duels_won"] == 1

    def test_un_tackle_perdido_suma_al_total_pero_no_a_los_ganados(self) -> None:
        r = stats([fac.duelo_tackle((40, 40), resultado="Lost In Play", jug=10)])[10]
        assert r["tackles"] == 1
        assert r["tackles_won"] == 0
        assert r["duels_total"] == 1
        assert r["duels_won"] == 0

    def test_dribbled_past(self) -> None:
        r = stats([fac.evento("Dribbled Past", jug=10, loc=(40, 40))])[10]
        assert r["dribbled_past"] == 1

    @pytest.mark.parametrize(
        ("resultado", "ganado"),
        [("Won", 1), ("Success To Team", 1), ("Lost", 0), ("Success To Opposition", 0)],
    )
    def test_los_balones_divididos_entran_en_los_duelos(self, resultado, ganado) -> None:
        e = fac.evento("50/50", jug=10, loc=(50, 40), **{"50_50": {"outcome": {"name": resultado}}})
        r = stats([e])[10]
        assert r["duels_total"] == 1
        assert r["duels_won"] == ganado

    def test_duels_total_agrega_tackles_aereos_y_divididos(self) -> None:
        """Invariante declarado en `database.PLAYER_METRIC_COLUMNS`."""
        eventos = [
            fac.duelo_tackle((40, 40), resultado="Won", jug=10),
            fac.duelo_tackle((40, 40), resultado="Lost In Play", jug=10),
            fac.pase((70, 40), (80, 40), jug=10, aereo_ganado=True),
            fac.duelo_aereo_perdido((40, 40), jug=10),
            fac.evento("50/50", jug=10, loc=(50, 40), **{"50_50": {"outcome": {"name": "Won"}}}),
        ]
        r = stats(eventos)[10]
        assert r["duels_total"] == r["tackles"] + r["aerial_won"] + r["aerial_lost"] + 1
        assert r["duels_total"] == 5
        assert r["duels_won"] == 3   # tackle ganado + aereo ganado + 50/50 ganado


class TestJuegoAereo:
    def test_el_aereo_ganado_se_lee_del_flag_de_la_accion(self) -> None:
        r = stats([fac.pase((70, 40), (80, 40), jug=10, aereo_ganado=True)])[10]
        assert r["aerial_won"] == 1
        assert r["duels_total"] == 1
        assert r["duels_won"] == 1

    def test_el_aereo_perdido_tiene_evento_propio(self) -> None:
        r = stats([fac.duelo_aereo_perdido((40, 40), jug=10)])[10]
        assert r["aerial_lost"] == 1
        assert r["duels_total"] == 1
        assert r["duels_won"] == 0

    def test_los_aereos_se_reparten_por_mitad_de_campo(self) -> None:
        r = stats([
            fac.pase((70, 40), (80, 40), jug=10, aereo_ganado=True),   # campo rival
            fac.pase((30, 40), (40, 40), jug=10, aereo_ganado=True),   # campo propio
        ])[10]
        assert r["aerial_won"] == 2
        assert r["aerial_won_off"] == 1
        assert r["aerial_won_def"] == 1

    def test_el_centro_del_campo_cuenta_como_ofensivo(self) -> None:
        r = stats([fac.pase((60, 40), (70, 40), jug=10, aereo_ganado=True)])[10]
        assert r["aerial_won_off"] == 1
        assert r["aerial_won_def"] == 0

    def test_los_aereos_por_zona_suman_el_total(self) -> None:
        r = stats([
            fac.pase((70, 40), (80, 40), jug=10, aereo_ganado=True),
            fac.pase((30, 40), (40, 40), jug=10, aereo_ganado=True),
        ])[10]
        assert r["aerial_won_off"] + r["aerial_won_def"] == r["aerial_won"]


class TestDefensa:
    def test_intercepciones_bloqueos_y_despejes(self) -> None:
        r = stats([
            fac.evento("Interception", jug=10, loc=(40, 40)),
            fac.evento("Block", jug=10, loc=(40, 40)),
            fac.evento("Clearance", jug=10, loc=(20, 40)),
        ])[10]
        assert r["interceptions"] == 1
        assert r["blocks"] == 1
        assert r["clearances"] == 1

    def test_recuperacion_exitosa(self) -> None:
        r = stats([fac.recuperacion((40, 40), jug=10)])[10]
        assert r["ball_recoveries"] == 1

    def test_una_recuperacion_fallida_no_cuenta(self) -> None:
        r = stats([fac.recuperacion((40, 40), jug=10, fallida=True)])[10]
        assert r["ball_recoveries"] == 0

    def test_presiones(self) -> None:
        r = stats([fac.presion((60, 40), jug=10)])[10]
        assert r["pressures"] == 1

    def test_el_contrapress_se_cuenta_ademas_de_la_accion(self) -> None:
        """`counterpress` es un flag sobre la accion defensiva, no un evento."""
        r = stats([fac.presion((60, 40), jug=10, counterpress=True)])[10]
        assert r["pressures"] == 1
        assert r["counterpressures"] == 1

    def test_sin_flag_no_hay_contrapress(self) -> None:
        r = stats([fac.presion((60, 40), jug=10)])[10]
        assert r["counterpressures"] == 0


class TestPressureRegains:
    def test_recuperar_dentro_de_la_ventana_acredita_al_que_presiona(self) -> None:
        eventos = [
            fac.presion((60, 40), jug=10, minute=5, second=0),
            fac.recuperacion((60, 40), jug=11, minute=5, second=3),
        ]
        assert stats(eventos)[10]["pressure_regains"] == 1

    def test_la_ventana_de_5_segundos_es_inclusiva(self) -> None:
        eventos = [
            fac.presion((60, 40), jug=10, minute=5, second=0),
            fac.recuperacion((60, 40), jug=11, minute=5, second=5),
        ]
        assert stats(eventos)[10]["pressure_regains"] == 1

    def test_recuperar_despues_de_la_ventana_no_cuenta(self) -> None:
        eventos = [
            fac.presion((60, 40), jug=10, minute=5, second=0),
            fac.recuperacion((60, 40), jug=11, minute=5, second=6),
        ]
        assert stats(eventos)[10]["pressure_regains"] == 0

    def test_si_recupera_el_rival_no_hay_regain(self) -> None:
        eventos = [
            fac.presion((60, 40), jug=10, equipo=fac.EQUIPO_A, minute=5, second=0),
            fac.recuperacion((60, 40), jug=20, equipo=fac.EQUIPO_B, minute=5, second=2),
        ]
        assert stats(eventos)[10]["pressure_regains"] == 0

    def test_la_ventana_no_cruza_periodos(self) -> None:
        """El reloj sigue corriendo entre periodos: sin este corte, una presion al
        final del primer tiempo "recuperaria" en el segundo.
        """
        eventos = [
            fac.presion((60, 40), jug=10, minute=45, second=0, period=1),
            fac.recuperacion((60, 40), jug=11, minute=45, second=2, period=2),
        ]
        assert stats(eventos)[10]["pressure_regains"] == 0

    def test_solo_se_acredita_la_primera_recuperacion(self) -> None:
        eventos = [
            fac.presion((60, 40), jug=10, minute=5, second=0),
            fac.recuperacion((60, 40), jug=11, minute=5, second=1),
            fac.pase((60, 40), (70, 40), jug=11, minute=5, second=2),
        ]
        assert stats(eventos)[10]["pressure_regains"] == 1

    def test_una_presion_sin_jugador_no_rompe(self) -> None:
        e = fac.presion((60, 40), jug=None, minute=5, second=0)
        eventos = [e, fac.recuperacion((60, 40), jug=11, minute=5, second=1)]
        assert stats(eventos)[11]["pressure_regains"] == 0


class TestAislamiento:
    def test_los_eventos_de_la_tanda_de_penaltis_se_ignoran(self) -> None:
        eventos = [
            fac.tiro((100, 40), xg=0.3, resultado="Goal", jug=10, period=1),
            fac.tiro((108, 40), xg=0.78, resultado="Goal", jug=10,
                     tipo_tiro="Penalty", period=5),
        ]
        r = stats(eventos)[10]
        assert r["shots"] == 1
        assert r["goals"] == 1

    def test_los_eventos_sin_jugador_no_generan_fila(self) -> None:
        assert stats([fac.evento("Half Start", jug=None)]) == {}

    def test_cada_jugador_lleva_su_propia_cuenta(self) -> None:
        eventos = [
            fac.pase((20, 40), (60, 40), jug=10),
            fac.pase((20, 40), (25, 40), jug=11),
        ]
        r = stats(eventos)
        assert r[10]["progressive_passes"] == 1
        assert r[11]["progressive_passes"] == 0

    def test_las_metricas_no_activadas_quedan_a_cero_no_a_none(self) -> None:
        """La BD distingue 0 de NULL; un jugador que no tira tiene 0 tiros."""
        r = stats([fac.pase((20, 40), (60, 40), jug=10)])[10]
        assert r["shots"] == 0
        assert r["tackles"] == 0
        assert all(v is not None for v in r.values())

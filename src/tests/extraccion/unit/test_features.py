"""Utilidades sobre eventos: geometria, pases, xT, aereos y posesiones.

Estas funciones son el vocabulario del que dependen `player_stats` y
`team_stats`: un error aqui se propaga a todas las metricas a la vez.
"""

from __future__ import annotations

import pytest

from src.extraccion import config
from src.extraccion import features as F
from src.tests import factories as fac


class TestLocalizacion:
    def test_devuelve_la_coordenada(self) -> None:
        assert F.location({"location": [10, 20]}) == (10.0, 20.0)

    def test_sin_location_devuelve_none(self) -> None:
        """Muchos eventos (p. ej. sustituciones) no llevan coordenada."""
        assert F.location({}) is None

    def test_ignora_la_tercera_coordenada(self) -> None:
        """StatsBomb da z en algunos eventos; el plano del campo son x e y."""
        assert F.location({"location": [10, 20, 3.5]}) == (10.0, 20.0)

    def test_location_malformada_devuelve_none(self) -> None:
        assert F.location({"location": [10]}) is None
        assert F.location({"location": None}) is None


class TestDistanciaYZonas:
    def test_distancia_a_la_porteria_rival(self) -> None:
        assert F.dist_to_goal((120.0, 40.0)) == pytest.approx(0.0)
        assert F.dist_to_goal((0.0, 40.0)) == pytest.approx(120.0)
        assert F.dist_to_goal((120.0, 0.0)) == pytest.approx(40.0)

    @pytest.mark.parametrize(
        ("punto", "dentro"),
        [
            ((110.0, 40.0), True),
            ((102.0, 18.0), True),    # esquina exacta: el borde cuenta
            ((102.0, 62.0), True),
            ((101.9, 40.0), False),   # justo fuera en x
            ((110.0, 17.9), False),   # justo fuera en y
            ((110.0, 62.1), False),
        ],
    )
    def test_area_de_penalti_incluye_sus_bordes(self, punto, dentro) -> None:
        assert F.in_penalty_area(punto) is dentro

    @pytest.mark.parametrize(
        ("x", "dentro"),
        [(80.0, True), (119.0, True), (79.9, False), (0.0, False)],
    )
    def test_tercio_final(self, x: float, dentro: bool) -> None:
        assert F.in_final_third((x, 40.0)) is dentro


class TestProgresion:
    def test_el_umbral_del_25_por_ciento_es_inclusivo(self) -> None:
        """Desde (0,40) quedan 120 a puerta: reducir exactamente 30 (25%) progresa."""
        assert F.is_progressive((0.0, 40.0), (30.0, 40.0)) is True

    def test_justo_por_debajo_del_umbral_no_progresa(self) -> None:
        assert F.is_progressive((0.0, 40.0), (29.0, 40.0)) is False

    def test_un_pase_hacia_atras_no_progresa(self) -> None:
        assert F.is_progressive((60.0, 40.0), (30.0, 40.0)) is False

    def test_desde_la_propia_porteria_del_rival_no_progresa(self) -> None:
        """d_start = 0: la reduccion relativa no esta definida; se declara no
        progresivo en vez de dividir por cero.
        """
        assert F.is_progressive((120.0, 40.0), (120.0, 40.0)) is False

    def test_el_criterio_es_relativo_no_absoluto(self) -> None:
        """20 yardas progresan cerca del area (queda poco) pero no desde el
        propio campo: esa es justo la diferencia entre StatsBomb y FBref/Opta.
        """
        assert F.is_progressive((100.0, 40.0), (115.0, 40.0)) is True   # 20 -> 5
        assert F.is_progressive((0.0, 40.0), (20.0, 40.0)) is False     # 120 -> 100


class TestPases:
    def test_pase_sin_outcome_esta_completado(self) -> None:
        assert F.pass_completed(fac.pase((10, 40), (30, 40))) is True

    def test_pase_con_outcome_no_esta_completado(self) -> None:
        assert F.pass_completed(fac.pase((10, 40), (30, 40), completado=False)) is False

    def test_un_evento_que_no_es_pase_nunca_esta_completado(self) -> None:
        assert F.pass_completed(fac.tiro((110, 40))) is False

    def test_is_pass(self) -> None:
        assert F.is_pass(fac.pase((10, 40), (30, 40))) is True
        assert F.is_pass(fac.conduccion((10, 40), (30, 40))) is False

    @pytest.mark.parametrize("tipo", sorted(config.SET_PIECE_PASS_TYPES))
    def test_los_balones_parados_se_reconocen(self, tipo: str) -> None:
        assert F.is_set_piece_pass(fac.pase((10, 40), (30, 40), tipo_pase=tipo)) is True

    def test_un_pase_de_jugada_no_es_balon_parado(self) -> None:
        assert F.is_set_piece_pass(fac.pase((10, 40), (30, 40))) is False
        # Recycled/Interception son tipos de pase que NO son balon parado.
        assert F.is_set_piece_pass(
            fac.pase((10, 40), (30, 40), tipo_pase="Recovery")
        ) is False

    def test_destino_de_pase_y_de_conduccion(self) -> None:
        assert F.pass_end(fac.pase((10, 40), (30, 45))) == (30.0, 45.0)
        assert F.carry_end(fac.conduccion((10, 40), (30, 45))) == (30.0, 45.0)

    def test_sin_destino_devuelve_none(self) -> None:
        assert F.pass_end(fac.conduccion((10, 40), (30, 45))) is None
        assert F.carry_end(fac.pase((10, 40), (30, 45))) is None


class TestToques:
    def test_una_recepcion_completada_es_un_toque(self) -> None:
        assert F.is_touch(fac.recepcion((50, 40))) is True

    def test_una_recepcion_fallida_no_es_un_toque(self) -> None:
        """Si el balon no llega, el jugador no lo ha tocado."""
        assert F.is_touch(fac.recepcion((50, 40), completada=False)) is False

    def test_una_conduccion_es_un_toque(self) -> None:
        assert F.is_touch(fac.conduccion((50, 40), (60, 40))) is True

    def test_un_pase_no_cuenta_como_toque(self) -> None:
        """El toque ya lo contabiliza el `Ball Receipt*` previo: contar tambien el
        pase duplicaria cada posesion en `possession_pct` y `field_tilt`.
        """
        assert F.is_touch(fac.pase((50, 40), (60, 40))) is False

    def test_una_presion_no_es_un_toque(self) -> None:
        assert F.is_touch(fac.presion((50, 40))) is False


class TestExpectedThreat:
    def test_el_area_pequena_vale_el_maximo_de_la_rejilla(self) -> None:
        maximo = max(v for fila in config.XT_GRID for v in fila)
        assert F.xt_at((119.0, 40.0)) == pytest.approx(maximo)

    def test_la_esquina_propia_vale_el_minimo(self) -> None:
        minimo = min(v for fila in config.XT_GRID for v in fila)
        assert F.xt_at((0.0, 0.0)) == pytest.approx(minimo)

    def test_la_coordenada_maxima_no_se_sale_de_la_rejilla(self) -> None:
        """x=120 daria el bin 12 en una rejilla de 12 columnas (0..11): se recorta."""
        assert F.xt_at((120.0, 80.0)) == config.XT_GRID[config.XT_ROWS - 1][config.XT_COLS - 1]

    def test_coordenadas_fuera_del_campo_se_recortan(self) -> None:
        assert F.xt_at((-10.0, -10.0)) == config.XT_GRID[0][0]
        assert F.xt_at((999.0, 999.0)) == config.XT_GRID[config.XT_ROWS - 1][config.XT_COLS - 1]

    def test_avanzar_hacia_la_porteria_anade_amenaza(self) -> None:
        assert F.xt_delta((20.0, 40.0), (110.0, 40.0)) > 0

    def test_retroceder_resta_amenaza(self) -> None:
        assert F.xt_delta((110.0, 40.0), (20.0, 40.0)) < 0

    def test_moverse_dentro_del_mismo_bin_no_cambia_la_amenaza(self) -> None:
        assert F.xt_delta((1.0, 1.0), (2.0, 2.0)) == pytest.approx(0.0)


class TestAereos:
    @pytest.mark.parametrize("clave", config.AERIAL_WON_KEYS)
    def test_el_flag_aerial_won_se_detecta_en_cualquier_sub_objeto(self, clave: str) -> None:
        """StatsBomb no emite evento de "aereo ganado": marca el flag en la accion
        con la que el ganador resuelve el salto, que puede ser de varios tipos.
        """
        e = fac.evento("Pass", **{clave: {"aerial_won": True}})
        assert F.is_aerial_won(e) is True

    def test_sin_el_flag_no_hay_aereo_ganado(self) -> None:
        assert F.is_aerial_won(fac.pase((50, 40), (60, 40))) is False

    def test_el_perdedor_del_salto_si_tiene_evento_propio(self) -> None:
        assert F.is_aerial_lost(fac.duelo_aereo_perdido((50, 40))) is True

    def test_un_tackle_no_es_un_aereo_perdido(self) -> None:
        assert F.is_aerial_lost(fac.duelo_tackle((50, 40))) is False


class TestTiempo:
    def test_los_segundos_son_de_reloj_continuo(self) -> None:
        """`minute` es acumulado desde el inicio del partido, no por periodo."""
        assert F.event_seconds({"minute": 90, "second": 5}) == 5405

    def test_evento_sin_reloj_es_el_minuto_cero(self) -> None:
        assert F.event_seconds({}) == 0

    def test_la_tanda_de_penaltis_se_descarta(self) -> None:
        eventos = [
            fac.pase((10, 40), (30, 40), period=1),
            fac.tiro((108, 40), period=5),   # tanda: no es juego
            fac.pase((10, 40), (30, 40), period=4),
        ]
        jugables = F.playable_events(eventos)
        assert len(jugables) == 2
        assert all(e["period"] != 5 for e in jugables)

    def test_evento_sin_periodo_se_considera_jugable(self) -> None:
        assert F.playable_events([{"type": {"name": "Pass"}}]) != []


class TestPosesiones:
    def _eventos_dos_posesiones(self) -> list[dict]:
        return [
            fac.pase((10, 40), (30, 40), jug=10, possession=1,
                     possession_team=fac.EQUIPO_A, equipo=fac.EQUIPO_A),
            fac.pase((30, 40), (50, 40), jug=11, possession=1,
                     possession_team=fac.EQUIPO_A, equipo=fac.EQUIPO_A),
            # Presion del rival DENTRO de la posesion de A: es del equipo B.
            fac.presion((50, 40), jug=20, possession=1,
                        possession_team=fac.EQUIPO_A, equipo=fac.EQUIPO_B),
            fac.pase((10, 40), (30, 40), jug=20, possession=2,
                     possession_team=fac.EQUIPO_B, equipo=fac.EQUIPO_B),
        ]

    def test_agrupa_por_id_de_posesion(self) -> None:
        posesiones = F.group_possessions(self._eventos_dos_posesiones())
        assert [p.id for p in posesiones] == [1, 2]

    def test_team_events_solo_trae_los_del_dueno_de_la_posesion(self) -> None:
        """La presion del rival ocurre durante la posesion de A pero no es suya:
        contarla inflaria los pases por secuencia y la amplitud.
        """
        p1 = F.group_possessions(self._eventos_dos_posesiones())[0]
        assert p1.team_name == "Equipo A"
        assert p1.team_id == fac.EQUIPO_A["id"]
        assert len(p1.events) == 3
        assert len(p1.team_events) == 2

    def test_cuenta_solo_los_pases_completados_del_dueno(self) -> None:
        eventos = [
            fac.pase((10, 40), (30, 40), possession=1),
            fac.pase((30, 40), (50, 40), completado=False, possession=1),
        ]
        assert F.group_possessions(eventos)[0].n_completed_passes() == 1

    def test_la_posesion_arranca_donde_el_primer_evento_con_coordenada(self) -> None:
        eventos = [
            fac.evento("Half Start", possession=1),   # sin location
            fac.pase((25.0, 30.0), (50, 40), possession=1),
        ]
        assert F.group_possessions(eventos)[0].start_location() == (25.0, 30.0)

    def test_posesion_sin_coordenadas_no_tiene_inicio(self) -> None:
        assert F.group_possessions([fac.evento("Half Start", possession=1)])[0].start_location() is None

    def test_se_respeta_el_orden_de_aparicion(self) -> None:
        eventos = [
            fac.pase((10, 40), (30, 40), possession=7),
            fac.pase((10, 40), (30, 40), possession=3),
            fac.pase((10, 40), (30, 40), possession=7),
        ]
        posesiones = F.group_possessions(eventos)
        assert [p.id for p in posesiones] == [7, 3]
        assert len(posesiones[0].events) == 2

    def test_los_eventos_sin_posesion_se_descartan(self) -> None:
        eventos = [fac.evento("Half Start"), fac.pase((10, 40), (30, 40), possession=1)]
        del eventos[0]["possession"]
        assert len(F.group_possessions(eventos)) == 1

    def test_la_tanda_de_penaltis_no_genera_posesiones(self) -> None:
        eventos = [fac.tiro((108, 40), period=5, possession=99)]
        assert F.group_possessions(eventos) == []

    @pytest.mark.parametrize(
        ("patron", "es_abierto"),
        [
            ("Regular Play", True),
            ("From Counter", True),
            ("From Kick Off", True),
            ("From Corner", False),
            ("From Free Kick", False),
            ("From Throw In", False),
        ],
    )
    def test_juego_abierto_excluye_el_balon_parado(self, patron: str, es_abierto: bool) -> None:
        eventos = [fac.pase((10, 40), (30, 40), possession=1, play_pattern=patron)]
        poss = F.group_possessions(eventos)[0]
        assert poss.play_pattern == patron
        assert F.is_open_play_possession(poss) is es_abierto

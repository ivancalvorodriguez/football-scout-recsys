"""Constantes de dominio: geometria, catalogo de posiciones y rejilla xT.

Son constantes, pero no son arbitrarias: cada una codifica un criterio de
StatsBomb documentado en `docs/metricas_finales.md`. Un dedazo aqui (un 0.25 que
pasa a 0.025, una fila de la rejilla xT descolocada) no rompe nada visiblemente:
solo desplaza en silencio todas las metricas del proyecto.
"""

from __future__ import annotations

import pytest

from src.extraccion import config


class TestGeometria:
    def test_campo_es_120x80(self) -> None:
        """Sistema StatsBomb. Muchas metricas por zona dependen de esta escala."""
        assert (config.FIELD_LENGTH, config.FIELD_WIDTH) == (120.0, 80.0)

    def test_porterias_centradas_en_los_extremos(self) -> None:
        assert config.GOAL == (config.FIELD_LENGTH, config.FIELD_WIDTH / 2)
        assert config.OWN_GOAL == (0.0, config.FIELD_WIDTH / 2)

    def test_area_de_penalti_dentro_del_campo_y_simetrica(self) -> None:
        assert config.PEN_AREA_X < config.FIELD_LENGTH
        assert 0.0 < config.PEN_AREA_Y_MIN < config.PEN_AREA_Y_MAX < config.FIELD_WIDTH
        centro = (config.PEN_AREA_Y_MIN + config.PEN_AREA_Y_MAX) / 2
        assert centro == pytest.approx(config.FIELD_WIDTH / 2)

    def test_tercio_final_es_el_ultimo_tercio(self) -> None:
        assert config.FINAL_THIRD_X == pytest.approx(config.FIELD_LENGTH * 2 / 3)

    def test_deep_completion_son_20_metros_en_unidades_statsbomb(self) -> None:
        """Criterio Wyscout (20 m), convertido a yardas StatsBomb."""
        assert config.DEEP_COMPLETION_RADIUS * config.YARD_TO_M == pytest.approx(20.0)

    def test_high_turnover_son_40_metros_desde_la_porteria_rival(self) -> None:
        """Criterio Opta (40 m), expresado como umbral de x."""
        distancia = config.FIELD_LENGTH - config.HIGH_TURNOVER_X
        assert distancia * config.YARD_TO_M == pytest.approx(40.0)

    def test_umbral_progresivo_es_el_25_por_ciento_de_statsbomb(self) -> None:
        """StatsBomb usa 25% de la distancia restante; FBref/Opta usan ~10 yardas.
        Los datos son StatsBomb: el criterio debe ser el suyo (ver CLAUDE.md).
        """
        assert config.PROGRESSIVE_FRACTION == 0.25


class TestCatalogos:
    def test_hay_25_posiciones_con_ids_del_1_al_25(self) -> None:
        assert len(config.POSITIONS_25) == 25
        assert sorted(config.POSITIONS_25) == list(range(1, 26))

    def test_los_nombres_de_posicion_no_se_repiten(self) -> None:
        nombres = list(config.POSITIONS_25.values())
        assert len(set(nombres)) == len(nombres)

    @pytest.mark.parametrize(
        ("nombre", "esperado"),
        [
            ("Right Back", "pos_right_back"),
            ("Goalkeeper", "pos_goalkeeper"),
            ("Center Defensive Midfield", "pos_center_defensive_midfield"),
        ],
    )
    def test_position_slug(self, nombre: str, esperado: str) -> None:
        assert config.position_slug(nombre) == esperado

    def test_position_slug_es_inyectivo_sobre_el_catalogo(self) -> None:
        """Dos posiciones distintas no pueden compartir columna one-hot."""
        slugs = [config.position_slug(n) for n in config.POSITIONS_25.values()]
        assert len(set(slugs)) == len(slugs)

    def test_sca_se_acredita_a_las_2_ultimas_acciones(self) -> None:
        """Criterio FBref. Compartido por `player_stats` y `team_stats`."""
        assert config.SCA_MAX_ACTIONS == 2
        assert "Pass" in config.SCA_ACTION_TYPES

    def test_el_gol_cuenta_como_tiro_a_puerta(self) -> None:
        assert "Goal" in config.SHOT_ON_TARGET_OUTCOMES


class TestRejillaXT:
    def test_dimensiones_coinciden_con_las_constantes(self) -> None:
        assert len(config.XT_GRID) == config.XT_ROWS
        assert all(len(fila) == config.XT_COLS for fila in config.XT_GRID)

    def test_la_rejilla_es_simetrica_respecto_al_eje_del_campo(self) -> None:
        """La superficie de Karun Singh es simetrica: atacar por la banda derecha
        vale lo mismo que por la izquierda.
        """
        for y in range(config.XT_ROWS // 2):
            assert config.XT_GRID[y] == config.XT_GRID[config.XT_ROWS - 1 - y]

    def test_el_maximo_de_cada_fila_esta_pegado_a_la_porteria(self) -> None:
        """La amenaza no es monotona en x (hay dientes en la rejilla publica),
        pero el maximo de cada fila cae siempre en la ultima columna.
        """
        for fila in config.XT_GRID:
            assert fila.index(max(fila)) == config.XT_COLS - 1

    def test_todos_los_valores_son_probabilidades(self) -> None:
        assert all(0.0 <= v <= 1.0 for fila in config.XT_GRID for v in fila)

    def test_el_centro_del_area_es_la_zona_de_mas_amenaza(self) -> None:
        maximo = max(v for fila in config.XT_GRID for v in fila)
        filas_centrales = (config.XT_ROWS // 2 - 1, config.XT_ROWS // 2)
        assert maximo == config.XT_GRID[filas_centrales[0]][config.XT_COLS - 1]
        assert maximo == config.XT_GRID[filas_centrales[1]][config.XT_COLS - 1]

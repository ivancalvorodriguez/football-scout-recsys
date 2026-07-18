"""Ensamblado de filas y resolucion de argumentos del extractor."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.extraccion import database as db
from src.extraccion import extract
from src.tests import factories as fac


class TestPadjFactor:
    def test_con_posesion_del_50_por_ciento_no_ajusta(self) -> None:
        """El factor esta calibrado con la referencia en 50% de posesion rival."""
        assert extract._padj_factor(0.5) == pytest.approx(1.0)

    def test_si_tienes_mucho_el_balon_tus_acciones_defensivas_valen_mas(self) -> None:
        """Con 75% de posesion propia el rival solo ataca el 25% del tiempo: cada
        accion defensiva es mas escasa y se infla (x2).
        """
        assert extract._padj_factor(0.75) == pytest.approx(2.0)

    def test_si_defiendes_todo_el_partido_tus_acciones_valen_menos(self) -> None:
        assert extract._padj_factor(0.25) == pytest.approx(2 / 3)

    def test_sin_posesion_conocida_no_se_ajusta(self) -> None:
        assert extract._padj_factor(None) == 1.0

    def test_posesion_total_no_divide_por_cero(self) -> None:
        """Caso degenerado (el rival no toca el balon): se deja el factor neutro."""
        assert extract._padj_factor(1.0) == 1.0

    def test_el_factor_es_decreciente_en_la_posesion_del_rival(self) -> None:
        factores = [extract._padj_factor(p) for p in (0.2, 0.4, 0.6, 0.8)]
        assert factores == sorted(factores)


class TestBuildPlayerRow:
    REC = {
        "player_id": 10, "team_id": 1, "minutes_played": 87.5,
        "primary_position_id": 23, "primary_position_name": "Center Forward",
        "positions": [],
    }

    def test_lleva_el_contexto_del_jugador_partido(self) -> None:
        fila = extract._build_player_row(7, self.REC, {}, 1.0)
        assert fila["player_id"] == 10
        assert fila["match_id"] == 7
        assert fila["team_id"] == 1
        assert fila["minutes_played"] == 87.5
        assert fila["primary_position_name"] == "Center Forward"

    def test_tiene_exactamente_las_columnas_del_esquema(self) -> None:
        fila = extract._build_player_row(7, self.REC, {}, 1.0)
        assert set(fila) == set(db.PLAYER_COLUMNS)

    def test_copia_las_metricas_calculadas(self) -> None:
        fila = extract._build_player_row(7, self.REC, {"goals": 2, "xg": 1.5}, 1.0)
        assert fila["goals"] == 2
        assert fila["xg"] == 1.5

    def test_padj_ajusta_intercepciones_mas_recuperaciones(self) -> None:
        metricas = {"interceptions": 3, "ball_recoveries": 5}
        fila = extract._build_player_row(7, self.REC, metricas, 2.0)
        assert fila["padj_def_actions"] == pytest.approx(16.0)

    def test_padj_se_calcula_aunque_falten_las_metricas(self) -> None:
        """Un jugador que no toco el balon tiene 0 acciones, no None."""
        fila = extract._build_player_row(7, self.REC, {}, 2.0)
        assert fila["padj_def_actions"] == 0.0

    def test_padj_se_redondea_a_tres_decimales(self) -> None:
        metricas = {"interceptions": 1, "ball_recoveries": 0}
        fila = extract._build_player_row(7, self.REC, metricas, 1.0 / 3.0)
        assert fila["padj_def_actions"] == 0.333


class TestBuildTeamRow:
    CONTEXTO = {"opponent_id": 2, "is_home": 1, "goals_for": 3, "goals_against": 1}

    def test_lleva_el_contexto_del_equipo_partido(self) -> None:
        fila = extract._build_team_row(7, 1, {}, self.CONTEXTO)
        assert fila["team_id"] == 1
        assert fila["match_id"] == 7
        assert fila["opponent_id"] == 2
        assert fila["is_home"] == 1
        assert fila["goals_for"] == 3
        assert fila["goals_against"] == 1

    def test_tiene_exactamente_las_columnas_del_esquema(self) -> None:
        fila = extract._build_team_row(7, 1, {}, self.CONTEXTO)
        assert set(fila) == set(db.TEAM_COLUMNS)

    def test_las_secuencias_de_juego_abierto_salen_del_contador_interno(self) -> None:
        """`seq_count` es un acumulador de `team_stats`; se persiste con su nombre
        de esquema, `open_play_sequences`.
        """
        fila = extract._build_team_row(7, 1, {"seq_count": 87.0}, self.CONTEXTO)
        assert fila["open_play_sequences"] == 87
        assert isinstance(fila["open_play_sequences"], int)

    def test_sin_secuencias_se_guarda_cero(self) -> None:
        assert extract._build_team_row(7, 1, {}, self.CONTEXTO)["open_play_sequences"] == 0


class TestResolveFromArgs:
    @pytest.fixture
    def raiz(self, tmp_path: Path) -> Path:
        return fac.escribir_open_data(
            tmp_path / "data",
            competiciones=[
                fac.fila_competicion(1, 100, competition_name="Premier League",
                                     season_name="2015/2016"),
                fac.fila_competicion(2, 200, competition_name="Liga Sin Datos",
                                     season_name="2019/2020"),
            ],
            partidos={(1, 100): [
                fac.partido(i, jornada=i, fecha=f"2020-01-{i:02d}") for i in range(1, 5)
            ]},
        )

    def _args(self, **cambios) -> argparse.Namespace:
        base = {"competition": "Premier League", "season": "2015/2016",
                "pct": 100.0, "db": "outputs/db/x.db"}
        base.update(cambios)
        return argparse.Namespace(**base)

    def test_resuelve_competicion_temporada_y_partidos(self, raiz: Path) -> None:
        seleccion = extract._resolve_from_args(self._args(), raiz)
        assert seleccion is not None
        assert seleccion["competition"]["competition_id"] == 1
        assert seleccion["season"]["season_id"] == 100
        assert len(seleccion["matches"]) == 4
        assert seleccion["db_path"] == Path("outputs/db/x.db")

    def test_aplica_el_porcentaje_de_jornadas(self, raiz: Path) -> None:
        seleccion = extract._resolve_from_args(self._args(pct=50.0), raiz)
        assert len(seleccion["matches"]) == 2

    def test_competicion_inexistente(self, raiz: Path, capsys: pytest.CaptureFixture) -> None:
        assert extract._resolve_from_args(self._args(competition="Inventada"), raiz) is None
        assert "No se encontró la competición" in capsys.readouterr().out

    def test_temporada_inexistente_lista_las_disponibles(
        self, raiz: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert extract._resolve_from_args(self._args(season="1999/2000"), raiz) is None
        salida = capsys.readouterr().out
        assert "No se encontró la temporada" in salida
        assert "2015/2016" in salida   # ayuda al usuario con las validas

    def test_una_temporada_sin_partidos_en_disco(
        self, raiz: Path, capsys: pytest.CaptureFixture
    ) -> None:
        args = self._args(competition="Liga Sin Datos", season="2019/2020")
        assert extract._resolve_from_args(args, raiz) is None
        assert "No hay partidos en disco" in capsys.readouterr().out

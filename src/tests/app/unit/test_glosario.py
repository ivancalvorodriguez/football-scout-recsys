"""Rótulos y unidades de las métricas."""

from __future__ import annotations

import pytest

from src.app import glosario
from src.similitud import config as config_similitud


class TestEtiqueta:
    def test_traduce_las_metricas_del_catalogo(self) -> None:
        assert glosario.etiqueta("np_goals_minus_npxg") == "Goles menos xG (sin penaltis)"

    def test_las_posiciones_llevan_prefijo(self) -> None:
        assert glosario.etiqueta("pos_center_forward") == "Posición: Delantero centro"

    def test_una_metrica_desconocida_no_rompe(self) -> None:
        assert glosario.etiqueta("metrica_inventada") == "Metrica inventada"

    def test_estan_catalogadas_todas_las_del_pipeline(self) -> None:
        """Una métrica sin rótulo saldría en la interfaz con su nombre en crudo."""
        c = config_similitud
        derivadas = (
            {n for n, _, _ in c.PLAYER_RATIO_FEATURES}
            | {n for n, _, _ in c.TEAM_RATIO_FEATURES}
            | {n for n, _, _ in c.PLAYER_DIFF_FEATURES}
            | {n for n, _, _ in c.TEAM_DIFF_FEATURES}
        )
        todas = (
            set(c.PLAYER_COUNT_FEATURES) | set(c.TEAM_COUNT_FEATURES)
            | set(c.TEAM_RATE_FEATURES) | derivadas
        )
        assert todas <= set(glosario.ETIQUETAS)


class TestFormatear:
    def test_un_conteo_lleva_dos_decimales(self) -> None:
        assert glosario.formatear("progressive_passes", 3.456) == "3.46"

    def test_un_conteo_grande_se_acorta(self) -> None:
        """676,53 pases por partido no necesita dos decimales."""
        assert glosario.formatear("passes", 676.53) == "676.5"

    def test_un_porcentaje_se_escribe_como_tal(self) -> None:
        """La métrica es una fracción; enseñar «0.74» sería ilegible."""
        assert glosario.formatear("pass_completion_pct", 0.7413) == "74 %"

    def test_la_posesion_tambien_es_un_porcentaje(self) -> None:
        assert glosario.formatear("possession_pct", 0.62) == "62 %"

    def test_la_posicion_se_escribe_en_porcentaje_de_minutos(self) -> None:
        assert glosario.formatear("pos_left_wing", 0.921) == "92 %"

    def test_los_valores_esperados_van_con_dos_decimales(self) -> None:
        assert glosario.formatear("npxg", 0.4271) == "0.43"

    def test_una_diferencia_lleva_el_signo_explicito(self) -> None:
        """En G−xG el signo ES la información."""
        assert glosario.formatear("np_goals_minus_npxg", 0.31) == "+0.31"
        assert glosario.formatear("goals_minus_xg", -0.31) == "-0.31"

    def test_las_distancias_llevan_metros(self) -> None:
        assert glosario.formatear("absolute_width", 55.83) == "55.8 m"

    def test_la_velocidad_lleva_su_unidad(self) -> None:
        assert glosario.formatear("direct_speed", 3.0212) == "3.02 m/s"

    def test_sin_dato_no_se_inventa_un_cero(self) -> None:
        """Un 0 se leería como un rendimiento nulo real."""
        assert glosario.formatear("passes", None) == "—"
        assert glosario.formatear("take_ons_pct", float("nan")) == "—"

    def test_ppda_es_un_ratio_de_dos_decimales(self) -> None:
        assert glosario.formatear("ppda", 9.4271) == "9.43"

    @pytest.mark.parametrize("entidad,esperado", [
        ("jugador", "por 90 minutos"), ("equipo", "por partido"),
    ])
    def test_la_unidad_del_conteo_depende_de_la_entidad(
        self, entidad: str, esperado: str
    ) -> None:
        assert glosario.UNIDAD_CONTEO[entidad] == esperado


class TestUnidad:
    def test_un_conteo_de_jugador_va_por_90(self) -> None:
        assert glosario.unidad("progressive_passes", "jugador") == "por 90 min"

    def test_una_metrica_nueva_hereda_el_caso_por_defecto(self) -> None:
        """Los conteos son el default; solo se enumeran las excepciones."""
        assert glosario.unidad("metrica_inventada", "equipo") == "por partido"

    def test_una_posicion_se_expresa_en_minutos_propios(self) -> None:
        """No aparece en las listas de métricas, pero el rótulo es coherente."""
        assert glosario.unidad("pos_left_wing", "jugador") == "% de sus minutos"

    def test_un_porcentaje_no_repite_unidad(self) -> None:
        assert glosario.unidad("sot_pct", "jugador") == ""

    def test_una_entidad_desconocida_no_inventa_unidad(self) -> None:
        assert glosario.unidad("passes", "arbitro") == ""

"""Capa de features: per-90, ratios, diferencias y estandarizacion.

Es la capa que decide QUE se compara. Los dos riesgos que se vigilan aqui:
colapsar observaciones (romperia el principio central del modelo) y desalinear
filas al estandarizar por liga.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.similitud import config, features
from src.similitud.data import LEAGUE_KEY


def _df_jugador(filas: list[dict]) -> pd.DataFrame:
    """DataFrame con todas las columnas que exige `_derivar_jugador`.

    Todo a 0 salvo lo que indique cada fila: asi cada prueba mueve una sola
    variable y el resto no interfiere.
    """
    base: dict = {c: 0.0 for c in config.PLAYER_COUNT_FEATURES}
    base.update({
        "minutes_played": 90.0, "entity_id": 1, "entity_name": "Jugador",
        LEAGUE_KEY: "L1",
    })
    return pd.DataFrame([{**base, **f} for f in filas])


def _df_equipo(filas: list[dict]) -> pd.DataFrame:
    base: dict = {c: 0.0 for c in config.TEAM_COUNT_FEATURES}
    base.update({c: 0.0 for c in config.TEAM_RATE_FEATURES})
    base.update({"entity_id": 1, "entity_name": "Equipo", LEAGUE_KEY: "L1"})
    return pd.DataFrame([{**base, **f} for f in filas])


class TestRatioSeguro:
    def test_division_normal(self) -> None:
        num, den = pd.Series([8.0, 5.0]), pd.Series([10.0, 10.0])
        assert features._safe_ratio(num, den).tolist() == [0.8, 0.5]

    def test_denominador_cero_da_nan_no_infinito(self) -> None:
        """0 intentos no es 0% de acierto: es "no definido". Un inf o un 0
        silencioso mentiria sobre el jugador.
        """
        resultado = features._safe_ratio(pd.Series([3.0]), pd.Series([0.0]))
        assert np.isnan(resultado[0])


class TestPer90:
    def test_escala_a_90_minutos(self) -> None:
        df = _df_jugador([{"minutes_played": 45.0, "goals": 1.0}])
        assert features._per90(df, ["goals"], df["minutes_played"])["goals"][0] == 2.0

    def test_noventa_minutos_no_cambia_nada(self) -> None:
        df = _df_jugador([{"minutes_played": 90.0, "goals": 1.0}])
        assert features._per90(df, ["goals"], df["minutes_played"])["goals"][0] == 1.0

    def test_cero_minutos_da_nan_no_infinito(self) -> None:
        df = _df_jugador([{"minutes_played": 0.0, "goals": 1.0}])
        assert np.isnan(features._per90(df, ["goals"], df["minutes_played"])["goals"][0])


class TestDerivarJugador:
    def test_los_conteos_pasan_a_per90(self) -> None:
        df = _df_jugador([{"minutes_played": 45.0, "shots": 2.0}])
        assert features._derivar_jugador(df)["shots"][0] == pytest.approx(4.0)

    def test_pass_completion_pct(self) -> None:
        df = _df_jugador([{"passes": 50.0, "passes_completed": 40.0}])
        assert features._derivar_jugador(df)["pass_completion_pct"][0] == pytest.approx(0.8)

    def test_el_ratio_no_depende_de_los_minutos(self) -> None:
        """Es eficiencia, no volumen: el per-90 se cancela en la division."""
        df = _df_jugador([
            {"minutes_played": 45.0, "passes": 50.0, "passes_completed": 40.0},
            {"minutes_played": 90.0, "passes": 50.0, "passes_completed": 40.0},
        ])
        feats = features._derivar_jugador(df)
        assert feats["pass_completion_pct"].tolist() == [pytest.approx(0.8)] * 2
        # El volumen si depende de los minutos: son senales distintas.
        assert feats["passes"][0] == 2 * feats["passes"][1]

    def test_aerial_won_pct_usa_el_denominador_compuesto(self) -> None:
        df = _df_jugador([{"aerial_won": 3.0, "aerial_lost": 1.0}])
        assert features._derivar_jugador(df)["aerial_won_pct"][0] == pytest.approx(0.75)

    def test_tackle_pct_usa_ganados_sobre_ganados_mas_regateado(self) -> None:
        df = _df_jugador([{"tackles_won": 3.0, "dribbled_past": 1.0}])
        assert features._derivar_jugador(df)["tackle_pct"][0] == pytest.approx(0.75)

    def test_duels_won_pct(self) -> None:
        df = _df_jugador([{"duels_won": 6.0, "duels_total": 10.0}])
        assert features._derivar_jugador(df)["duels_won_pct"][0] == pytest.approx(0.6)

    def test_sot_pct_excluye_los_penaltis(self) -> None:
        """El denominador es `np_shots`: ambos deben excluir el penalti."""
        df = _df_jugador([{"shots_on_target": 2.0, "np_shots": 5.0, "shots": 6.0}])
        assert features._derivar_jugador(df)["sot_pct"][0] == pytest.approx(0.4)

    def test_npxg_per_shot(self) -> None:
        df = _df_jugador([{"npxg": 1.0, "np_shots": 5.0}])
        assert features._derivar_jugador(df)["npxg_per_shot"][0] == pytest.approx(0.2)

    def test_un_jugador_que_no_lo_intenta_no_tiene_ratio(self) -> None:
        df = _df_jugador([{"take_ons": 0.0, "take_ons_won": 0.0}])
        assert np.isnan(features._derivar_jugador(df)["take_ons_pct"][0])

    def test_np_goals_minus_npxg_se_calcula_sobre_per90(self) -> None:
        """Ambos terminos deben estar en la misma escala o la diferencia miente."""
        df = _df_jugador([{"minutes_played": 45.0, "np_goals": 1.0, "npxg": 0.25}])
        feats = features._derivar_jugador(df)
        assert feats["np_goals_minus_npxg"][0] == pytest.approx(2.0 - 0.5)

    def test_estan_todas_las_features_esperadas(self) -> None:
        feats = features._derivar_jugador(_df_jugador([{}]))
        esperadas = (
            set(config.PLAYER_COUNT_FEATURES)
            | {n for n, _, _ in config.PLAYER_RATIO_FEATURES}
            | {n for n, _, _ in config.PLAYER_DIFF_FEATURES}
        )
        assert set(feats.columns) == esperadas

    def test_los_denominadores_auxiliares_no_se_cuelan_como_feature(self) -> None:
        feats = features._derivar_jugador(_df_jugador([{}]))
        assert not [c for c in feats.columns if c.startswith("__")]


class TestDerivarEquipo:
    def test_los_conteos_se_usan_por_partido_sin_per90(self) -> None:
        """El equipo siempre juega el partido completo."""
        df = _df_equipo([{"shots": 12.0}])
        assert features._derivar_equipo(df)["shots"][0] == 12.0

    def test_las_tasas_se_usan_tal_cual(self) -> None:
        df = _df_equipo([{"ppda": 9.5, "possession_pct": 0.62}])
        feats = features._derivar_equipo(df)
        assert feats["ppda"][0] == 9.5
        assert feats["possession_pct"][0] == 0.62

    def test_xg_per_shot(self) -> None:
        df = _df_equipo([{"xg": 1.5, "shots": 10.0}])
        assert features._derivar_equipo(df)["xg_per_shot"][0] == pytest.approx(0.15)

    def test_goals_minus_xg(self) -> None:
        df = _df_equipo([{"goals": 3.0, "xg": 1.5}])
        assert features._derivar_equipo(df)["goals_minus_xg"][0] == pytest.approx(1.5)

    def test_estan_todas_las_features_esperadas(self) -> None:
        feats = features._derivar_equipo(_df_equipo([{}]))
        esperadas = (
            set(config.TEAM_COUNT_FEATURES)
            | set(config.TEAM_RATE_FEATURES)
            | {n for n, _, _ in config.TEAM_RATIO_FEATURES}
            | {n for n, _, _ in config.TEAM_DIFF_FEATURES}
        )
        assert set(feats.columns) == esperadas


class TestEstandarizacion:
    def test_por_liga_deja_media_cero_dentro_de_cada_liga(self) -> None:
        df = _df_jugador([
            {"entity_id": 1, LEAGUE_KEY: "L1", "goals": 0.0},
            {"entity_id": 2, LEAGUE_KEY: "L1", "goals": 4.0},
            {"entity_id": 3, LEAGUE_KEY: "L2", "goals": 100.0},
            {"entity_id": 4, LEAGUE_KEY: "L2", "goals": 104.0},
        ])
        mf = features.construir(df, "jugador", normalizacion="por_liga")
        i = mf.feat_names.index("goals")
        # Las dos ligas tienen niveles muy distintos pero la misma forma:
        # estandarizadas por liga quedan identicas.
        assert mf.X[:, i].tolist() == [-1.0, 1.0, -1.0, 1.0]

    def test_global_conserva_la_diferencia_de_nivel_entre_ligas(self) -> None:
        """Es justo lo que habilita la comparacion inter-liga."""
        df = _df_jugador([
            {"entity_id": 1, LEAGUE_KEY: "L1", "goals": 0.0},
            {"entity_id": 2, LEAGUE_KEY: "L1", "goals": 4.0},
            {"entity_id": 3, LEAGUE_KEY: "L2", "goals": 100.0},
            {"entity_id": 4, LEAGUE_KEY: "L2", "goals": 104.0},
        ])
        mf = features.construir(df, "jugador", normalizacion="global")
        i = mf.feat_names.index("goals")
        z = mf.X[:, i]
        assert z[0] < 0 and z[1] < 0        # la liga floja queda por debajo
        assert z[2] > 0 and z[3] > 0
        assert z.mean() == pytest.approx(0.0, abs=1e-9)

    def test_el_orden_de_las_filas_se_conserva_al_agrupar_por_liga(self) -> None:
        """Si `groupby(...).apply(...)` devolviese las filas en orden de grupo, X
        quedaria desalineada con `entity_id` y CADA recomendacion seria de otra
        entidad, sin ningun error visible. Las ligas van entrelazadas a proposito.
        """
        df = _df_jugador([
            {"entity_id": 1, LEAGUE_KEY: "L2", "goals": 0.0},
            {"entity_id": 2, LEAGUE_KEY: "L1", "goals": 0.0},
            {"entity_id": 3, LEAGUE_KEY: "L2", "goals": 10.0},
            {"entity_id": 4, LEAGUE_KEY: "L1", "goals": 10.0},
        ])
        mf = features.construir(df, "jugador", normalizacion="por_liga")
        assert mf.entity_id.tolist() == [1, 2, 3, 4]
        assert mf.league.tolist() == ["L2", "L1", "L2", "L1"]
        i = mf.feat_names.index("goals")
        assert mf.X[:, i].tolist() == [-1.0, -1.0, 1.0, 1.0]

    def test_una_feature_constante_no_divide_por_cero(self) -> None:
        df = _df_jugador([{"entity_id": 1, "goals": 3.0}, {"entity_id": 2, "goals": 3.0}])
        mf = features.construir(df, "jugador")
        i = mf.feat_names.index("goals")
        assert mf.X[:, i].tolist() == [0.0, 0.0]

    def test_una_liga_con_una_sola_observacion_no_rompe(self) -> None:
        """std de un unico valor es NaN: se sustituye por 1."""
        df = _df_jugador([
            {"entity_id": 1, LEAGUE_KEY: "L1", "goals": 5.0},
            {"entity_id": 2, LEAGUE_KEY: "L2", "goals": 3.0},
        ])
        mf = features.construir(df, "jugador", normalizacion="por_liga")
        assert np.isfinite(mf.X).all()
        assert mf.X.tolist() == [[0.0] * mf.X.shape[1]] * 2

    def test_los_ratios_sin_definir_quedan_en_la_media(self) -> None:
        """NaN -> 0 = la media estandarizada: no inventa un valor extremo."""
        df = _df_jugador([
            {"entity_id": 1, "take_ons": 4.0, "take_ons_won": 2.0},
            {"entity_id": 2, "take_ons": 0.0, "take_ons_won": 0.0},
        ])
        mf = features.construir(df, "jugador")
        i = mf.feat_names.index("take_ons_pct")
        assert mf.X[1, i] == 0.0

    def test_los_zscores_extremos_se_recortan(self) -> None:
        """Un cameo de 5' con un gol da un per-90 de 18: sin winsorizar, ese
        z-score domina la seleccion de vecinos.
        """
        filas = [{"entity_id": i, "goals": 0.0} for i in range(49)]
        filas.append({"entity_id": 49, "goals": 1.0})
        mf = features.construir(_df_jugador(filas), "jugador")
        i = mf.feat_names.index("goals")
        assert mf.X[:, i].max() == pytest.approx(config.F_CLIP_Z)

    def test_nada_supera_el_recorte(self) -> None:
        filas = [{"entity_id": i, "goals": 0.0} for i in range(49)]
        filas.append({"entity_id": 49, "goals": 1.0})
        mf = features.construir(_df_jugador(filas), "jugador")
        assert np.abs(mf.X).max() <= config.F_CLIP_Z


class TestConstruir:
    def test_entran_m_filas_y_salen_m_filas(self) -> None:
        """El principio central: la matriz nunca colapsa observaciones."""
        df = _df_jugador([{"entity_id": 1}, {"entity_id": 1}, {"entity_id": 2}])
        mf = features.construir(df, "jugador")
        assert mf.X.shape[0] == 3
        assert len(mf.entity_id) == 3

    def test_la_matriz_no_tiene_nan_ni_infinitos(self) -> None:
        df = _df_jugador([{"entity_id": 1}, {"entity_id": 2, "passes": 10.0}])
        assert np.isfinite(features.construir(df, "jugador").X).all()

    def test_las_dimensiones_cuadran_con_los_nombres(self) -> None:
        mf = features.construir(_df_jugador([{"entity_id": 1}, {"entity_id": 2}]), "jugador")
        assert mf.X.shape[1] == len(mf.feat_names)

    def test_el_peso_del_jugador_son_sus_minutos_entre_90(self) -> None:
        """Un cameo de 10' no puede pesar lo mismo que un partido completo."""
        df = _df_jugador([
            {"entity_id": 1, "minutes_played": 90.0},
            {"entity_id": 2, "minutes_played": 45.0},
        ])
        mf = features.construir(df, "jugador")
        assert mf.weight.tolist() == [1.0, 0.5]

    def test_el_peso_del_equipo_es_uniforme(self) -> None:
        """No hay `minutes_played` de equipo: juega el partido entero."""
        df = _df_equipo([{"entity_id": 1}, {"entity_id": 2}])
        mf = features.construir(df, "equipo")
        assert mf.weight.tolist() == [1.0, 1.0]

    def test_se_arrastran_los_metadatos_por_fila(self) -> None:
        df = _df_jugador([
            {"entity_id": 7, "entity_name": "Messi", LEAGUE_KEY: "11-90"},
            {"entity_id": 8, "entity_name": "Pedri", LEAGUE_KEY: "11-90"},
        ])
        mf = features.construir(df, "jugador")
        assert mf.entity_id.tolist() == [7, 8]
        assert mf.entity_name.tolist() == ["Messi", "Pedri"]
        assert mf.league.tolist() == ["11-90", "11-90"]

    @pytest.mark.parametrize("modo", features.NORMALIZACIONES_VALIDAS)
    def test_los_modos_validos_funcionan(self, modo: str) -> None:
        df = _df_jugador([{"entity_id": 1}, {"entity_id": 2}])
        assert features.construir(df, "jugador", normalizacion=modo).X.shape[0] == 2

    def test_una_normalizacion_desconocida_falla_pronto(self) -> None:
        with pytest.raises(ValueError, match="normalizacion desconocida"):
            features.construir(_df_jugador([{}]), "jugador", normalizacion="minmax")

    def test_una_entidad_desconocida_falla_pronto(self) -> None:
        with pytest.raises(ValueError, match="entidad desconocida"):
            features.construir(_df_jugador([{}]), "arbitro")

    def test_las_dos_normalizaciones_estan_declaradas(self) -> None:
        assert set(features.NORMALIZACIONES_VALIDAS) == {"por_liga", "global"}


class TestConDatosReales:
    def test_jugador_sobre_la_bd(self, resumen_bd: dict) -> None:
        from src.similitud import data

        df = data.cargar(resumen_bd["db_path"], "jugador")
        mf = features.construir(df, "jugador")
        assert mf.X.shape[0] == resumen_bd["n_obs_jugador"]
        assert np.isfinite(mf.X).all()
        assert (mf.weight > 0).all()

    def test_equipo_sobre_la_bd(self, resumen_bd: dict) -> None:
        from src.similitud import data

        df = data.cargar(resumen_bd["db_path"], "equipo")
        mf = features.construir(df, "equipo")
        assert mf.X.shape[0] == resumen_bd["n_obs_equipo"]
        assert np.isfinite(mf.X).all()

    @pytest.mark.parametrize("entidad", ["jugador", "equipo"])
    def test_cada_liga_queda_centrada_por_separado(self, resumen_bd: dict, entidad: str) -> None:
        from src.similitud import data

        df = data.cargar(resumen_bd["db_path"], entidad)
        mf = features.construir(df, entidad, normalizacion="por_liga")
        for liga in set(mf.league):
            medias = mf.X[mf.league == liga].mean(axis=0)
            assert np.abs(medias).max() < 1e-6, f"liga {liga} descentrada"

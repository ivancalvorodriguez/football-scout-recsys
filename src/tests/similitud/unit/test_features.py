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

    def test_estan_todas_las_features_esperadas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", False)
        feats = features._derivar_jugador(_df_jugador([{}]))
        esperadas = (
            set(config.PLAYER_COUNT_FEATURES)
            | {n for n, _, _ in config.PLAYER_RATIO_FEATURES}
            | {n for n, _, _ in config.PLAYER_DIFF_FEATURES}
        )
        assert set(feats.columns) == esperadas

    def test_con_el_default_incluye_el_bloque_de_posicion(self) -> None:
        """En los modos con z-score (el default lo es) la posicion se concatena
        aqui, para estandarizarse con el resto; en modo `cruda` se añade despues.
        """
        feats = features._derivar_jugador(_df_jugador([{}]))
        assert set(config.POSITION_FEATURES) <= set(feats.columns)

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


class TestDerivarSinEstandarizar:
    """`derivar` es `construir` parando antes del z-score.

    Existe para poder MOSTRAR una metrica en la interfaz (a un ojeador «+2,41» no
    le dice nada y «3,4 pases progresivos por 90'» si). Lo critico es que sus
    columnas correspondan una a una con las del artefacto: de ahi el test de
    orden, que se ejecuta en los tres modos de POSITION_SCALING.
    """

    @pytest.mark.parametrize("escalado", ["zscore_sqrt", "zscore", "cruda"])
    def test_mismas_columnas_y_orden_que_construir(
        self, escalado: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "POSITION_SCALING", escalado)
        df = _df_jugador([{"entity_id": 1}, {"entity_id": 2}])
        assert list(features.derivar(df, "jugador").columns) == \
            features.construir(df, "jugador").feat_names

    def test_mismas_columnas_en_equipo(self) -> None:
        df = _df_equipo([{"entity_id": 1}, {"entity_id": 2}])
        assert list(features.derivar(df, "equipo").columns) == \
            features.construir(df, "equipo").feat_names

    def test_devuelve_el_per90_sin_estandarizar(self) -> None:
        """45 pases en 45' son 90 por 90', no un z-score."""
        df = _df_jugador([{"passes": 45.0, "minutes_played": 45.0}])
        assert features.derivar(df, "jugador")["passes"].iloc[0] == 90.0

    def test_el_equipo_va_por_partido(self) -> None:
        df = _df_equipo([{"passes": 500.0}])
        assert features.derivar(df, "equipo")["passes"].iloc[0] == 500.0

    def test_un_ratio_sin_denominador_queda_en_nan(self) -> None:
        """Sin regates intentados no hay porcentaje; un 0 diria «0 % de acierto»."""
        df = _df_jugador([{"take_ons": 0.0, "take_ons_won": 0.0}])
        assert np.isnan(features.derivar(df, "jugador")["take_ons_pct"].iloc[0])

    def test_entidad_desconocida(self) -> None:
        with pytest.raises(ValueError, match="entidad desconocida"):
            features.derivar(_df_jugador([{}]), "arbitro")


class TestMasa:
    def test_el_jugador_pesa_por_minutos(self) -> None:
        df = _df_jugador([{"minutes_played": 45.0}, {"minutes_played": 90.0}])
        assert list(features.masa(df, "jugador")) == [0.5, 1.0]

    def test_el_equipo_pesa_uniforme(self) -> None:
        """Juega el partido completo: no hay minutos de equipo en la BD."""
        assert list(features.masa(_df_equipo([{}, {}]), "equipo")) == [1.0, 1.0]

    def test_es_la_misma_que_usa_construir(self) -> None:
        df = _df_jugador([{"minutes_played": 30.0}, {"minutes_played": 90.0}])
        assert list(features.masa(df, "jugador")) == \
            list(features.construir(df, "jugador").weight)

    def test_entidad_desconocida(self) -> None:
        with pytest.raises(ValueError, match="entidad desconocida"):
            features.masa(_df_jugador([{}]), "arbitro")


class TestPosicionOpcional:
    """La posicion (one-hot ponderado) forma parte del vector del jugador por
    defecto; `config.USE_POSITION_FEATURES` la apaga y nunca afecta al equipo.
    """

    def test_encendida_por_defecto(self) -> None:
        mf = features.construir(_df_jugador([{"entity_id": 1}, {"entity_id": 2}]), "jugador")
        assert set(config.POSITION_FEATURES) <= set(mf.feat_names)

    def test_apagada_no_añade_columnas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", False)
        mf = features.construir(_df_jugador([{"entity_id": 1}, {"entity_id": 2}]), "jugador")
        assert not any(c.startswith("pos_") for c in mf.feat_names)

    def test_hay_25_slugs_canonicos_declarados(self) -> None:
        assert len(config.POSITION_FEATURES) == 25
        assert config.POSITION_FEATURES[0] == "pos_goalkeeper"

    def test_encendida_añade_las_25_columnas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", True)
        mf = features.construir(_df_jugador([{"entity_id": 1}, {"entity_id": 2}]), "jugador")
        assert set(config.POSITION_FEATURES) <= set(mf.feat_names)
        assert mf.X.shape[1] == len(mf.feat_names)

    def test_el_equipo_nunca_recibe_posicion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """El equipo no tiene posicion: el flag no debe tocar su vector."""
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", True)
        mf = features.construir(_df_equipo([{"entity_id": 1}, {"entity_id": 2}]), "equipo")
        assert not any(c.startswith("pos_") for c in mf.feat_names)

    def test_columnas_de_posicion_ausentes_se_rellenan_a_cero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Un df sin la capa `pos_*` (construido a mano) no debe romper: 0 = sin
        senal de posicion, no NaN.
        """
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", True)
        mf = features.construir(_df_jugador([{"entity_id": 1}, {"entity_id": 2}]), "jugador")
        assert np.isfinite(mf.X).all()

    def test_escalado_desconocido_falla_pronto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "POSITION_SCALING", "raiz_cuadrada")
        with pytest.raises(ValueError, match="POSITION_SCALING"):
            features.construir(_df_jugador([{"entity_id": 1}]), "jugador")


class TestPosicionCruda:
    """Modo alternativo: la fraccion de minutos entra en [0,1], fuera del
    z-score, para que el BLOQUE entero pese como una feature (no como 25, ni como
    1/25), pero sin estandarizar: `POSITION_SCALING = "cruda"`.
    """

    @pytest.fixture(autouse=True)
    def _modo_crudo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", True)
        monkeypatch.setattr(config, "POSITION_SCALING", "cruda")

    def test_conserva_la_fraccion_sin_estandarizar(self) -> None:
        """Con z-score estas dos filas darian [1, -1]; crudas quedan [1, 0]."""
        df = _df_jugador([
            {"entity_id": 1, "pos_right_back": 1.0},
            {"entity_id": 2, "pos_center_forward": 1.0},
        ])
        mf = features.construir(df, "jugador")
        i = mf.feat_names.index("pos_right_back")
        assert mf.X[:, i].tolist() == [1.0, 0.0]

    def test_reparto_de_minutos_entre_dos_posiciones(self) -> None:
        """Un jugador que cambia de posicion conserva sus fracciones tal cual."""
        df = _df_jugador([
            {"entity_id": 1, "pos_right_back": 0.75, "pos_right_wing": 0.25},
            {"entity_id": 2, "pos_center_forward": 1.0},
        ])
        mf = features.construir(df, "jugador")
        rb = mf.feat_names.index("pos_right_back")
        rw = mf.feat_names.index("pos_right_wing")
        assert mf.X[0, rb] == pytest.approx(0.75)
        assert mf.X[0, rw] == pytest.approx(0.25)

    def test_el_bloque_pesa_como_una_sola_feature(self) -> None:
        """El criterio del escalado, comprobado sobre la distancia^2.

        Una feature z-scoreada de varianza 1 aporta 2.0 a la distancia^2 entre
        dos observaciones; el bloque de posicion, con dos one-hot distintos,
        tiene que aportar lo mismo. Si se dividiese por sqrt(n_posiciones)
        aportaria 25 veces menos.
        """
        df = _df_jugador([
            {"entity_id": 1, "pos_right_back": 1.0},
            {"entity_id": 2, "pos_center_forward": 1.0},
        ])
        mf = features.construir(df, "jugador")
        cols = [mf.feat_names.index(c) for c in config.POSITION_FEATURES]
        bloque = mf.X[:, cols]
        d2_bloque = float(np.sum((bloque[0] - bloque[1]) ** 2))
        assert d2_bloque == pytest.approx(2.0)

    def test_no_la_toca_el_winsorizado(self) -> None:
        """Al ir fuera del clip, el bloque nunca sale de [0,1]."""
        df = _df_jugador([
            {"entity_id": i, "pos_goalkeeper": 1.0} for i in range(1, 6)
        ] + [{"entity_id": 6, "pos_right_back": 1.0}])
        mf = features.construir(df, "jugador")
        cols = [mf.feat_names.index(c) for c in config.POSITION_FEATURES]
        bloque = mf.X[:, cols]
        assert bloque.min() >= 0.0 and bloque.max() <= 1.0

    def test_no_altera_las_demas_features(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Encender la posicion en modo crudo solo AÑADE columnas: las que ya
        estaban se estandarizan igual que sin el flag (no entran en su z-score).
        """
        filas = [
            {"entity_id": 1, "shots": 3.0, "pos_right_back": 1.0},
            {"entity_id": 2, "shots": 1.0, "pos_center_forward": 1.0},
        ]
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", False)
        sin_pos = features.construir(_df_jugador(filas), "jugador")
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", True)
        con_pos = features.construir(_df_jugador(filas), "jugador")
        i = sin_pos.feat_names.index("shots")
        j = con_pos.feat_names.index("shots")
        assert con_pos.X[:, j] == pytest.approx(sin_pos.X[:, i])


class TestPosicionZscore:
    """Modo alternativo: las 25 columnas pasan por el z-score sin dividir, con lo
    que el bloque acaba pesando como ~25 features. El default (`zscore_sqrt`)
    parte de este y lo reescala.
    """

    def test_el_modo_por_defecto_es_el_reescalado(self) -> None:
        assert config.POSITION_SCALING == "zscore_sqrt"

    def test_estandariza_la_fraccion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Aqui si: dos filas one-hot opuestas dan z = [1, -1] en cada columna."""
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", True)
        monkeypatch.setattr(config, "POSITION_SCALING", "zscore")
        df = _df_jugador([
            {"entity_id": 1, "pos_right_back": 1.0},
            {"entity_id": 2, "pos_center_forward": 1.0},
        ])
        mf = features.construir(df, "jugador")
        i = mf.feat_names.index("pos_right_back")
        assert mf.X[:, i].tolist() == [1.0, -1.0]

    def test_el_bloque_pesa_mucho_mas_que_una_feature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contraste explicito con `cruda`: aqui el bloque aporta 2 por cada
        columna con senal, no 2 en total.
        """
        monkeypatch.setattr(config, "USE_POSITION_FEATURES", True)
        monkeypatch.setattr(config, "POSITION_SCALING", "zscore")
        df = _df_jugador([
            {"entity_id": 1, "pos_right_back": 1.0},
            {"entity_id": 2, "pos_center_forward": 1.0},
        ])
        mf = features.construir(df, "jugador")
        cols = [mf.feat_names.index(c) for c in config.POSITION_FEATURES]
        bloque = mf.X[:, cols]
        d2_bloque = float(np.sum((bloque[0] - bloque[1]) ** 2))
        assert d2_bloque == pytest.approx(8.0)  # 2 columnas con senal x 4


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

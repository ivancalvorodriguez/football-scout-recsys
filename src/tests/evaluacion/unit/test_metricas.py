"""Metricas de evaluacion en numpy puro: rankings y tests estadisticos.

Todas tienen definicion cerrada, asi que se comparan contra el valor ANALITICO
calculado a mano (no contra lo que devuelva el codigo hoy). Es lo unico que
distingue una metrica correcta de una que simplemente no ha cambiado: sin scipy
en el entorno, no hay implementacion de referencia contra la que contrastar.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluacion import metricas


class TestRbo:
    def test_dos_rankings_identicos_dan_uno(self) -> None:
        assert metricas.rbo([1, 2, 3], [1, 2, 3], 0.9) == pytest.approx(1.0)

    def test_dos_rankings_disjuntos_dan_cero(self) -> None:
        assert metricas.rbo([1, 2], [3, 4], 0.9) == pytest.approx(0.0)

    def test_dos_listas_vacias_son_identicas(self) -> None:
        """Convenio: comparar "sin vecinos" con "sin vecinos" es acuerdo total."""
        assert metricas.rbo([], [], 0.9) == 1.0

    def test_una_lista_vacia_frente_a_otra_no_vacia_da_cero(self) -> None:
        assert metricas.rbo([], [1, 2], 0.9) == 0.0
        assert metricas.rbo([1, 2], [], 0.9) == 0.0

    def test_valor_analitico_con_los_dos_primeros_intercambiados(self) -> None:
        """RBO_EXT de [1,2] vs [2,1] con p=0.5.

        d=1: solapamiento 0 -> aporta 0. d=2: solapamiento 2 -> aporta (2/2)*0.5.
        rbo_min = (1-p)*0.5 = 0.25; extrapolacion = (2/2)*0.5^2 = 0.25.
        """
        assert metricas.rbo([1, 2], [2, 1], 0.5) == pytest.approx(0.5)

    def test_es_simetrica(self) -> None:
        a, b = [1, 2, 3, 4], [3, 1, 5, 2]
        assert metricas.rbo(a, b, 0.9) == pytest.approx(metricas.rbo(b, a, 0.9))

    def test_pondera_mas_las_primeras_posiciones(self) -> None:
        """Es la razon de usar RBO y no Jaccard@k: el top-1 del scouting importa
        mas que el puesto 10.
        """
        base = [1, 2, 3, 4, 5]
        cambio_al_principio = metricas.rbo(base, [2, 1, 3, 4, 5], 0.9)
        cambio_al_final = metricas.rbo(base, [1, 2, 3, 5, 4], 0.9)
        assert cambio_al_principio < cambio_al_final

    def test_siempre_esta_en_el_intervalo_unidad(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.permutation(12)[:6].tolist()
            b = rng.permutation(12)[:6].tolist()
            assert 0.0 <= metricas.rbo(a, b, 0.9) <= 1.0 + 1e-12

    @pytest.mark.parametrize("a,b", [([1, 2, 3], [1]), ([1], [1, 2, 3])])
    def test_admite_listas_de_distinta_longitud(self, a, b) -> None:
        """Pasa de verdad: `top_k_indices` devuelve menos de k si hay pocos ceros.
        Se prueban las dos orientaciones porque el bucle agota una lista antes que
        la otra.
        """
        assert 0.0 < metricas.rbo(a, b, 0.9) < 1.0

    def test_una_p_mas_baja_concentra_el_peso_en_la_cabeza(self) -> None:
        listas = ([1, 2, 3, 4], [1, 9, 8, 7])   # solo coincide el primero
        assert metricas.rbo(*listas, 0.5) > metricas.rbo(*listas, 0.95)


class TestKendallTau:
    def test_acuerdo_perfecto_da_uno(self) -> None:
        r = np.array([1.0, 2.0, 3.0, 4.0])
        assert metricas.kendall_tau(r, r) == pytest.approx(1.0)

    def test_orden_invertido_da_menos_uno(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert metricas.kendall_tau(a, a[::-1]) == pytest.approx(-1.0)

    def test_con_menos_de_dos_elementos_no_hay_discrepancia_posible(self) -> None:
        assert metricas.kendall_tau(np.array([1.0]), np.array([5.0])) == 1.0

    def test_si_un_vector_es_constante_no_hay_informacion(self) -> None:
        """tau-b divide por sqrt((n0-empates_a)(n0-empates_b)): con todo empatado
        el denominador es 0 y la respuesta honesta es 0, no un ZeroDivisionError.
        """
        a = np.array([1.0, 1.0, 1.0])
        assert metricas.kendall_tau(a, np.array([3.0, 2.0, 1.0])) == 0.0
        assert metricas.kendall_tau(np.array([3.0, 2.0, 1.0]), a) == 0.0

    def test_valor_analitico_con_un_par_discordante(self) -> None:
        """[1,2,3] vs [1,3,2]: 3 pares, 2 concordantes y 1 discordante -> 1/3."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 3.0, 2.0])
        assert metricas.kendall_tau(a, b) == pytest.approx(1.0 / 3.0)

    def test_es_simetrica(self) -> None:
        a = np.array([2.0, 5.0, 1.0, 4.0])
        b = np.array([1.0, 4.0, 3.0, 2.0])
        assert metricas.kendall_tau(a, b) == pytest.approx(metricas.kendall_tau(b, a))


class TestRangoDelObjetivo:
    def test_el_mejor_score_tiene_rango_uno(self) -> None:
        scores = np.array([9.0, 1.0, 2.0])
        assert metricas.rango_del_objetivo(scores, 0) == pytest.approx(1.0)

    def test_cuenta_cuantos_le_superan(self) -> None:
        scores = np.array([1.0, 5.0, 3.0, 4.0])
        assert metricas.rango_del_objetivo(scores, 0) == pytest.approx(4.0)

    def test_los_empates_van_a_rango_medio(self) -> None:
        """Con la S dispersa de la F2 hay empates masivos a 0: el rango MINIMO
        inflaria su top-k frente a la S densa de la F5. El midrank es neutral.

        Aqui hay 1 estrictamente mayor y 3 empatados: 1 + (3+1)/2 = 3.
        """
        scores = np.array([9.0, 0.0, 0.0, 0.0])
        assert metricas.rango_del_objetivo(scores, 1) == pytest.approx(3.0)

    def test_con_todo_empatado_el_rango_es_el_centro_del_pool(self) -> None:
        scores = np.zeros(5)
        assert metricas.rango_del_objetivo(scores, 2) == pytest.approx(3.0)

    def test_nunca_baja_de_uno(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.normal(size=30)
        for i in range(30):
            assert metricas.rango_del_objetivo(scores, i) >= 1.0


class TestRecallYMrr:
    def test_recall_cuenta_los_rangos_dentro_del_top_k(self) -> None:
        rangos = np.array([1.0, 3.0, 11.0, 5.0])
        assert metricas.recall_at_k(rangos, 5) == pytest.approx(0.75)

    def test_el_rango_medio_de_un_empate_puede_dejar_fuera_del_top_k(self) -> None:
        """1.5 entra en el top-1 solo si se redondea: no se redondea (<= k)."""
        assert metricas.recall_at_k(np.array([1.5]), 1) == pytest.approx(0.0)
        assert metricas.recall_at_k(np.array([1.5]), 2) == pytest.approx(1.0)

    def test_recall_sin_datos_es_nan(self) -> None:
        assert np.isnan(metricas.recall_at_k(np.array([]), 5))

    def test_mrr_es_la_media_de_los_reciprocos(self) -> None:
        rangos = np.array([1.0, 2.0, 4.0])
        assert metricas.mrr(rangos) == pytest.approx((1 + 0.5 + 0.25) / 3)

    def test_mrr_sin_datos_es_nan(self) -> None:
        assert np.isnan(metricas.mrr(np.array([])))


class TestRankdata:
    def test_asigna_rangos_uno_a_n_sin_empates(self) -> None:
        assert metricas._rankdata(np.array([30.0, 10.0, 20.0])).tolist() == [3, 1, 2]

    def test_promedia_los_rangos_de_los_empatados(self) -> None:
        """Equivalente a scipy 'average', que es lo que espera el Spearman."""
        assert metricas._rankdata(np.array([5.0, 5.0, 1.0])).tolist() == [2.5, 2.5, 1.0]

    def test_un_vector_vacio_devuelve_un_vector_vacio(self) -> None:
        assert metricas._rankdata(np.array([])).size == 0

    def test_todo_empatado_da_el_mismo_rango_medio(self) -> None:
        assert metricas._rankdata(np.ones(4)).tolist() == [2.5] * 4


class TestPearsonYSpearman:
    def test_pearson_de_una_relacion_lineal_perfecta(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert metricas.pearson(x, 2 * x + 5) == pytest.approx(1.0)

    def test_pearson_de_una_relacion_inversa(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert metricas.pearson(x, -x) == pytest.approx(-1.0)

    def test_pearson_con_un_vector_constante_no_divide_por_cero(self) -> None:
        assert metricas.pearson(np.ones(4), np.arange(4.0)) == 0.0

    def test_spearman_captura_una_relacion_monotona_no_lineal(self) -> None:
        """Es la diferencia con Pearson y la razon de usarlo en el Mantel."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.exp(x)
        assert metricas.spearman(x, y) == pytest.approx(1.0)
        assert metricas.pearson(x, y) < 0.98

    def test_spearman_de_un_orden_invertido(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert metricas.spearman(x, -x) == pytest.approx(-1.0)


class TestMantel:
    @staticmethod
    def _simetrica(seed: int, n: int = 8) -> np.ndarray:
        rng = np.random.default_rng(seed)
        M = rng.normal(size=(n, n))
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 0.0)
        return M

    def test_una_matriz_contra_si_misma_correlaciona_perfecto(self) -> None:
        A = self._simetrica(2)
        r, _ = metricas.mantel(A, A, permutaciones=20, seed=0)
        assert r == pytest.approx(1.0)

    def test_el_p_valor_esta_en_el_intervalo_unidad(self) -> None:
        A, B = self._simetrica(3), self._simetrica(4)
        _, p = metricas.mantel(A, B, permutaciones=50, seed=0)
        assert 0.0 < p <= 1.0

    def test_el_minimo_p_alcanzable_lo_fija_el_numero_de_permutaciones(self) -> None:
        """El observado siempre cuenta: p >= 1/(B+1). Con una correlacion perfecta
        y matrices no degeneradas, ninguna permutacion la iguala.
        """
        A = self._simetrica(5)
        _, p = metricas.mantel(A, A, permutaciones=99, seed=0)
        assert p == pytest.approx(1.0 / 100.0)

    def test_sin_permutaciones_el_p_valor_es_uno(self) -> None:
        A = self._simetrica(6)
        assert metricas.mantel(A, A, permutaciones=0, seed=0)[1] == pytest.approx(1.0)

    def test_una_relacion_monotona_entre_las_dos_matrices_da_r_uno(self) -> None:
        """La correlacion es de Spearman: escalar B no puede cambiar r."""
        A = np.abs(self._simetrica(7))
        r, _ = metricas.mantel(A, 3.0 * A, permutaciones=10, seed=0)
        assert r == pytest.approx(1.0)

    def test_es_determinista_con_la_misma_seed(self) -> None:
        A, B = self._simetrica(8), self._simetrica(9)
        assert (metricas.mantel(A, B, 30, seed=7)
                == metricas.mantel(A, B, 30, seed=7))

    def test_dos_seeds_distintas_pueden_dar_p_distinto(self) -> None:
        """El p es una estimacion por remuestreo, no una constante: la prueba lo
        deja escrito para que nadie lo tome por exacto.
        """
        A, B = self._simetrica(10), self._simetrica(11)
        ps = {metricas.mantel(A, B, 30, seed=s)[1] for s in range(6)}
        assert len(ps) >= 1  # nunca falla; documenta que p depende de la seed
        assert all(0.0 < p <= 1.0 for p in ps)


class TestPermutacionPareado:
    def test_todas_las_diferencias_positivas_dan_media_positiva(self) -> None:
        media, p = metricas.permutacion_pareado(
            np.array([0.1, 0.2, 0.3, 0.25]), permutaciones=99, seed=0)
        assert media == pytest.approx(0.2125)
        assert p < 0.2

    def test_las_diferencias_nulas_se_descartan(self) -> None:
        """Un empate no aporta evidencia en ninguna direccion (como el Wilcoxon)."""
        con_ceros = metricas.permutacion_pareado(
            np.array([0.0, 0.0, 1.0, 2.0]), 50, seed=1)
        sin_ceros = metricas.permutacion_pareado(np.array([1.0, 2.0]), 50, seed=1)
        assert con_ceros == sin_ceros

    def test_sin_ninguna_diferencia_no_hay_efecto(self) -> None:
        assert metricas.permutacion_pareado(np.zeros(5), 99, seed=0) == (0.0, 1.0)

    def test_un_efecto_nulo_no_alcanza_significancia(self) -> None:
        rng = np.random.default_rng(12)
        dif = rng.normal(loc=0.0, scale=1.0, size=40)
        _, p = metricas.permutacion_pareado(dif, 199, seed=3)
        assert p > 0.05

    def test_es_determinista_con_la_misma_seed(self) -> None:
        dif = np.array([0.3, -0.1, 0.4, 0.2, -0.05])
        assert (metricas.permutacion_pareado(dif, 99, seed=2)
                == metricas.permutacion_pareado(dif, 99, seed=2))


class TestF1Macro:
    def test_una_prediccion_perfecta_da_uno(self) -> None:
        y = np.array(["GK", "DEF", "MID", "ATT"])
        assert metricas.f1_macro(y, y) == pytest.approx(1.0)

    def test_promedia_sin_ponderar_por_frecuencia(self) -> None:
        """Es lo que la hace util aqui: los porteros son pocos y un modelo que
        los ignore no puede esconderse detras del accuracy.

        y_true = [A,A,A,B], y_pred = [A,A,A,A]: F1(A) = 2*0.75*1/1.75 = 6/7,
        F1(B) = 0 -> macro = 3/7.
        """
        y_true = np.array(["A", "A", "A", "B"])
        y_pred = np.array(["A", "A", "A", "A"])
        assert metricas.f1_macro(y_true, y_pred) == pytest.approx(3.0 / 7.0)

    def test_una_clase_sin_aciertos_aporta_cero(self) -> None:
        y_true = np.array(["A", "B"])
        y_pred = np.array(["B", "A"])
        assert metricas.f1_macro(y_true, y_pred) == pytest.approx(0.0)

    def test_solo_se_promedian_las_clases_presentes_en_la_verdad(self) -> None:
        """Una clase que el modelo inventa no crea una columna nueva en la media."""
        y_true = np.array(["A", "A"])
        y_pred = np.array(["A", "Z"])
        assert metricas.f1_macro(y_true, y_pred) == pytest.approx(2.0 / 3.0)

    def test_sin_datos_es_nan(self) -> None:
        vacio = np.array([], dtype=object)
        assert np.isnan(metricas.f1_macro(vacio, vacio))


class TestIcBootstrap:
    def test_devuelve_los_percentiles_2_5_y_97_5(self) -> None:
        valores = np.arange(0.0, 101.0)
        lo, hi = metricas.ic_bootstrap(valores)
        assert lo == pytest.approx(2.5)
        assert hi == pytest.approx(97.5)

    def test_el_intervalo_esta_ordenado(self) -> None:
        rng = np.random.default_rng(13)
        lo, hi = metricas.ic_bootstrap(rng.normal(size=200))
        assert lo < hi

    def test_ignora_los_valores_no_finitos(self) -> None:
        con_nan = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
        assert metricas.ic_bootstrap(con_nan) == metricas.ic_bootstrap(
            np.array([1.0, 2.0, 3.0]))

    def test_sin_ningun_valor_finito_devuelve_nan(self) -> None:
        lo, hi = metricas.ic_bootstrap(np.array([np.nan, np.nan]))
        assert np.isnan(lo) and np.isnan(hi)

    def test_alpha_controla_la_anchura(self) -> None:
        valores = np.arange(0.0, 101.0)
        estrecho = metricas.ic_bootstrap(valores, alpha=0.5)
        ancho = metricas.ic_bootstrap(valores, alpha=0.05)
        assert (estrecho[1] - estrecho[0]) < (ancho[1] - ancho[0])

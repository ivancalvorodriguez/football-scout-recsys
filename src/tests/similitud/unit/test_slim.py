"""Nucleo SLIM: elastic-net no negativo, kNN, W instancia-instancia, EASE.

Donde hay solucion analitica se compara contra ella (no contra lo que el codigo
devuelve hoy): asi la prueba sabe cual es la respuesta correcta y no solo detecta
cambios.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.similitud import slim


class TestElasticNetNoNegativo:
    def test_recupera_una_solucion_exacta_sin_regularizacion(self) -> None:
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = slim.elasticnet_no_negativo(A, np.array([2.0, 3.0]), beta=0.0, l1=0.0)
        assert w == pytest.approx([2.0, 3.0])

    def test_la_solucion_nunca_es_negativa(self) -> None:
        """La restriccion w>=0 es lo que hace la similitud interpretable: un peso
        negativo diria "se parece por NO parecerse".
        """
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = slim.elasticnet_no_negativo(A, np.array([-5.0, 3.0]), beta=0.0, l1=0.0)
        assert (w >= 0).all()
        assert w[0] == 0.0

    def test_el_ridge_encoge_la_solucion(self) -> None:
        """min (1/2)(2-w)^2 + (1/2)w^2  ->  w = 1 (analitico)."""
        A = np.array([[1.0]])
        w = slim.elasticnet_no_negativo(A, np.array([2.0]), beta=1.0, l1=0.0)
        assert w == pytest.approx([1.0])

    def test_el_l1_aplica_soft_thresholding(self) -> None:
        """min (1/2)(2-w)^2 + 0.5*w  ->  w = 1.5 (analitico)."""
        A = np.array([[1.0]])
        w = slim.elasticnet_no_negativo(A, np.array([2.0]), beta=0.0, l1=0.5)
        assert w == pytest.approx([1.5])

    def test_un_l1_mayor_que_la_correlacion_anula_el_peso(self) -> None:
        """Es el mecanismo que da dispersion: los vecinos irrelevantes salen."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = slim.elasticnet_no_negativo(A, np.array([2.0, 3.0]), beta=0.0, l1=10.0)
        assert w == pytest.approx([0.0, 0.0])

    def test_a_mas_l1_mas_dispersion(self) -> None:
        """El L1 es el mando de la dispersion de W: sin el, la matriz seria densa."""
        rng = np.random.default_rng(0)
        A = np.abs(rng.normal(size=(20, 8)))
        y = np.abs(rng.normal(size=20))
        sin_l1 = int((slim.elasticnet_no_negativo(A, y, beta=0.1, l1=0.0) > 0).sum())
        con_l1 = int((slim.elasticnet_no_negativo(A, y, beta=0.1, l1=5.0) > 0).sum())
        assert sin_l1 > con_l1

    def test_sin_variables_devuelve_un_vector_vacio(self) -> None:
        """Ocurre con una sola observacion: no hay vecinas de las que reconstruir."""
        w = slim.elasticnet_no_negativo(np.zeros((3, 0)), np.ones(3), beta=1.0, l1=0.5)
        assert w.shape == (0,)

    def test_una_columna_nula_no_divide_por_cero(self) -> None:
        A = np.array([[0.0], [0.0]])
        w = slim.elasticnet_no_negativo(A, np.array([1.0, 1.0]), beta=0.0, l1=0.0)
        assert np.isfinite(w).all()
        assert w == pytest.approx([0.0])

    def test_reduce_el_error_de_reconstruccion(self) -> None:
        rng = np.random.default_rng(1)
        A = np.abs(rng.normal(size=(30, 5)))
        y = A @ np.array([1.0, 0.0, 2.0, 0.0, 0.5])
        w = slim.elasticnet_no_negativo(A, y, beta=0.01, l1=0.01)
        assert np.linalg.norm(y - A @ w) < np.linalg.norm(y)

    def test_es_determinista(self) -> None:
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        y = np.array([1.0, 2.0])
        a = slim.elasticnet_no_negativo(A, y, beta=0.5, l1=0.1)
        b = slim.elasticnet_no_negativo(A, y, beta=0.5, l1=0.1)
        assert a == pytest.approx(b)


class TestVecinosMasCercanos:
    X = np.array([[0.0], [1.0], [3.0], [10.0]])

    def test_encuentra_el_vecino_mas_cercano(self) -> None:
        assert slim.vecinos_mas_cercanos(self.X, 1).tolist() == [[1], [0], [1], [2]]

    def test_nunca_se_devuelve_a_si_mismo(self) -> None:
        """En SLIM, w_ss = 0: reconstruirse consigo mismo daria la solucion trivial."""
        vecinos = slim.vecinos_mas_cercanos(self.X, 3)
        for i, fila in enumerate(vecinos):
            assert i not in fila

    def test_k_se_recorta_al_numero_de_vecinas_disponibles(self) -> None:
        """`F2_N_NEIGHBORS` son 100 pero la BD puede tener menos observaciones."""
        assert slim.vecinos_mas_cercanos(self.X, 100).shape == (4, 3)

    def test_con_una_sola_observacion_no_hay_vecinas(self) -> None:
        assert slim.vecinos_mas_cercanos(np.array([[1.0, 2.0]]), 5).shape == (1, 0)

    def test_coincide_con_la_busqueda_por_fuerza_bruta(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 7))
        k = 5
        vecinos = slim.vecinos_mas_cercanos(X, k)
        for i in range(len(X)):
            d = np.linalg.norm(X - X[i], axis=1)
            d[i] = np.inf
            esperados = set(np.argsort(d)[:k].tolist())
            assert set(vecinos[i].tolist()) == esperados

    def test_el_calculo_por_bloques_cruza_bien_la_frontera(self) -> None:
        """Se calcula por bloques de 512 filas para no materializar M x M; un fallo
        de indice solo aparece a partir de la segunda pasada.
        """
        rng = np.random.default_rng(4)
        X = rng.normal(size=(600, 3))
        vecinos = slim.vecinos_mas_cercanos(X, 2)
        assert vecinos.shape == (600, 2)
        for i in (0, 511, 512, 599):
            d = np.linalg.norm(X - X[i], axis=1)
            d[i] = np.inf
            assert set(vecinos[i].tolist()) == set(np.argsort(d)[:2].tolist())


class TestSlimInstancia:
    @pytest.fixture
    def X(self) -> np.ndarray:
        rng = np.random.default_rng(5)
        return rng.normal(size=(25, 6))

    def test_devuelve_una_columna_por_observacion(self, X: np.ndarray) -> None:
        cols_idx, cols_val = slim.slim_instancia(X, n_neighbors=5, beta=1.0, l1=0.1)
        assert len(cols_idx) == len(X)
        assert len(cols_val) == len(X)

    def test_ninguna_observacion_se_reconstruye_consigo_misma(self, X: np.ndarray) -> None:
        cols_idx, _ = slim.slim_instancia(X, n_neighbors=5, beta=1.0, l1=0.1)
        for s, idx in enumerate(cols_idx):
            assert s not in idx.tolist()

    def test_solo_se_guardan_pesos_positivos(self, X: np.ndarray) -> None:
        _, cols_val = slim.slim_instancia(X, n_neighbors=5, beta=1.0, l1=0.1)
        for val in cols_val:
            assert (val > 0).all()

    def test_los_indices_y_los_pesos_van_alineados(self, X: np.ndarray) -> None:
        cols_idx, cols_val = slim.slim_instancia(X, n_neighbors=5, beta=1.0, l1=0.1)
        for idx, val in zip(cols_idx, cols_val):
            assert idx.shape == val.shape

    def test_solo_se_reconstruye_desde_las_vecinas(self, X: np.ndarray) -> None:
        """Seleccion de candidatos tipo fsSLIM: es lo que hace tratable la W MxM."""
        vecinos = slim.vecinos_mas_cercanos(X, 5)
        cols_idx, _ = slim.slim_instancia(X, n_neighbors=5, beta=1.0, l1=0.1)
        for s, idx in enumerate(cols_idx):
            assert set(idx.tolist()) <= set(vecinos[s].tolist())

    def test_un_l1_alto_deja_la_w_vacia(self, X: np.ndarray) -> None:
        _, cols_val = slim.slim_instancia(X, n_neighbors=5, beta=1.0, l1=1e6)
        assert sum(v.size for v in cols_val) == 0

    def test_las_observaciones_identicas_se_reconstruyen_entre_si(self) -> None:
        """Comprobacion de sentido: dos copias del mismo vector deben acabar con
        un peso mutuo, no con cero.
        """
        X = np.array([[5.0, 5.0], [5.0, 5.0], [-4.0, 1.0], [0.0, -3.0]])
        cols_idx, cols_val = slim.slim_instancia(X, n_neighbors=3, beta=0.01, l1=0.01)
        assert 1 in cols_idx[0].tolist()
        assert 0 in cols_idx[1].tolist()


class TestAgregarWAEntidades:
    def test_agrega_los_bloques_de_w_normalizando_por_masa(self) -> None:
        # 4 observaciones: 0,1 -> entidad 0;  2,3 -> entidad 1.
        # W[2,0] = 0.5 (la observacion 2 ayuda a reconstruir la 0).
        cols_idx = [np.array([2]), np.array([], dtype=np.int64),
                    np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
        cols_val = [np.array([0.5]), np.array([]), np.array([]), np.array([])]
        row_entity = np.array([0, 0, 1, 1])
        weight = np.ones(4)
        S = slim.agregar_W_a_entidades(cols_idx, cols_val, row_entity, weight, 2)
        # num[1,0] = m_2 * m_0 * 0.5 = 0.5;  Z = masa_1 * masa_0 = 2*2 = 4.
        # Simetrizado: 0.5 * (0.125 + 0) = 0.0625.
        assert S == pytest.approx(np.array([[0.0, 0.0625], [0.0625, 0.0]]))

    def test_la_matriz_resultante_es_simetrica(self) -> None:
        rng = np.random.default_rng(6)
        cols_idx = [rng.choice(6, size=2, replace=False) for _ in range(6)]
        cols_val = [np.abs(rng.normal(size=2)) for _ in range(6)]
        row_entity = np.array([0, 0, 1, 1, 2, 2])
        S = slim.agregar_W_a_entidades(cols_idx, cols_val, row_entity, np.ones(6), 3)
        assert S == pytest.approx(S.T)

    def test_la_diagonal_es_cero(self) -> None:
        cols_idx = [np.array([1]), np.array([0])]
        cols_val = [np.array([1.0]), np.array([1.0])]
        S = slim.agregar_W_a_entidades(cols_idx, cols_val, np.array([0, 0]), np.ones(2), 1)
        assert np.diag(S) == pytest.approx([0.0])

    def test_normalizar_por_la_masa_evita_que_domine_quien_juega_mas(self) -> None:
        """Sin dividir por N_p*N_q, un jugador con 40 partidos saldria similar a
        todo el mundo solo por acumular masa (el PDF: "con normalizacion por N_p,
        no domina").

        La entidad 1 tiene el DOBLE de observaciones que la 2 y cada una aporta lo
        mismo: la similitud con la 0 debe salir igual, no el doble.
        """
        # obs 0 -> ent 0 | obs 1,2 -> ent 1 (dos partidos) | obs 3 -> ent 2
        row_entity = np.array([0, 1, 1, 2])
        cols_idx = [np.array([1, 2, 3]), np.array([], dtype=np.int64),
                    np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
        cols_val = [np.array([0.5, 0.5, 0.5]), np.array([]), np.array([]), np.array([])]
        S = slim.agregar_W_a_entidades(cols_idx, cols_val, row_entity, np.ones(4), 3)
        # Numeradores sin normalizar: ent 1 -> 1.0, ent 2 -> 0.5 (dominaria la 1).
        # Normalizados por la masa: 1.0/2 = 0.5 y 0.5/1 = 0.5. Iguales.
        assert S[0, 1] == pytest.approx(S[0, 2])
        assert S[0, 1] == pytest.approx(0.25)   # 0.5 * (0.5 + 0) al simetrizar

    def test_la_masa_por_minutos_pondera_las_observaciones_de_una_entidad(self) -> None:
        """Opcion (b) del PDF: las masas entran al AGREGAR W, no al ajustar.

        Las entidades 1 y 2 tienen cada una una observacion que se parece a la 0 y
        otra que no. La diferencia: la observacion discrepante de la entidad 2 es
        un cameo (masa 0.1), asi que arrastra menos su media y la 2 debe quedar
        mas similar a la 0.
        """
        # obs 0 -> ent 0 | obs 1,2 -> ent 1 | obs 3,4 -> ent 2
        row_entity = np.array([0, 1, 1, 2, 2])
        # Solo las obs 1 y 3 ayudan a reconstruir la 0; las obs 2 y 4 no aportan.
        cols_idx = [np.array([1, 3])] + [np.array([], dtype=np.int64) for _ in range(4)]
        cols_val = [np.array([1.0, 1.0])] + [np.array([]) for _ in range(4)]
        pesos = np.array([1.0, 1.0, 1.0, 1.0, 0.1])   # la obs 4 es un cameo
        S = slim.agregar_W_a_entidades(cols_idx, cols_val, row_entity, pesos, 3)
        assert S[0, 2] > S[0, 1]
        assert S[0, 1] == pytest.approx(0.5 * (1.0 / 2.0))   # masa de la ent 1 = 2.0
        assert S[0, 2] == pytest.approx(0.5 * (1.0 / 1.1))   # masa de la ent 2 = 1.1

    def test_una_entidad_sin_masa_no_divide_por_cero(self) -> None:
        row_entity = np.array([0, 1])
        cols_idx = [np.array([1]), np.array([], dtype=np.int64)]
        cols_val = [np.array([1.0]), np.array([])]
        S = slim.agregar_W_a_entidades(cols_idx, cols_val, row_entity, np.zeros(2), 2)
        assert np.isfinite(S).all()

    def test_una_w_vacia_da_similitud_cero(self) -> None:
        vacio = [np.array([], dtype=np.int64) for _ in range(3)]
        vals = [np.array([]) for _ in range(3)]
        S = slim.agregar_W_a_entidades(vacio, vals, np.array([0, 1, 2]), np.ones(3), 3)
        assert S == pytest.approx(np.zeros((3, 3)))


class TestEase:
    def test_la_diagonal_es_cero(self) -> None:
        """Restriccion del modelo: una entidad no se explica consigo misma."""
        rng = np.random.default_rng(7)
        S = np.abs(rng.normal(size=(5, 5)))
        S = 0.5 * (S + S.T)
        B = slim.ease(S, lam=1.0)
        assert np.diag(B) == pytest.approx(np.zeros(5))

    def test_coincide_con_la_formula_cerrada_de_steck(self) -> None:
        rng = np.random.default_rng(8)
        S = np.abs(rng.normal(size=(4, 4)))
        lam = 2.0
        B = slim.ease(S, lam)
        P = np.linalg.inv(S.T @ S + lam * np.eye(4))
        esperado = -P / np.diag(P)[None, :]
        np.fill_diagonal(esperado, 0.0)
        assert B == pytest.approx(esperado)

    def test_una_similitud_identidad_no_aprende_nada(self) -> None:
        """Sin relaciones cruzadas no hay nada que propagar: B = 0."""
        B = slim.ease(np.eye(4), lam=1.0)
        assert B == pytest.approx(np.zeros((4, 4)))

    def test_conserva_la_forma(self) -> None:
        rng = np.random.default_rng(9)
        S = np.abs(rng.normal(size=(6, 6)))
        assert slim.ease(S, lam=10.0).shape == (6, 6)

    def test_una_lambda_alta_encoge_los_pesos(self) -> None:
        rng = np.random.default_rng(10)
        S = np.abs(rng.normal(size=(6, 6)))
        S = 0.5 * (S + S.T)
        assert np.abs(slim.ease(S, lam=1000.0)).sum() < np.abs(slim.ease(S, lam=1.0)).sum()

    def test_una_matriz_singular_no_rompe_gracias_a_la_regularizacion(self) -> None:
        """S^T S es singular si hay entidades identicas; lam*I lo arregla."""
        S = np.ones((4, 4))
        assert np.isfinite(slim.ease(S, lam=1.0)).all()

"""Etapa 1 de la Formulacion 5: similitud distribucional entre nubes.

Lo que se vigila: que se compare la FORMA de la distribucion de cada entidad (no
un promedio de sus features) y que ningun paso numerico haga underflow o divida
por cero.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.similitud import config, distributional as D
from src.similitud.features import MatrizFeatures


def _mf(X: np.ndarray, weight: np.ndarray | None = None) -> MatrizFeatures:
    n = len(X)
    return MatrizFeatures(
        X=X,
        feat_names=[f"f{i}" for i in range(X.shape[1])],
        entity_id=np.arange(n),
        entity_name=np.array([f"e{i}" for i in range(n)]),
        weight=np.ones(n) if weight is None else weight,
        league=np.array(["L1"] * n),
    )


class TestLogSumExp:
    def test_coincide_con_el_calculo_directo(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert D._logsumexp(a, axis=1) == pytest.approx(np.log(np.exp(a).sum(axis=1)))

    def test_por_columnas(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert D._logsumexp(a, axis=0) == pytest.approx(np.log(np.exp(a).sum(axis=0)))

    def test_es_estable_con_valores_muy_grandes(self) -> None:
        """El calculo ingenuo (log(sum(exp))) desbordaria a inf."""
        a = np.array([[1000.0, 1000.0]])
        assert D._logsumexp(a, axis=1) == pytest.approx([1000.0 + np.log(2)])

    def test_es_estable_con_valores_muy_pequenos(self) -> None:
        a = np.array([[-1000.0, -1000.0]])
        assert D._logsumexp(a, axis=1) == pytest.approx([-1000.0 + np.log(2)])

    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
    def test_con_menos_infinito_no_produce_nan(self) -> None:
        """Sinkhorn llega aqui con -inf cuando una masa es cero. El resultado debe
        ser -inf (que las iteraciones absorben), nunca NaN (que envenenaria toda
        la matriz de costes).
        """
        a = np.array([[-np.inf, -np.inf]])
        assert not np.isnan(D._logsumexp(a, axis=1)).any()


class TestSigmaMediana:
    def test_es_positivo(self) -> None:
        rng = np.random.default_rng(0)
        assert D._sigma_mediana(rng.normal(size=(50, 4)), seed=1) > 0

    def test_es_determinista_con_la_misma_semilla(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        assert D._sigma_mediana(X, seed=1) == D._sigma_mediana(X, seed=1)

    def test_escala_con_la_dispersion_de_los_datos(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3))
        assert D._sigma_mediana(10.0 * X, seed=1) > D._sigma_mediana(X, seed=1)

    def test_con_puntos_identicos_cae_a_la_mediana_por_defecto(self) -> None:
        """Todas las distancias son 0, asi que no hay mediana que estimar: se cae a
        med=1 -> sigma=sqrt(1/2). Lo que importa es que sea > 0: un sigma de 0
        haria que el kernel RBF dividiese por cero.
        """
        sigma = D._sigma_mediana(np.ones((10, 3)), seed=1)
        assert sigma == pytest.approx(np.sqrt(0.5))
        assert sigma > 0

    def test_con_una_sola_observacion_no_rompe(self) -> None:
        """No hay ningun par del que sacar distancias."""
        sigma = D._sigma_mediana(np.array([[1.0, 2.0]]), seed=1)
        assert sigma == pytest.approx(np.sqrt(0.5))
        assert sigma > 0


class TestRandomFourierFeatures:
    def test_tiene_la_dimension_pedida(self) -> None:
        rng = np.random.default_rng(0)
        Z = D._rff(rng.normal(size=(10, 4)), dim=32, sigma=1.0, seed=5)
        assert Z.shape == (10, 32)

    def test_es_determinista_con_la_misma_semilla(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 4))
        assert D._rff(X, 32, 1.0, seed=5) == pytest.approx(D._rff(X, 32, 1.0, seed=5))

    def test_semillas_distintas_dan_embeddings_distintos(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 4))
        assert not np.allclose(D._rff(X, 32, 1.0, seed=5), D._rff(X, 32, 1.0, seed=6))

    def test_aproxima_el_kernel_rbf(self) -> None:
        """Es la razon de ser del RFF: z(x).z(y) ~ k(x,y) = exp(-||x-y||^2/2s^2)."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(6, 3))
        sigma = 1.5
        Z = D._rff(X, dim=20000, sigma=sigma, seed=11)
        aprox = Z @ Z.T
        sq = np.einsum("ij,ij->i", X, X)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        exacto = np.exp(-d2 / (2.0 * sigma ** 2))
        assert aprox == pytest.approx(exacto, abs=0.05)

    def test_el_mapa_es_no_lineal(self) -> None:
        """Es lo que permite que la media de phi(x) represente la DISTRIBUCION y no
        sea un colapso de features: phi(media) != media(phi).
        """
        X = np.array([[-2.0], [2.0]])
        Z = D._rff(X, dim=64, sigma=1.0, seed=3)
        media_de_phi = Z.mean(axis=0)
        phi_de_la_media = D._rff(X.mean(axis=0, keepdims=True), 64, 1.0, seed=3)[0]
        assert not np.allclose(media_de_phi, phi_de_la_media)


class TestEmbedEntidades:
    def test_promedia_las_observaciones_de_cada_entidad(self) -> None:
        Z = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]])
        mu = D._embed_entidades(Z, np.array([0, 0, 1]), np.ones(3), 2)
        assert mu == pytest.approx(np.array([[2.0, 0.0], [0.0, 5.0]]))

    def test_pondera_por_la_masa(self) -> None:
        """Un cameo de 9' no debe mover el perfil como un partido completo."""
        Z = np.array([[0.0], [10.0]])
        mu = D._embed_entidades(Z, np.array([0, 0]), np.array([1.0, 0.1]), 1)
        assert mu == pytest.approx(np.array([[10.0 * 0.1 / 1.1]]))

    def test_una_entidad_sin_masa_no_divide_por_cero(self) -> None:
        Z = np.array([[1.0], [2.0]])
        mu = D._embed_entidades(Z, np.array([0, 0]), np.zeros(2), 2)
        assert np.isfinite(mu).all()


class TestSimilitudMMD:
    def test_es_simetrica_con_diagonal_cero(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 4))
        ent = np.repeat(np.arange(6), 5)
        S = D.similitud_mmd(_mf(X), ent, 6)
        assert S.shape == (6, 6)
        assert S == pytest.approx(S.T)
        assert np.diag(S) == pytest.approx(np.zeros(6))

    def test_dos_entidades_con_la_misma_nube_son_lo_mas_similar(self) -> None:
        rng = np.random.default_rng(3)
        nube = rng.normal(size=(8, 3))
        lejos = rng.normal(size=(8, 3)) + 8.0
        X = np.vstack([nube, nube, lejos])
        ent = np.repeat([0, 1, 2], 8)
        S = D.similitud_mmd(_mf(X), ent, 3)
        assert S[0, 1] > S[0, 2]
        assert S[0, 1] > S[1, 2]

    def test_el_coseno_acota_la_similitud(self) -> None:
        """Normalizar por las normas deja el ANGULO entre embeddings; con kernel
        RBF, phi tiene componentes > 0, asi que queda en ~[0, 1] y EASE no
        necesita recalibrar lambda.
        """
        rng = np.random.default_rng(4)
        X = rng.normal(size=(40, 5))
        S = D.similitud_mmd(_mf(X), np.repeat(np.arange(8), 5), 8)
        assert S.min() >= -1.0 and S.max() <= 1.0

    def test_no_confunde_parecido_de_forma_con_consistencia(self) -> None:
        """El producto interno crudo <mu_a, mu_b> escala con ||mu_a||, que mide lo
        CONCENTRADA que es la nube. Sin normalizar, un jugador muy regular saldria
        similar a todos. Aqui las entidades 0 y 1 comparten centro y solo difieren
        en dispersion; 2 esta en otro sitio: 0-1 debe superar a 0-2 igualmente.
        """
        rng = np.random.default_rng(5)
        compacta = rng.normal(scale=0.05, size=(10, 3))
        dispersa = rng.normal(scale=1.5, size=(10, 3))
        otra = rng.normal(scale=0.05, size=(10, 3)) + 6.0
        X = np.vstack([compacta, dispersa, otra])
        S = D.similitud_mmd(_mf(X), np.repeat([0, 1, 2], 10), 3)
        assert S[0, 1] > S[0, 2]

    def test_una_entidad_con_una_sola_observacion_no_rompe(self) -> None:
        """Un jugador con un unico partido debe entrar sin caso especial."""
        rng = np.random.default_rng(6)
        X = rng.normal(size=(11, 3))
        ent = np.array([0] * 10 + [1])
        S = D.similitud_mmd(_mf(X), ent, 2)
        assert np.isfinite(S).all()

    def test_es_determinista(self) -> None:
        """La semilla del RFF esta fijada en config: dos ejecuciones del build
        deben dar el mismo modelo.
        """
        rng = np.random.default_rng(7)
        X = rng.normal(size=(20, 4))
        ent = np.repeat(np.arange(4), 5)
        assert D.similitud_mmd(_mf(X), ent, 4) == pytest.approx(
            D.similitud_mmd(_mf(X), ent, 4)
        )


class TestSimilitudSinkhorn:
    def test_es_simetrica_con_diagonal_cero(self) -> None:
        rng = np.random.default_rng(8)
        X = rng.normal(size=(12, 3))
        ent = np.repeat(np.arange(4), 3)
        S = D.similitud_sinkhorn(_mf(X), ent, 4)
        assert S.shape == (4, 4)
        assert S == pytest.approx(S.T)
        assert np.diag(S) == pytest.approx(np.zeros(4))

    def test_las_nubes_cercanas_son_mas_similares_que_las_lejanas(self) -> None:
        rng = np.random.default_rng(9)
        base = rng.normal(size=(4, 2))
        X = np.vstack([base, base + 0.1, base + 20.0])
        S = D.similitud_sinkhorn(_mf(X), np.repeat([0, 1, 2], 4), 3)
        assert S[0, 1] > S[0, 2]

    def test_el_kernel_no_hace_underflow(self) -> None:
        """En ~30 dimensiones estandarizadas el coste OT vale decenas y exp(-coste)
        daria toda S ~ 0, sin contraste. Se reescala por la mediana de los costes.
        """
        rng = np.random.default_rng(10)
        X = rng.normal(size=(24, 30)) * 3.0
        S = D.similitud_sinkhorn(_mf(X), np.repeat(np.arange(6), 4), 6)
        fuera = S[~np.eye(6, dtype=bool)]
        assert fuera.max() > 0.01           # hay senal
        assert fuera.std() > 1e-6           # y hay contraste

    def test_el_coste_de_una_nube_consigo_misma_es_casi_cero(self) -> None:
        rng = np.random.default_rng(11)
        nube = rng.normal(size=(5, 3))
        coste = D._sinkhorn_costo(nube, np.ones(5), nube.copy(), np.ones(5),
                                  reg=config.F5_SINKHORN_REG, iters=200)
        assert coste == pytest.approx(0.0, abs=0.2)

    def test_el_coste_crece_con_la_distancia_entre_nubes(self) -> None:
        rng = np.random.default_rng(12)
        a = rng.normal(size=(5, 2))
        cerca = D._sinkhorn_costo(a, np.ones(5), a + 1.0, np.ones(5), 0.5, 100)
        lejos = D._sinkhorn_costo(a, np.ones(5), a + 5.0, np.ones(5), 0.5, 100)
        assert lejos > cerca

    def test_el_coste_es_simetrico(self) -> None:
        rng = np.random.default_rng(13)
        a, b = rng.normal(size=(4, 2)), rng.normal(size=(6, 2))
        ab = D._sinkhorn_costo(a, np.ones(4), b, np.ones(6), 0.5, 100)
        ba = D._sinkhorn_costo(b, np.ones(6), a, np.ones(4), 0.5, 100)
        assert ab == pytest.approx(ba, rel=1e-6)

    def test_admite_nubes_de_distinto_tamano(self) -> None:
        """Los equipos no juegan todos los mismos partidos."""
        rng = np.random.default_rng(14)
        X = rng.normal(size=(9, 2))
        ent = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])   # 6 obs vs 3
        S = D.similitud_sinkhorn(_mf(X), ent, 2)
        assert np.isfinite(S).all()


class TestConstruirS:
    @pytest.mark.parametrize("metodo", ["mmd", "sinkhorn"])
    def test_despacha_por_metodo(self, metodo: str) -> None:
        rng = np.random.default_rng(15)
        X = rng.normal(size=(12, 3))
        S = D.construir_S(_mf(X), np.repeat(np.arange(4), 3), 4, metodo)
        assert S.shape == (4, 4)
        assert np.diag(S) == pytest.approx(np.zeros(4))

    def test_un_metodo_desconocido_falla_pronto(self) -> None:
        rng = np.random.default_rng(16)
        X = rng.normal(size=(6, 2))
        with pytest.raises(ValueError, match="metodo distribucional desconocido"):
            D.construir_S(_mf(X), np.repeat([0, 1], 3), 2, "bures")

    def test_los_metodos_configurados_son_validos(self) -> None:
        assert config.F5_METODO_JUGADOR in ("mmd", "sinkhorn")
        assert config.F5_METODO_EQUIPO in ("mmd", "sinkhorn")

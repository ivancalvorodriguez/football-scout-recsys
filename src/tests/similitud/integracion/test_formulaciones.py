"""Integracion de las dos formulaciones: BD -> features -> ajuste -> modelo servible.

Se ajustan de verdad sobre la BD sintetica (nada de mocks): es lo unico que
comprueba que las piezas (`slim`, `distributional`, `modelo`) encajan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.similitud import config, data, features, formulacion2, formulacion5
from src.similitud.features import MatrizFeatures
from src.similitud.modelo import ModeloSimilitud, top_k

pytestmark = [pytest.mark.integracion, pytest.mark.lento]


@pytest.fixture(scope="module")
def mf_jugador(bd_sintetica: Path) -> MatrizFeatures:
    return features.construir(data.cargar(bd_sintetica, "jugador"), "jugador")


@pytest.fixture(scope="module")
def mf_equipo(bd_sintetica: Path) -> MatrizFeatures:
    return features.construir(data.cargar(bd_sintetica, "equipo"), "equipo")


@pytest.fixture(scope="module")
def f2_jugador(mf_jugador: MatrizFeatures) -> ModeloSimilitud:
    return formulacion2.construir(mf_jugador, "jugador", "por_liga")


@pytest.fixture(scope="module")
def f5_jugador(mf_jugador: MatrizFeatures) -> ModeloSimilitud:
    return formulacion5.construir(mf_jugador, "jugador", "por_liga")


@pytest.fixture(scope="module")
def f5_equipo(mf_equipo: MatrizFeatures) -> ModeloSimilitud:
    return formulacion5.construir(mf_equipo, "equipo", "por_liga")


def _n_entidades(mf: MatrizFeatures) -> int:
    return len(set(mf.entity_id.tolist()))


class TestContratoComun:
    """Ambas formulaciones deben devolver un modelo servible equivalente."""

    @pytest.fixture(params=["f2_jugador", "f5_jugador"])
    def modelo(self, request: pytest.FixtureRequest) -> ModeloSimilitud:
        return request.getfixturevalue(request.param)

    def test_s_es_cuadrada_del_tamano_del_universo(
        self, modelo: ModeloSimilitud, mf_jugador: MatrizFeatures
    ) -> None:
        n = _n_entidades(mf_jugador)
        assert modelo.S.shape == (n, n)

    def test_s_es_simetrica(self, modelo: ModeloSimilitud) -> None:
        """Se simetriza a proposito para que el ranking sea estable."""
        assert modelo.S == pytest.approx(modelo.S.T)

    def test_la_diagonal_es_cero(self, modelo: ModeloSimilitud) -> None:
        assert np.diag(modelo.S) == pytest.approx(np.zeros(len(modelo.S)))

    def test_s_no_tiene_nan_ni_infinitos(self, modelo: ModeloSimilitud) -> None:
        assert np.isfinite(modelo.S).all()

    def test_el_indice_va_alineado_con_s(self, modelo: ModeloSimilitud) -> None:
        assert len(modelo.entity_ids) == len(modelo.S)
        assert len(modelo.entity_names) == len(modelo.S)

    def test_feat_display_tiene_una_fila_por_entidad(
        self, modelo: ModeloSimilitud, mf_jugador: MatrizFeatures
    ) -> None:
        assert modelo.feat_display.shape == (len(modelo.S), len(mf_jugador.feat_names))

    def test_el_meta_declara_el_principio_central(
        self, modelo: ModeloSimilitud, mf_jugador: MatrizFeatures
    ) -> None:
        """M observaciones entran al ajuste y P entidades salen, con M > P: si
        fuesen iguales, alguien habria promediado los partidos antes de entrar.
        """
        assert modelo.meta["n_observaciones"] == mf_jugador.X.shape[0]
        assert modelo.meta["n_entidades"] == _n_entidades(mf_jugador)
        assert modelo.meta["n_observaciones"] > modelo.meta["n_entidades"]

    def test_el_meta_lleva_la_normalizacion_y_la_descripcion(
        self, modelo: ModeloSimilitud
    ) -> None:
        assert modelo.meta["normalizacion"] == "por_liga"
        assert modelo.meta["descripcion"]

    def test_el_meta_lleva_las_ligas_de_cada_entidad(self, modelo: ModeloSimilitud) -> None:
        ligas = modelo.meta["ligas_por_entidad"]
        assert set(ligas) == {str(i) for i in modelo.entity_ids}
        assert all(v for v in ligas.values())

    def test_el_top_k_devuelve_entidades_validas(self, modelo: ModeloSimilitud) -> None:
        for j, score in top_k(modelo, 0, 5):
            assert 0 <= j < len(modelo.S)
            assert j != 0
            assert np.isfinite(score)

    def test_el_top_k_va_ordenado_de_mas_a_menos_similar(
        self, modelo: ModeloSimilitud
    ) -> None:
        scores = [s for _, s in top_k(modelo, 0, 5)]
        assert scores == sorted(scores, reverse=True)


class TestFormulacion2:
    def test_se_identifica_como_la_2(self, f2_jugador: ModeloSimilitud) -> None:
        assert f2_jugador.formulacion == "2"
        assert f2_jugador.entidad == "jugador"

    def test_el_meta_lleva_los_hiperparametros_de_slim(self, f2_jugador: ModeloSimilitud) -> None:
        assert f2_jugador.meta["n_neighbors"] == config.F2_N_NEIGHBORS
        assert f2_jugador.meta["beta"] == config.F2_BETA
        assert f2_jugador.meta["l1"] == config.F2_L1

    def test_la_w_aprendida_es_dispersa_pero_no_vacia(self, f2_jugador: ModeloSimilitud) -> None:
        """Si nnz fuese 0, el L1 estaria matando todo y S seria toda ceros."""
        nnz = f2_jugador.meta["nnz_W"]
        M = f2_jugador.meta["n_observaciones"]
        assert nnz > 0
        assert nnz < M * M

    def test_la_similitud_no_es_negativa(self, f2_jugador: ModeloSimilitud) -> None:
        """W se aprende con w>=0 y la agregacion solo suma y divide por masas."""
        assert (f2_jugador.S >= 0).all()

    def test_la_matriz_es_dispersa(self, f2_jugador: ModeloSimilitud) -> None:
        """Muchos pares no tienen relacion aprendida: `top_k` descarta esos ceros."""
        assert (f2_jugador.S == 0).sum() > 0


class TestFormulacion5:
    def test_se_identifica_como_la_5(self, f5_jugador: ModeloSimilitud) -> None:
        assert f5_jugador.formulacion == "5"

    def test_el_jugador_usa_mmd(self, f5_jugador: ModeloSimilitud) -> None:
        """Escala a miles de entidades y es robusto con un solo partido."""
        assert f5_jugador.meta["metodo_distribucional"] == "mmd"
        assert f5_jugador.meta["rff_dim"] == config.F5_RFF_DIM
        assert f5_jugador.meta["sinkhorn_reg"] is None

    def test_el_equipo_usa_sinkhorn(self, f5_equipo: ModeloSimilitud) -> None:
        """Hay pocos equipos: el OT entropico exacto sale barato."""
        assert f5_equipo.meta["metodo_distribucional"] == "sinkhorn"
        assert f5_equipo.meta["sinkhorn_reg"] == config.F5_SINKHORN_REG
        assert f5_equipo.meta["rff_dim"] is None

    def test_el_meta_lleva_la_lambda_de_ease(self, f5_jugador: ModeloSimilitud) -> None:
        assert f5_jugador.meta["ease_lambda"] == config.F5_EASE_LAMBDA

    def test_el_modelo_de_equipo_cubre_a_todos_los_equipos(
        self, f5_equipo: ModeloSimilitud, mf_equipo: MatrizFeatures
    ) -> None:
        assert len(f5_equipo.S) == _n_entidades(mf_equipo)
        assert f5_equipo.entidad == "equipo"


class TestDiferenciasEntreFormulaciones:
    def test_comparten_el_universo_de_entidades(
        self, f2_jugador: ModeloSimilitud, f5_jugador: ModeloSimilitud
    ) -> None:
        """Es lo que permite comparar sus rankings sin traducir indices."""
        assert f2_jugador.entity_ids.tolist() == f5_jugador.entity_ids.tolist()
        assert f2_jugador.entity_names == f5_jugador.entity_names

    def test_son_modelos_distintos(
        self, f2_jugador: ModeloSimilitud, f5_jugador: ModeloSimilitud
    ) -> None:
        """Optimizan cosas distintas (partidos vs entidades): si S saliese igual,
        una de las dos no estaria haciendo lo que dice.
        """
        assert not np.allclose(f2_jugador.S, f5_jugador.S)


class TestNormalizaciones:
    @pytest.mark.parametrize("modo", ["por_liga", "global"])
    def test_las_dos_normalizaciones_producen_modelo(
        self, bd_sintetica: Path, modo: str
    ) -> None:
        mf = features.construir(data.cargar(bd_sintetica, "jugador"), "jugador",
                                normalizacion=modo)
        modelo = formulacion5.construir(mf, "jugador", modo)
        assert modelo.meta["normalizacion"] == modo
        assert np.isfinite(modelo.S).all()

    def test_con_dos_ligas_las_normalizaciones_dan_modelos_distintos(
        self, bd_sintetica: Path
    ) -> None:
        """Con una sola liga serian identicas por construccion; la BD sintetica
        tiene dos justo para que esta comparacion diga algo.
        """
        modelos = {}
        for modo in ("por_liga", "global"):
            mf = features.construir(data.cargar(bd_sintetica, "jugador"), "jugador",
                                    normalizacion=modo)
            modelos[modo] = formulacion5.construir(mf, "jugador", modo)
        assert not np.allclose(modelos["por_liga"].S, modelos["global"].S)


class TestDeterminismo:
    def test_dos_ajustes_seguidos_dan_el_mismo_modelo(self, mf_equipo: MatrizFeatures) -> None:
        """Sin determinismo, `comparar` mediria ruido en vez de la normalizacion."""
        a = formulacion5.construir(mf_equipo, "equipo", "por_liga")
        b = formulacion5.construir(mf_equipo, "equipo", "por_liga")
        assert a.S == pytest.approx(b.S)

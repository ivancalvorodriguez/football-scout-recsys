"""La distancia llega de verdad al ajuste, y la euclidea no ha cambiado nada.

Las pruebas de `test_distancias.py` fijan la geometria en abstracto; estas la
siguen hasta el artefacto: que cambiar de distancia cambia la S, que cada una
produce su propio fichero sin pisar al vecino, y —lo mas importante para no
invalidar lo que ya hay en `outputs/`— que el camino euclideo sigue dando
exactamente lo mismo que cuando no habia distancia que elegir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.similitud import build, data, features, formulacion2, formulacion5
from src.similitud import distancias as D
from src.similitud.modelo import cargar_modelo, stem_artefacto
from src.similitud.warm import EstadoWarm, ruta_estado

pytestmark = pytest.mark.integracion

_FORMULACIONES = {"2": formulacion2, "5": formulacion5}


@pytest.fixture(scope="module")
def mf(bd_sintetica: Path):
    return features.construir(data.cargar(bd_sintetica, "equipo"), "equipo",
                              normalizacion="global")


class TestLaEuclideaNoSeMueve:
    """Todo `outputs/` se construyo con ella: su camino tiene que ser el de antes."""

    @pytest.mark.parametrize("formulacion", ["2", "5"])
    def test_no_pasarla_es_lo_mismo_que_pasarla(self, mf, formulacion) -> None:
        por_defecto = _FORMULACIONES[formulacion].construir(mf, "equipo", "global")
        explicita = _FORMULACIONES[formulacion].construir(
            mf, "equipo", "global", distancia="euclidea")
        assert por_defecto.S == pytest.approx(explicita.S)

    def test_el_artefacto_euclideo_no_lleva_sufijo(self, mf, tmp_path) -> None:
        """Si lo llevara habria que renombrar miles de ficheros ya construidos."""
        modelo = formulacion5.construir(mf, "equipo", "global", distancia="euclidea")
        assert modelo.guardar(tmp_path).name == "formulacion5_equipo_global.npz"

    def test_un_artefacto_sin_distancia_se_lee_como_euclideo(
        self, mf, tmp_path
    ) -> None:
        """Es lo que hay en `outputs/`: se construyo antes de que existiera el eje."""
        modelo = formulacion5.construir(mf, "equipo", "global")
        modelo.meta.pop("distancia")
        modelo.guardar(tmp_path)
        assert cargar_modelo(tmp_path, "5", "equipo", "global",
                             distancia="euclidea").S == pytest.approx(modelo.S)


class TestCadaDistanciaEsOtroModelo:
    @pytest.mark.parametrize("formulacion", ["2", "5"])
    @pytest.mark.parametrize("distancia", ["mahalanobis", "coseno", "manhattan"])
    def test_cambiar_la_distancia_cambia_la_similitud(
        self, mf, formulacion, distancia
    ) -> None:
        """Si la S saliera igual, la flag no estaria midiendo nada."""
        base = _FORMULACIONES[formulacion].construir(mf, "equipo", "global")
        otra = _FORMULACIONES[formulacion].construir(
            mf, "equipo", "global", distancia=distancia)
        assert not np.allclose(base.S, otra.S)

    @pytest.mark.parametrize("distancia", D.DISTANCIAS_VALIDAS)
    def test_la_distancia_queda_declarada_en_el_artefacto(
        self, mf, distancia
    ) -> None:
        modelo = formulacion5.construir(mf, "equipo", "global", distancia=distancia)
        assert modelo.meta["distancia"] == distancia

    def test_los_artefactos_conviven_sin_pisarse(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        for distancia in D.DISTANCIAS_VALIDAS:
            build._construir_uno(bd_sintetica, tmp_path, "5", "equipo", "global",
                                 distancia=distancia)
        assert {p.name for p in tmp_path.glob("*.npz") if ".warm" not in p.name} == {
            f"{stem_artefacto('5', 'equipo', 'global', d)}.npz"
            for d in D.DISTANCIAS_VALIDAS
        }

    def test_la_S_sigue_siendo_servible(self, mf) -> None:
        """Simetrica y con la diagonal a cero, sea cual sea la geometria."""
        for distancia in D.DISTANCIAS_VALIDAS:
            S = formulacion5.construir(mf, "equipo", "global",
                                       distancia=distancia).S
            assert S == pytest.approx(S.T)
            assert np.diag(S) == pytest.approx(np.zeros(len(S)))


class TestEstadoWarm:
    """La geometria se congela con el ajuste; si no, reentrenar mueve a todos."""

    def test_el_estado_guarda_la_distancia_y_su_blanqueo(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        build._construir_uno(bd_sintetica, tmp_path, "5", "equipo", "global",
                             distancia="mahalanobis")
        estado = EstadoWarm.cargar(
            ruta_estado(tmp_path, "5", "equipo", "global", "mahalanobis"))
        assert estado.distancia == "mahalanobis"
        assert estado.espacio_L is not None
        assert estado.espacio().nombre == "mahalanobis"

    def test_el_blanqueo_se_reutiliza_en_vez_de_reestimarse(self, mf) -> None:
        """Reestimarlo con datos ampliados moveria a las entidades que no han
        jugado, igual que recalcular las mu/sd del z-score."""
        _, previo = formulacion5.construir_con_estado(
            mf, "equipo", "global", distancia="mahalanobis")
        _, segundo = formulacion5.construir_con_estado(
            mf, "equipo", "global", previo=previo, distancia="mahalanobis")
        assert segundo.espacio_L == pytest.approx(previo.espacio_L)

    def test_un_estado_de_otra_geometria_no_se_reutiliza(self, mf) -> None:
        """Arrancar de la solucion de otro espacio seria arrancar de otro problema."""
        _, previo = formulacion2.construir_con_estado(
            mf, "equipo", "global", distancia="euclidea")
        modelo, _ = formulacion2.construir_con_estado(
            mf, "equipo", "global", previo=previo, distancia="coseno")
        assert not modelo.meta["warm"]["aplicado"]
        assert "distancia" in modelo.meta["warm"]["motivo"]

    def test_un_estado_sin_distancia_se_lee_como_euclideo(self, mf) -> None:
        _, estado = formulacion5.construir_con_estado(mf, "equipo", "global")
        assert estado.distancia == "euclidea"


class TestResiduoL1:
    """La manhattan no es un minimo cuadrado: se resuelve por IRLS."""

    def test_la_manhattan_usa_el_resolvedor_l1(self, mf, monkeypatch) -> None:
        llamadas = {"l1": 0, "l2": 0}
        from src.similitud import slim

        original_l1 = slim.elasticnet_residuo_l1
        original_l2 = slim.elasticnet_no_negativo

        def espia_l1(*a, **kw):
            llamadas["l1"] += 1
            return original_l1(*a, **kw)

        def espia_l2(*a, **kw):
            llamadas["l2"] += 1
            return original_l2(*a, **kw)

        monkeypatch.setattr(slim, "elasticnet_residuo_l1", espia_l1)
        monkeypatch.setattr(slim, "elasticnet_no_negativo", espia_l2)
        formulacion2.construir(mf, "equipo", "global", distancia="manhattan")
        assert llamadas["l1"] > 0

    def test_el_irls_reduce_el_residuo_l1_frente_al_minimo_cuadrado(self) -> None:
        """Es lo unico que hay que comprobar del metodo: que optimiza LA norma
        que dice optimizar. Con un residuo de cola pesada, el minimo cuadrado se
        deja arrastrar por el punto extremo y el L1 no."""
        from src.similitud import slim

        rng = np.random.default_rng(3)
        A = np.abs(rng.normal(size=(30, 6)))
        y = A @ np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.2])
        y[0] += 25.0                      # un outlier que domina el cuadrado

        w_l2 = slim.elasticnet_no_negativo(A, y, beta=0.1, l1=0.01)
        w_l1 = slim.elasticnet_residuo_l1(A, y, beta=0.1, l1=0.01)
        assert np.abs(y - A @ w_l1).sum() < np.abs(y - A @ w_l2).sum()

    def test_el_resolvedor_l1_respeta_la_no_negatividad(self) -> None:
        from src.similitud import slim

        rng = np.random.default_rng(5)
        A, y = rng.normal(size=(20, 5)), rng.normal(size=20)
        assert (slim.elasticnet_residuo_l1(A, y, beta=0.5, l1=0.1) >= 0).all()

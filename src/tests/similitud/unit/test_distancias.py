"""Las cuatro distancias y la geometria que definen.

Lo que se fija aqui, por orden de importancia:

1. **Cada funcion calcula lo que dice su nombre**, contrastada contra la
   definicion elemental escrita a mano (no contra otra implementacion nuestra).
2. **La euclidea no ha cambiado nada.** Es la distancia con la que se construyo y
   midio todo lo que hay en `outputs/`, asi que su camino tiene que dar
   exactamente lo de antes: mismos vecinos, mismo ancho de kernel, mismo Omega.
3. **El invariante de transformar una vez**: medir en el espacio transformado
   equivale a medir la distancia directamente sobre los vectores crudos.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.similitud import distancias as D


def _bruta(X: np.ndarray, f) -> np.ndarray:
    """Matriz de distancias calculada par a par, sin trucos."""
    n = len(X)
    return np.array([[f(X[i], X[j]) for j in range(n)] for i in range(n)])


@pytest.fixture
def X() -> np.ndarray:
    return np.random.default_rng(20240720).normal(size=(40, 6))


class TestDefiniciones:
    """Cada distancia contra su formula, escrita a mano."""

    def test_euclidea_es_la_norma_2_de_la_diferencia(self, X) -> None:
        # atol flojo a proposito: la identidad ||a||^2+||b||^2-2ab pierde
        # precision entre vectores casi iguales (la diagonal sale ~1e-8, no 0).
        assert D.euclidea(X) == pytest.approx(
            _bruta(X, lambda a, b: np.linalg.norm(a - b)), abs=1e-7)

    def test_manhattan_es_la_suma_de_diferencias_absolutas(self, X) -> None:
        assert D.manhattan(X) == pytest.approx(
            _bruta(X, lambda a, b: np.abs(a - b).sum()))

    def test_coseno_es_uno_menos_el_coseno(self, X) -> None:
        assert D.coseno(X) == pytest.approx(_bruta(
            X, lambda a, b: 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))

    def test_mahalanobis_usa_la_inversa_de_la_covarianza(self, X) -> None:
        L = D.blanqueo(X)
        precision = L @ L.T
        assert D.mahalanobis(X, L=L) == pytest.approx(
            _bruta(X, lambda a, b: np.sqrt((a - b) @ precision @ (a - b))),
            abs=1e-7)

    def test_mahalanobis_sin_blanqueo_se_niega_en_vez_de_inventarselo(self, X) -> None:
        """Estimarlo aqui pondria cada consulta en un espacio distinto."""
        with pytest.raises(ValueError, match="blanqueo"):
            D.mahalanobis(X)


class TestBloques:
    """La manhattan se trocea para acotar memoria: el resultado no cambia."""

    def test_el_troceado_no_altera_el_resultado(self, monkeypatch, X) -> None:
        entera = D.manhattan(X)
        monkeypatch.setattr(D, "MEMORIA_BLOQUE", 8)   # fuerza el peor troceado
        assert D.manhattan(X) == pytest.approx(entera)

    def test_entre_dos_matrices_distintas(self, X) -> None:
        A, B = X[:7], X[7:20]
        esperado = np.array([[np.abs(a - b).sum() for b in B] for a in A])
        assert D.manhattan(A, B) == pytest.approx(esperado)

    def test_sin_filas_devuelve_una_matriz_vacia(self, X) -> None:
        assert D.manhattan(np.empty((0, 6)), X).shape == (0, len(X))


class TestBlanqueo:
    def test_deja_la_covarianza_en_la_identidad(self, X) -> None:
        Xt = X @ D.blanqueo(X)
        cov = np.cov(Xt.T)
        assert np.diag(cov) == pytest.approx(np.ones(X.shape[1]), abs=0.01)
        fuera = cov - np.diag(np.diag(cov))
        assert np.abs(fuera).max() < 0.01

    def test_es_simetrico(self, X) -> None:
        """ZCA y no Cholesky: las dos dan la misma distancia, pero la simetrica
        deja el espacio alineado con las features originales."""
        L = D.blanqueo(X)
        assert L == pytest.approx(L.T)

    def test_una_direccion_casi_constante_no_lo_hace_explotar(self) -> None:
        """Sin el piso relativo, una columna casi nula se amplificaria hasta
        gobernar la distancia entera (p. ej. una `pos_*` que casi no aparece)."""
        rng = np.random.default_rng(1)
        X = np.hstack([rng.normal(size=(50, 3)), np.zeros((50, 1))])
        L = D.blanqueo(X)
        assert np.isfinite(L).all()

    def test_con_menos_de_dos_filas_degrada_a_la_identidad(self) -> None:
        """Sin covarianza estimable, mahalanobis se queda en euclidea en vez de
        romper el ajuste."""
        assert D.blanqueo(np.zeros((1, 4))) == pytest.approx(np.eye(4))


class TestEspacio:
    """El invariante: transformar UNA vez y medir despues == medir directamente."""

    @pytest.mark.parametrize("nombre", D.DISTANCIAS_VALIDAS)
    def test_medir_en_el_espacio_transformado_es_medir_la_distancia(
        self, nombre, X
    ) -> None:
        esp = D.preparar(X, nombre)
        medida = esp.distancias(esp.transformar(X))
        if nombre == "coseno":
            # En el espacio de filas normalizadas la euclidea es la CORDAL:
            # d^2 = 2(1-cos), monotona en la distancia coseno (mismo ranking).
            assert medida ** 2 == pytest.approx(2 * D.coseno(X), abs=1e-9)
        else:
            directa = {"euclidea": lambda: D.euclidea(X),
                       "manhattan": lambda: D.manhattan(X),
                       "mahalanobis": lambda: D.mahalanobis(X, L=esp.L)}[nombre]()
            assert medida == pytest.approx(directa, abs=1e-7)

    @pytest.mark.parametrize("nombre", D.DISTANCIAS_VALIDAS)
    def test_el_orden_de_los_vecinos_es_el_de_la_distancia(self, nombre, X) -> None:
        """`distancias2` solo promete ser monotona en la distancia; es lo que
        necesita el `argpartition` de la seleccion de candidatos."""
        esp = D.preparar(X, nombre)
        Xt = esp.transformar(X)
        d = esp.distancias(Xt)[0]
        d2 = esp.distancias2(Xt)[0]
        assert np.argsort(d).tolist() == np.argsort(d2).tolist()

    def test_solo_mahalanobis_aprende_algo_de_los_datos(self, X) -> None:
        assert D.preparar(X, "mahalanobis").L is not None
        for otra in ("euclidea", "coseno", "manhattan"):
            assert D.preparar(X, otra).L is None

    def test_la_euclidea_y_la_manhattan_no_transforman(self, X) -> None:
        """`es_identidad` evita copiar matrices de 50.000 filas para nada."""
        assert D.Espacio("euclidea").es_identidad
        assert D.Espacio("manhattan").es_identidad
        assert not D.Espacio("coseno").es_identidad

    def test_solo_la_manhattan_deja_de_ser_euclidea_tras_transformar(self) -> None:
        """De esta propiedad cuelgan las dos vias rapidas: el kNN por producto de
        matrices y el residuo como minimo cuadrado."""
        assert not D.Espacio("manhattan").euclidea_transformada
        for otra in ("euclidea", "coseno", "mahalanobis"):
            assert D.Espacio(otra).euclidea_transformada


class TestKernel:
    def test_el_ancho_euclideo_es_el_de_siempre(self, X) -> None:
        """Formula historica: sqrt(mediana(d^2)/2). No puede moverse: cambiaria
        el embedding de todos los modelos F5 ya construidos."""
        esp = D.Espacio("euclidea")
        d2 = D.euclidea(X)[np.triu_indices(len(X), k=1)] ** 2
        assert esp.ancho_kernel(X) == pytest.approx(np.sqrt(np.median(d2) / 2))

    def test_el_ancho_manhattan_es_la_mediana_de_las_l1(self, X) -> None:
        """Es la escala del kernel laplaciano, no la del RBF."""
        d1 = D.manhattan(X)[np.triu_indices(len(X), k=1)]
        assert D.Espacio("manhattan").ancho_kernel(X) == pytest.approx(np.median(d1))

    def test_sin_ningun_par_la_mediana_se_toma_como_uno(self) -> None:
        """Convencion del calculo original: no se devuelve 1.0 directamente,
        porque en el caso euclideo eso saltaria la raiz y daria otro ancho."""
        assert D.Espacio("euclidea").ancho_kernel(
            np.array([[1.0, 2.0]])) == pytest.approx(np.sqrt(0.5))

    def test_la_euclidea_muestrea_omega_como_antes(self) -> None:
        """Mismo generador, misma llamada: el mapa RFF de un modelo F5 euclideo
        tiene que salir identico al de antes de que la distancia fuera un eje."""
        esperado = np.random.default_rng(7).normal(0.0, 1.0 / 2.5, size=(3, 8))
        obtenido = D.Espacio("euclidea").omega(
            3, 8, 2.5, np.random.default_rng(7))
        assert obtenido == pytest.approx(esperado)

    def test_la_manhattan_muestrea_una_cauchy(self) -> None:
        """Por Bochner, la densidad espectral del kernel laplaciano —el que le
        corresponde a la L1— es una Cauchy, no una gaussiana."""
        esperado = np.random.default_rng(7).standard_cauchy(size=(3, 8)) / 2.5
        obtenido = D.Espacio("manhattan").omega(
            3, 8, 2.5, np.random.default_rng(7))
        assert obtenido == pytest.approx(esperado)


class TestCosteOT:
    def test_las_euclideas_dan_el_coste_cuadratico(self, X) -> None:
        """Coste cuadratico = 2-Wasserstein, que es el de siempre."""
        esp = D.Espacio("euclidea")
        assert esp.coste_ot(X[:5], X[5:9]) == pytest.approx(
            D.euclidea(X[:5], X[5:9]) ** 2, abs=1e-7)

    def test_la_manhattan_da_el_coste_lineal(self, X) -> None:
        """Coste lineal = 1-Wasserstein: mas estable si una nube trae un partido
        atipico, que es el caso de una entidad con pocas observaciones."""
        esp = D.Espacio("manhattan")
        assert esp.coste_ot(X[:5], X[5:9]) == pytest.approx(
            D.manhattan(X[:5], X[5:9]))

    def test_el_coste_nunca_es_negativo(self, X) -> None:
        """El Sinkhorn en log-domain lo exige."""
        assert (D.Espacio("euclidea").coste_ot(X[:5], X[:5]) >= 0).all()


class TestVecinos:
    def test_devuelve_los_k_mas_cercanos(self, X) -> None:
        esp = D.Espacio("euclidea")
        propia = np.arange(len(X))
        vecinos = D.vecinos(esp, X, X, 3, propia=propia)
        d = D.euclidea(X)
        for i, fila in enumerate(vecinos):
            resto = [j for j in range(len(X)) if j != i and j not in fila]
            assert max(d[i, j] for j in fila) <= min(d[i, j] for j in resto)

    def test_nunca_se_elige_a_si_misma(self, X) -> None:
        """Es el `w_ss = 0` de SLIM: sin esto cada observacion se reconstruiria
        consigo misma y la W seria la identidad."""
        vecinos = D.vecinos(D.Espacio("euclidea"), X, X, 5,
                            propia=np.arange(len(X)))
        assert all(i not in fila for i, fila in enumerate(vecinos))

    def test_el_troceado_por_bloques_no_cambia_los_vecinos(self, X) -> None:
        esp = D.Espacio("euclidea")
        propia = np.arange(len(X))
        entero = D.vecinos(esp, X, X, 4, propia=propia, paso=1000)
        troceado = D.vecinos(esp, X, X, 4, propia=propia, paso=3)
        assert np.array_equal(np.sort(entero, axis=1), np.sort(troceado, axis=1))

    def test_la_manhattan_elige_otros_vecinos_que_la_euclidea(self, X) -> None:
        """Si dieran siempre lo mismo, la flag no estaria midiendo nada."""
        propia = np.arange(len(X))
        eu = D.vecinos(D.Espacio("euclidea"), X, X, 3, propia=propia)
        man = D.vecinos(D.Espacio("manhattan"), X, X, 3, propia=propia)
        assert not np.array_equal(np.sort(eu, axis=1), np.sort(man, axis=1))

    def test_pedir_mas_vecinos_que_candidatas_no_rompe(self, X) -> None:
        vecinos = D.vecinos(D.Espacio("euclidea"), X[:4], X[:4], 100,
                            propia=np.arange(4))
        assert vecinos.shape == (4, 3)


class TestValidacion:
    def test_una_distancia_desconocida_falla_al_pedirla(self) -> None:
        with pytest.raises(ValueError, match="distancia desconocida"):
            D.validar("mahalanobbis")

    def test_none_es_la_de_por_defecto(self) -> None:
        assert D.validar(None) == D.POR_DEFECTO == "euclidea"


class TestSerializacion:
    """El espacio viaja en el estado warm: hay que poder recuperarlo entero."""

    def test_ida_y_vuelta_conserva_el_blanqueo(self, X) -> None:
        esp = D.preparar(X, "mahalanobis")
        vuelta = D.Espacio.desde_dict(esp.como_dict())
        assert vuelta.nombre == "mahalanobis"
        assert vuelta.L == pytest.approx(esp.L)

    def test_sin_datos_se_asume_la_euclidea(self) -> None:
        """Un estado warm anterior a este eje no declara distancia: era la unica."""
        assert D.Espacio.desde_dict(None).nombre == "euclidea"
        assert D.Espacio.desde_dict({}).nombre == "euclidea"

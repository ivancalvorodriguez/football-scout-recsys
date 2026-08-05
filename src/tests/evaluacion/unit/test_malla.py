"""Geometria de las superficies: escala real de cada eje e interpolacion PCHIP.

Las dos propiedades que la figura no puede perder son: (1) que los ejes respeten
la distancia REAL entre los valores barridos, porque si no una meseta ancha puede
parecer una cresta estrecha; y (2) que la interpolacion NO sobrepase, porque en
una figura cuya lectura es "donde esta el optimo" un sobrepaso dibuja un maximo que
nadie ha medido. Casi todo lo demas es resolucion de dibujo.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluacion import malla


class TestNumericos:
    def test_convierte_una_lista_de_numeros(self) -> None:
        assert malla._numericos([1, 2.5]).tolist() == [1.0, 2.5]

    def test_los_booleanos_no_tienen_coordenada_real(self) -> None:
        """`True` es un `int` para Python, pero un eje True/False no se puede
        dibujar a escala.
        """
        assert malla._numericos([True, False]) is None

    def test_un_valor_no_numerico_descarta_el_eje_entero(self) -> None:
        assert malla._numericos([1, "mmd"]) is None


class TestDetectar:
    def test_unos_valores_multiplicativos_piden_escala_logaritmica(self) -> None:
        """512/1024/2048 es perfectamente regular en log y no en lineal."""
        assert malla.detectar([512, 1024, 2048]) == "log"

    def test_unos_valores_bien_repartidos_se_quedan_en_lineal(self) -> None:
        """Ante la duda, la escala honesta es la lineal. Son los valores reales
        del eje `F2_L1` en `barrido.HIPERPARAMETROS`: equiespaciados, con un
        recorrido de x7 que por si solo bastaria para plantearse la logaritmica.
        """
        assert malla.detectar([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]) == "lineal"

    def test_unos_valores_geometricos_piden_logaritmica_aunque_sean_pequenos(
        self,
    ) -> None:
        """La decision no mira la magnitud, mira el reparto: 0.125…1.0 doblando
        cada paso esta tan mal repartido en lineal como 512/1024/2048.
        """
        assert malla.detectar([0.125, 0.25, 0.5, 1.0]) == "log"

    def test_un_recorrido_corto_no_justifica_la_logaritmica(self) -> None:
        assert malla.detectar([1.0, 2.0, 3.0]) == "lineal"

    def test_con_dos_puntos_cualquier_escala_dibuja_lo_mismo(self) -> None:
        """Declarar "log" para una recta entre dos nodos solo confundiria el pie."""
        assert malla.detectar([10, 1000]) == "lineal"

    def test_unos_valores_no_numericos_van_a_posiciones_ordinales(self) -> None:
        assert malla.detectar(["mmd", "sinkhorn", "otro"]) == "ordinal"

    def test_un_eje_de_un_solo_valor_no_tiene_coordenada(self) -> None:
        assert malla.detectar([5.0]) == "ordinal"

    def test_unos_valores_desordenados_no_admiten_escala_real(self) -> None:
        assert malla.detectar([3.0, 1.0, 2.0]) == "ordinal"

    def test_un_cero_impide_la_logaritmica(self) -> None:
        assert malla.detectar([0.0, 10.0, 1000.0]) == "lineal"


class TestConstruirEje:
    def test_los_valores_medidos_son_nodos_de_la_malla_fina(self) -> None:
        """Es lo que hace que la superficie pase EXACTAMENTE por lo medido."""
        e = malla.eje("F2_L1", [0.25, 0.5, 1.0])
        assert e.uf[e.idx] == pytest.approx(e.u)

    def test_la_malla_fina_es_creciente(self) -> None:
        e = malla.eje("F2_L1", [0.25, 0.5, 1.0])
        assert np.all(np.diff(e.uf) > 0)

    def test_reparte_los_nodos_en_proporcion_al_ancho_real_del_tramo(self) -> None:
        """Un tramo `5 -> 7.5` no puede recibir la misma resolucion que un `1 -> 5`."""
        e = malla.eje("x", [0.0, 1.0, 11.0], densidad=44)
        estrecho = int(e.idx[1] - e.idx[0])
        ancho = int(e.idx[2] - e.idx[1])
        assert ancho > estrecho

    def test_con_densidad_cero_se_dibuja_la_rejilla_cruda(self) -> None:
        e = malla.eje("x", [1.0, 2.0, 3.0], densidad=0)
        assert e.uf.tolist() == e.u.tolist()

    def test_la_escala_logaritmica_usa_el_logaritmo_como_coordenada(self) -> None:
        e = malla.eje("d", [10, 100, 1000], escala="log")
        assert e.u.tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_la_escala_ordinal_reparte_a_intervalos_iguales(self) -> None:
        e = malla.eje("d", [1, 10, 1000], escala="ordinal")
        assert e.u.tolist() == [0.0, 1.0, 2.0]

    def test_un_eje_no_numerico_cae_a_ordinal_avisando(self) -> None:
        avisos: list[str] = []
        e = malla.eje("m", ["mmd", "sinkhorn"], escala="lineal", avisar=avisos.append)
        assert e.escala == "ordinal"
        assert avisos and "ordinales" in avisos[0]

    def test_la_logaritmica_con_valores_no_positivos_cae_a_lineal_avisando(self) -> None:
        avisos: list[str] = []
        e = malla.eje("x", [0.0, 1.0, 2.0], escala="log", avisar=avisos.append)
        assert e.escala == "lineal"
        assert avisos and "logaritmica no es posible" in avisos[0]

    def test_auto_elige_la_escala_por_eje(self) -> None:
        assert malla.eje("d", [512, 1024, 2048]).escala == "log"
        assert malla.eje("l1", [0.125, 0.25, 0.375, 0.5]).escala == "lineal"


class TestEje:
    def test_las_etiquetas_llevan_siempre_el_valor_real(self) -> None:
        """Sea cual sea la escala: el tick no puede anunciar un logaritmo."""
        e = malla.eje("d", [512, 1024, 2048])
        assert e.etiquetas == ["512", "1024", "2048"]

    def test_los_flotantes_se_imprimen_sin_ceros_de_relleno(self) -> None:
        assert malla.eje("l1", [0.25, 0.5, 2.0]).etiquetas == ["0.25", "0.5", "2"]

    def test_normalizar_lleva_los_extremos_a_menos_uno_y_uno(self) -> None:
        """Es el cubo en que dibuja el visor HTML; conservando el reparto real, que
        es lo que hace que el HTML y el PNG enseñen la misma superficie.
        """
        e = malla.eje("x", [0.0, 1.0, 10.0])
        norm = e.normalizar(e.u)
        assert norm[0] == pytest.approx(-1.0)
        assert norm[-1] == pytest.approx(1.0)

    def test_normalizar_conserva_el_reparto_desigual(self) -> None:
        e = malla.eje("x", [0.0, 1.0, 10.0])
        norm = e.normalizar(e.u)
        assert norm[1] == pytest.approx(-0.8)

    def test_un_eje_degenerado_se_normaliza_a_cero(self) -> None:
        e = malla.eje("x", [5.0], densidad=0)
        assert e.normalizar(e.u).tolist() == [0.0]

    def test_la_mascara_marca_solo_los_nodos_medidos(self) -> None:
        """La superficie estimada se dibuja entera, pero solo esos puntos son datos."""
        e = malla.eje("x", [1.0, 2.0, 3.0])
        assert int(e.medidos().sum()) == 3
        assert e.medidos()[e.idx].all()

    def test_la_descripcion_declara_la_escala_y_el_rango(self) -> None:
        assert "logaritmica" in malla.eje("d", [10, 100, 1000], escala="log").descripcion()
        assert "lineal" in malla.eje("x", [1.0, 2.0, 3.0]).descripcion()

    def test_la_descripcion_ordinal_avisa_de_que_los_saltos_son_iguales(self) -> None:
        d = malla.eje("m", ["a", "b"], escala="ordinal").descripcion()
        assert "saltos del dibujo son iguales" in d


class TestInterpolar:
    @pytest.fixture
    def ejes(self) -> tuple[malla.Eje, malla.Eje]:
        return (malla.eje("x", [0.0, 1.0, 2.0]), malla.eje("y", [0.0, 1.0, 2.0]))

    def test_la_superficie_pasa_por_los_puntos_medidos(self, ejes) -> None:
        ex, ey = ejes
        Z = np.array([[0.0, 1.0, 4.0], [1.0, 2.0, 5.0], [4.0, 5.0, 8.0]])
        fina = malla.interpolar(Z, ex, ey)
        assert fina[np.ix_(ey.idx, ex.idx)] == pytest.approx(Z)

    def test_no_sobrepasa_el_rango_de_los_puntos_medidos(self, ejes) -> None:
        """La razon de elegir PCHIP y no un spline cubico: un sobrepaso pintaria un
        optimo que nadie ha evaluado.
        """
        ex, ey = ejes
        Z = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]])
        fina = malla.interpolar(Z, ex, ey)
        assert np.nanmax(fina) <= Z.max() + 1e-9
        assert np.nanmin(fina) >= Z.min() - 1e-9

    def test_no_crea_optimos_nuevos_entre_dos_puntos(self, ejes) -> None:
        """El maximo de la superficie estimada esta siempre en un punto medido."""
        ex, ey = ejes
        Z = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        fina = malla.interpolar(Z, ex, ey)
        assert np.nanmax(fina) == pytest.approx(Z.max())

    def test_con_densidad_cero_devuelve_la_rejilla_tal_cual(self) -> None:
        ex = malla.eje("x", [0.0, 1.0], densidad=0)
        ey = malla.eje("y", [0.0, 1.0], densidad=0)
        Z = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert malla.interpolar(Z, ex, ey) == pytest.approx(Z)

    def test_devuelve_la_forma_de_la_malla_fina(self, ejes) -> None:
        ex, ey = ejes
        fina = malla.interpolar(np.zeros((3, 3)), ex, ey)
        assert fina.shape == (ey.uf.size, ex.uf.size)

    def test_donde_la_rejilla_tiene_huecos_no_extrapola(self, ejes) -> None:
        """Una combinacion evaluada con otras fases deja NaN: rellenar ahi seria
        inventar un numero que nadie ha medido.
        """
        ex, ey = ejes
        Z = np.full((3, 3), np.nan)
        Z[0, :] = [1.0, 2.0, 3.0]
        fina = malla.interpolar(Z, ex, ey)
        assert np.isnan(fina).any()

    def test_una_fila_con_un_solo_punto_no_se_interpola(self, ejes) -> None:
        ex, ey = ejes
        Z = np.full((3, 3), np.nan)
        Z[1, 1] = 5.0
        fina = malla.interpolar(Z, ex, ey)
        assert np.isnan(fina).all()


class TestPchip1d:
    def test_reproduce_los_valores_en_los_nodos(self) -> None:
        x = np.array([0.0, 1.0, 3.0, 6.0])
        y = np.array([1.0, 3.0, 2.0, 5.0])
        assert malla._linea(x, y, x) == pytest.approx(y)

    def test_una_serie_monotona_sigue_siendo_monotona(self) -> None:
        """Fritsch-Carlson: es lo que impide la ondulacion tipica del spline."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 0.1, 0.9, 1.0])
        xf = np.linspace(0.0, 3.0, 200)
        assert np.all(np.diff(malla._linea(x, y, xf)) >= -1e-12)

    def test_dos_puntos_dan_una_recta(self) -> None:
        x = np.array([0.0, 2.0])
        y = np.array([0.0, 4.0])
        assert malla._linea(x, y, np.array([1.0])) == pytest.approx([2.0])

    def test_fuera_del_tramo_medido_devuelve_nan(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        out = malla._linea(x, y, np.array([0.0, 2.0, 9.0]))
        assert np.isnan(out[0]) and np.isnan(out[2])
        assert out[1] == pytest.approx(2.0)

    def test_con_menos_de_dos_puntos_finitos_no_hay_nada_que_interpolar(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([np.nan, 5.0, np.nan])
        assert np.isnan(malla._linea(x, y, x)).all()

    def test_salta_los_puntos_no_finitos(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, np.nan, 2.0])
        assert malla._linea(x, y, np.array([1.0])) == pytest.approx([1.0])

    def test_la_pendiente_se_anula_en_los_extremos_locales(self) -> None:
        """Es el mecanismo concreto que impide pasarse del punto medido."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        assert malla._pendientes(x, y)[1] == pytest.approx(0.0)

    def test_la_pendiente_de_un_extremo_se_recorta_al_triple_de_la_diferencia(
        self,
    ) -> None:
        """Freno clasico de PCHIP: si la extrapolacion cuadratica del borde supera
        el triple de la diferencia adyacente, se recorta ahi.
        """
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, -10.0])
        assert malla._pendientes(x, y)[0] == pytest.approx(3.0)

    def test_la_pendiente_de_un_extremo_no_despega_la_curva(self) -> None:
        """Sin el recorte, la superficie se dispararia justo en el borde, que es
        donde se lee si el optimo se sale de la rejilla.
        """
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 5.0, 5.1])
        out = malla._linea(x, y, np.linspace(0.0, 2.0, 50))
        assert np.nanmax(out) <= y.max() + 1e-9

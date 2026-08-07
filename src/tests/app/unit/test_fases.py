"""Perfil por fases de juego y geometría del radar."""

from __future__ import annotations

import numpy as np
import pytest

from src.app import fases

# Cuerpo de letra de los rótulos, declarado en `static/estilo.css`
# (`.radar-etiquetas text`). Se repite aquí porque el CSS no es importable; si se
# cambia allí hay que cambiarlo aquí, y el test de recorte avisa si el nuevo valor
# no cabe en el `viewBox`.
FUENTE_ROTULOS = 4.8


class TestTaxonomia:
    def test_el_equipo_tiene_las_siete_fases(self) -> None:
        """Las 7 de docs/metricas_finales.md, todas calculables."""
        assert len(fases.FASES_EQUIPO) == 7

    def test_el_jugador_tiene_seis(self) -> None:
        """La 7.a (Perfil/Valor global) no es calculable y no se finge.

        Rellenarla con la posicion seria confundir QUE es un jugador con COMO
        juega, que es lo que el radar compara.
        """
        assert len(fases.FASES_JUGADOR) == 6

    def test_ninguna_fase_de_jugador_es_posicional(self) -> None:
        todas = [m for f in fases.FASES_JUGADOR for m in f.features]
        assert not any(m.startswith("pos_") for m in todas)

    @pytest.mark.parametrize("entidad", ["jugador", "equipo"])
    def test_ninguna_metrica_esta_en_dos_fases(self, entidad: str) -> None:
        """Una métrica en dos fases se contaría dos veces al promediar."""
        todas = [m for f in fases.fases_de(entidad) for m in f.features]
        assert len(todas) == len(set(todas))

    def test_las_claves_de_fase_son_unicas(self) -> None:
        for entidad in ("jugador", "equipo"):
            claves = [f.clave for f in fases.fases_de(entidad)]
            assert len(claves) == len(set(claves))

    def test_todas_las_fases_tienen_metricas(self) -> None:
        """Un vertice sin metricas quedaria siempre hueco: no vale como eje."""
        assert all(f.features for f in fases.FASES_EQUIPO)
        assert all(f.features for f in fases.FASES_JUGADOR)

    def test_una_entidad_desconocida_no_tiene_fases(self) -> None:
        assert fases.fases_de("arbitro") == ()

    def test_fase_por_feature_localiza_la_metrica(self) -> None:
        mapa = fases.fase_por_feature("jugador")
        assert mapa["npxg"].clave == "finalizacion"
        assert "ppda" not in mapa  # es de equipo


class TestPercentilRango:
    def test_el_menor_es_cero_y_el_mayor_uno(self) -> None:
        p = fases.percentil_rango(np.array([5.0, 1.0, 3.0]))
        assert p[1] == 0.0 and p[0] == 1.0 and p[2] == 0.5

    def test_los_empates_comparten_percentil(self) -> None:
        """El orden de indexación no puede decidir quién va delante."""
        p = fases.percentil_rango(np.array([2.0, 2.0, 1.0]))
        assert p[0] == p[1]

    def test_un_unico_valor_va_a_la_mediana(self) -> None:
        """Sin universo con el que comparar, ni 0 ni 1 serían honestos."""
        assert fases.percentil_rango(np.array([7.0]))[0] == 0.5

    def test_los_nan_no_cuentan_ni_contaminan(self) -> None:
        p = fases.percentil_rango(np.array([1.0, np.nan, 3.0]))
        assert np.isnan(p[1])
        assert (p[0], p[2]) == (0.0, 1.0)

    def test_todo_nan_devuelve_todo_nan(self) -> None:
        assert np.isnan(fases.percentil_rango(np.array([np.nan, np.nan]))).all()


class TestConstruirMatriz:
    def _matriz(self, valores: list[list[float]], nombres: list[str]) -> fases.MatrizFases:
        return fases.construir_matriz("jugador", nombres, np.array(valores))

    def test_promedia_las_metricas_de_la_fase(self) -> None:
        m = self._matriz([[2.0, 4.0], [0.0, 0.0]], ["passes", "progressive_passes"])
        assert m.z[0, 0] == 3.0

    def test_invierte_las_metricas_en_las_que_subir_es_empeorar(self) -> None:
        """Sin esto, «le regatean mucho» subiría el vértice de Duelos."""
        m = self._matriz([[2.0], [0.0]], ["dribbled_past"])
        duelos = [f.clave for f in m.fases].index("duelos")
        assert m.z[0, duelos] == -2.0

    def test_una_fase_sin_metricas_queda_marcada_como_no_medible(self) -> None:
        m = self._matriz([[1.0], [0.0]], ["passes"])
        assert m.medibles[0] is True
        assert not any(m.medibles[1:])
        assert np.isnan(m.z[:, 1]).all()

    def test_el_percentil_usa_el_universo_entero(self) -> None:
        m = self._matriz([[3.0], [1.0], [2.0]], ["passes"])
        assert list(m.percentil[:, 0]) == [1.0, 0.0, 0.5]

    def test_el_bloque_de_posicion_no_alimenta_ninguna_fase(self) -> None:
        """Las 25 pos_* no son métricas de rendimiento de ninguna fase."""
        m = self._matriz([[5.0], [0.0]], ["pos_left_wing"])
        assert not any(m.medibles)


class TestPerfil:
    def _matriz(self) -> fases.MatrizFases:
        return fases.construir_matriz(
            "jugador", ["passes"], np.array([[2.0], [0.0], [1.0]])
        )

    def test_devuelve_un_valor_por_fase(self) -> None:
        assert len(fases.perfil(self._matriz(), 0)) == len(fases.FASES_JUGADOR)

    def test_una_fase_no_medible_va_sin_percentil(self) -> None:
        assert fases.perfil(self._matriz(), 0)[1].percentil is None

    def test_el_porcentaje_de_una_fase_sin_dato_no_finge_un_cero(self) -> None:
        assert fases.perfil(self._matriz(), 0)[1].porcentaje == "—"


class TestRadar:
    N = len(fases.FASES_JUGADOR)

    def _valores(self, percentiles: list[float | None]) -> tuple[fases.ValorFase, ...]:
        return tuple(
            fases.ValorFase(fase=f, percentil=p, z=None)
            for f, p in zip(fases.FASES_JUGADOR, percentiles)
        )

    def test_dibuja_un_punto_por_fase(self) -> None:
        r = fases.radar([("A", "referencia", self._valores([0.5] * self.N))])
        assert len(r.series[0].puntos.split()) == self.N
        assert len(r.etiquetas) == self.N

    def test_el_percentil_uno_llega_al_borde(self) -> None:
        r = fases.radar([("A", "referencia", self._valores([1.0] * self.N))])
        x, y = r.series[0].puntos.split()[0].split(",")
        assert (float(x), float(y)) == (fases.CENTRO, fases.CENTRO - fases.RADIO)

    def test_una_fase_sin_dato_se_ancla_en_el_centro(self) -> None:
        r = fases.radar([("A", "referencia", self._valores([None] + [0.5] * (self.N - 1)))])
        assert r.series[0].puntos.split()[0] == f"{fases.CENTRO:.2f},{fases.CENTRO:.2f}"
        assert r.sin_dato == ("Progresión",)

    def test_una_fase_medida_por_alguna_serie_no_es_sin_dato(self) -> None:
        r = fases.radar([
            ("A", "referencia", self._valores([None] + [0.5] * (self.N - 1))),
            ("B", "candidato", self._valores([0.9] + [0.5] * (self.N - 1))),
        ])
        assert r.sin_dato == ()

    def test_dos_series_comparten_ejes(self) -> None:
        r = fases.radar([
            ("A", "referencia", self._valores([0.2] * self.N)),
            ("B", "candidato", self._valores([0.8] * self.N)),
        ])
        assert [s.clase for s in r.series] == ["referencia", "candidato"]
        assert len(r.rejilla) == len(fases.FRACCIONES_REJILLA)

    def test_series_con_fases_distintas_se_rechazan(self) -> None:
        """Comparar sobre ejes distintos daría una figura bonita y sin sentido."""
        equipo = tuple(
            fases.ValorFase(fase=f, percentil=0.5, z=None) for f in fases.FASES_EQUIPO
        )
        with pytest.raises(ValueError, match="mismas fases"):
            fases.radar([("A", "referencia", self._valores([0.5] * self.N)),
                         ("B", "candidato", equipo)])

    def test_sin_series_el_radar_esta_vacio(self) -> None:
        assert fases.radar([("A", "referencia", ())]).vacio

    @pytest.mark.parametrize("entidad", ["jugador", "equipo"])
    def test_ningun_rotulo_se_sale_del_viewbox(self, entidad: str) -> None:
        """Lo que se sale del viewBox lo RECORTA el navegador y no se lee.

        Pasa con los ejes laterales, que son los que llevan los rótulos largos
        («Finalización», «Juego aéreo»): el texto arranca en el vértice y crece
        hacia fuera. La anchura se estima con un factor por carácter deliberadamente
        generoso, porque el objetivo es avisar antes de que el recorte aparezca,
        no medir la fuente al píxel.
        """
        ancho_caracter = 0.52 * FUENTE_ROTULOS
        izquierda = -fases.MARGEN_X
        derecha = 100.0 + fases.MARGEN_X
        for i, fase in enumerate(fases.fases_de(entidad)):
            x, _ = fases.vertice(i, len(fases.fases_de(entidad)), fases.RADIO_ETIQUETA)
            ancho = len(fase.etiqueta) * ancho_caracter
            dx = x - fases.CENTRO
            if dx > 1.0:      # anclado por el principio: crece hacia la derecha
                inicio, fin = x, x + ancho
            elif dx < -1.0:   # anclado por el final: crece hacia la izquierda
                inicio, fin = x - ancho, x
            else:             # centrado (ejes de arriba y abajo)
                inicio, fin = x - ancho / 2.0, x + ancho / 2.0
            assert izquierda <= inicio and fin <= derecha, (
                f"«{fase.etiqueta}» ({entidad}) ocupa {inicio:.1f}..{fin:.1f}, "
                f"fuera del viewBox {izquierda:.0f}..{derecha:.0f}. Sube "
                f"fases.MARGEN_X o baja RADIO_ETIQUETA / el cuerpo de letra."
            )

    def test_el_radar_declara_su_viewbox(self) -> None:
        r = fases.radar([("A", "referencia", self._valores([0.5] * self.N))])
        assert r.view_box == fases.VIEW_BOX

    def test_el_viewbox_deja_hueco_a_los_lados(self) -> None:
        """El área de dibujo es 100x100; el resto es sitio para los rótulos."""
        x, y, ancho, alto = (float(v) for v in fases.VIEW_BOX.split())
        assert (x, y) == (-fases.MARGEN_X, -fases.MARGEN_Y)
        assert ancho == 100.0 + 2 * fases.MARGEN_X
        assert alto == 100.0 + 2 * fases.MARGEN_Y

    def test_las_etiquetas_no_invaden_la_figura(self) -> None:
        """El texto de la derecha empieza en el vértice; el de la izquierda acaba."""
        r = fases.radar([("A", "referencia", self._valores([0.5] * self.N))])
        assert r.etiquetas[0].anclaje == "middle"          # vértice superior
        assert r.etiquetas[1].anclaje == "start"           # lado derecho
        assert r.etiquetas[-1].anclaje == "end"            # lado izquierdo

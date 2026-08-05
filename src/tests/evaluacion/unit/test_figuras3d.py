"""Superficies 3D: lectura del barrido, rejilla Z y geometria que va a las figuras.

`figuras3d` no construye ni evalua nada: lee una carpeta de barrido ya generada.
Lo que puede fallar aqui es de LECTURA (etiquetar una casilla con los ejes
equivocados, dar por medido un punto interpolado, no avisar de un optimo en el
borde), asi que eso es lo que se prueba. El dibujo en si se ejercita en
`integracion/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluacion import figuras3d, malla, puntuacion, registro
from src.tests.evaluacion.conftest import escribir_barrido, fila_metrica


class TestLeerValores:
    def test_lee_la_tabla_larga_acumulada(self, barrido_2x2: Path) -> None:
        df = figuras3d.leer_valores(barrido_2x2)
        assert set(df["combinacion"]) == {"v01", "v02", "v03", "v04"}

    def test_la_formulacion_se_lee_como_texto(self, barrido_2x2: Path) -> None:
        """`read_csv` la devolveria como entero y dejaria de casar con la rejilla."""
        assert figuras3d.leer_valores(barrido_2x2)["formulacion"].tolist()[0] == "2"

    def test_sin_barrido_previo_el_error_dice_que_hay_que_lanzarlo(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit, match="src.evaluacion.barrido"):
            figuras3d.leer_valores(tmp_path)


class TestCombinacionesDelBarrido:
    def test_se_leen_del_registro_de_la_carpeta(self, barrido_2x2: Path) -> None:
        """No se recalculan desde `barrido.HIPERPARAMETROS`: esa lista se EDITA
        entre corridas y etiquetaria las figuras con los ejes de hoy.
        """
        df = figuras3d.leer_valores(barrido_2x2)
        combos = figuras3d.combinaciones_del_barrido(barrido_2x2, df)
        assert combos["v02"]["F2_L1"] == 0.5

    def test_van_en_orden_natural(self, barrido_2x2: Path) -> None:
        df = figuras3d.leer_valores(barrido_2x2)
        combos = figuras3d.combinaciones_del_barrido(barrido_2x2, df)
        assert list(combos) == ["v01", "v02", "v03", "v04"]

    def test_sin_registro_se_recurre_al_resumen_markdown(self, tmp_path: Path) -> None:
        """Es lo que tienen los barridos anteriores al registro."""
        d = escribir_barrido(
            tmp_path / "b",
            [fila_metrica("v01", "F2_equipo_global", "top1", 0.5)],
            {"v01": {"F2_L1": 0.5}})
        registro.ruta(d).unlink()
        (d / "resumen_barrido.md").write_text(
            "| combinacion | F2_L1 |\n|---|---|\n| v01 | `0.5` |\n", encoding="utf-8")
        df = figuras3d.leer_valores(d)
        assert figuras3d.combinaciones_del_barrido(d, df) == {"v01": {"F2_L1": 0.5}}

    def test_sin_ninguna_de_las_dos_se_recomponen_avisando(
        self, tmp_path: Path, capsys
    ) -> None:
        """Nombra por POSICION, que es justo lo que el registro dejo de hacer: el
        aviso es parte del contrato.
        """
        d = escribir_barrido(
            tmp_path / "b",
            [fila_metrica("v01", "F2_equipo_global", "top1", 0.5)],
            {"v01": {}})
        registro.ruta(d).unlink()
        df = figuras3d.leer_valores(d)
        figuras3d.combinaciones_del_barrido(d, df)
        assert "se recomponen los hiperparametros" in capsys.readouterr().out

    def test_un_csv_que_no_declara_ninguna_celda_conocida_aborta(
        self, tmp_path: Path
    ) -> None:
        """Sin registro ni resumen se recomponen desde `barrido`, pero eso exige
        saber que formulaciones y entidades hay en la rejilla.
        """
        d = escribir_barrido(tmp_path / "b", [{
            "combinacion": "v01", "modelo": "F9_arbitro_global",
            "formulacion": "9", "entidad": "arbitro", "normalizacion": "global",
            "fase": "1", "metrica": "top1", "valor": 0.5}], {"v01": {}})
        registro.ruta(d).unlink()
        with pytest.raises(SystemExit, match="ninguna formulacion/entidad"):
            figuras3d.combinaciones_del_barrido(d, figuras3d.leer_valores(d))

    def test_un_csv_de_otra_carpeta_aborta(self, barrido_2x2: Path) -> None:
        """Si el indice no describe una combinacion del CSV, los dos ficheros no
        son de la misma carpeta y las etiquetas de los ejes serian falsas.
        """
        df = figuras3d.leer_valores(barrido_2x2)
        df.loc[0, "combinacion"] = "v99"
        with pytest.raises(SystemExit, match="no son de la misma carpeta"):
            figuras3d.combinaciones_del_barrido(barrido_2x2, df)


class TestEjes:
    def test_solo_son_ejes_los_hiperparametros_que_se_movieron(
        self, barrido_2x2: Path
    ) -> None:
        df = figuras3d.leer_valores(barrido_2x2)
        combos = figuras3d.combinaciones_del_barrido(barrido_2x2, df)
        assert set(figuras3d.ejes_variables(combos)) == {"F2_L1", "F2_BETA"}

    def test_los_demas_se_declaran_como_fijos(self, barrido_2x2: Path) -> None:
        """Van al pie de la figura: sin ellos, la superficie afirma menos de lo
        que parece (no se sabe con que EASE se midio).
        """
        df = figuras3d.leer_valores(barrido_2x2)
        combos = figuras3d.combinaciones_del_barrido(barrido_2x2, df)
        variables = set(figuras3d.ejes_variables(combos))
        assert figuras3d._fijos(combos, variables) == {"F5_RFF_DIM": 512}

    def test_sin_combinaciones_no_hay_fijos(self) -> None:
        assert figuras3d._fijos({}, set()) == {}


class TestConstruirRejilla:
    COMBOS = {
        "v01": {"x": 1, "y": 10}, "v02": {"x": 2, "y": 10},
        "v03": {"x": 1, "y": 20}, "v04": {"x": 2, "y": 20},
    }

    def test_coloca_cada_combinacion_en_su_casilla(self) -> None:
        Z = figuras3d.construir_rejilla(
            {"v01": 0.1, "v02": 0.2, "v03": 0.3, "v04": 0.4},
            self.COMBOS, "x", [1, 2], "y", [10, 20], "max", "max")
        assert Z.tolist() == [[0.1, 0.2], [0.3, 0.4]]

    def test_una_casilla_sin_combinacion_queda_a_nan(self) -> None:
        """La superficie simplemente no la dibuja: no se extrapola."""
        Z = figuras3d.construir_rejilla(
            {"v01": 0.1}, self.COMBOS, "x", [1, 2], "y", [10, 20], "max", "max")
        assert np.isnan(Z[1, 1])

    def test_los_valores_no_finitos_no_ocupan_casilla(self) -> None:
        Z = figuras3d.construir_rejilla(
            {"v01": float("nan")}, self.COMBOS, "x", [1, 2], "y", [10, 20],
            "max", "max")
        assert not np.any(np.isfinite(Z))

    def test_con_mas_de_dos_ejes_reduce_por_el_mejor_valor(self) -> None:
        """Es una PROYECCION, no un corte: la casilla enseña lo mejor alcanzable
        ahi, no una combinacion concreta.
        """
        combos = {"a": {"x": 1, "y": 10}, "b": {"x": 1, "y": 10}}
        Z = figuras3d.construir_rejilla({"a": 0.1, "b": 0.9}, combos,
                                        "x", [1], "y", [10], "max", "max")
        assert Z[0, 0] == pytest.approx(0.9)

    def test_para_una_metrica_de_menor_es_mejor_reduce_por_el_minimo(self) -> None:
        combos = {"a": {"x": 1, "y": 10}, "b": {"x": 1, "y": 10}}
        Z = figuras3d.construir_rejilla({"a": 0.1, "b": 0.9}, combos,
                                        "x", [1], "y", [10], "max", "min")
        assert Z[0, 0] == pytest.approx(0.1)

    def test_puede_reducirse_por_la_media(self) -> None:
        combos = {"a": {"x": 1, "y": 10}, "b": {"x": 1, "y": 10}}
        Z = figuras3d.construir_rejilla({"a": 0.1, "b": 0.9}, combos,
                                        "x", [1], "y", [10], "media", "max")
        assert Z[0, 0] == pytest.approx(0.5)

    def test_una_combinacion_que_no_declara_los_ejes_se_ignora(self) -> None:
        """Pasa con una carpeta acumulada cuyo registro venga de un barrido con
        otros ejes: colocarla donde no esta seria peor que omitirla.
        """
        combos = {"v01": {"x": 1, "y": 10}, "v09": {"otro": 5}}
        Z = figuras3d.construir_rejilla({"v01": 0.5, "v09": 0.9}, combos,
                                        "x", [1], "y", [10], "max", "max")
        assert Z[0, 0] == pytest.approx(0.5)


class TestOptimo:
    def test_encuentra_el_maximo(self) -> None:
        Z = np.array([[0.1, 0.9], [0.3, 0.2]])
        assert figuras3d._optimo(Z, "max") == (0, 1)

    def test_encuentra_el_minimo_cuando_menor_es_mejor(self) -> None:
        Z = np.array([[0.1, 0.9], [0.3, 0.2]])
        assert figuras3d._optimo(Z, "min") == (0, 0)

    def test_ignora_las_casillas_sin_medir(self) -> None:
        Z = np.array([[np.nan, 0.5], [0.2, np.nan]])
        assert figuras3d._optimo(Z, "max") == (0, 1)

    def test_una_rejilla_vacia_no_tiene_optimo(self) -> None:
        assert figuras3d._optimo(np.full((2, 2), np.nan), "max") is None

    def test_un_optimo_en_el_borde_es_el_limite_del_rango_probado(self) -> None:
        """La lectura mas util de la figura, y la que las tablas del resumen no dan."""
        Z = np.zeros((3, 3))
        assert figuras3d.en_borde(Z, (0, 1))
        assert figuras3d.en_borde(Z, (1, 2))
        assert not figuras3d.en_borde(Z, (1, 1))

    def test_con_un_solo_valor_en_un_eje_no_hay_borde_que_declarar(self) -> None:
        """Con dos casillas todo es borde: avisar siempre no informaria de nada."""
        assert not figuras3d.en_borde(np.zeros((2, 2)), (0, 0))


class TestGeometriaParaElVisor:
    def test_los_nan_viajan_como_null(self) -> None:
        """JSON no tiene NaN: dejarlo como texto romperia el visor."""
        assert figuras3d._z_json(np.array([[1.0, np.nan]])) == [[1.0, None]]

    def test_los_valores_se_redondean_conservando_las_cifras_utiles(self) -> None:
        """`%g` mantiene la precision tambien en las metricas muy pequeñas, cosa
        que redondear a un numero fijo de decimales no haria.
        """
        assert figuras3d._z_json(np.array([[1.23456789e-7]]))[0][0] == pytest.approx(
            1.23457e-7)

    def test_la_geometria_declara_los_dos_ejes_y_su_escala(self) -> None:
        ex = malla.eje("F2_L1", [0.25, 0.5])
        ey = malla.eje("F5_RFF_DIM", [512, 1024, 2048])
        geo = figuras3d._geometria_json(ex, ey)
        assert geo["ejeX"] == "F2_L1" and geo["ejeY"] == "F5_RFF_DIM"
        assert geo["escalaY"] == "log"

    def test_las_coordenadas_van_normalizadas_al_cubo_del_visor(self) -> None:
        """Conservando el reparto REAL: es lo que hace que el HTML y el PNG
        enseñen la misma superficie.
        """
        ex = malla.eje("x", [0.0, 1.0, 10.0])
        geo = figuras3d._geometria_json(ex, ex)
        assert geo["coordX"][0] == pytest.approx(-1.0)
        assert geo["coordX"][-1] == pytest.approx(1.0)
        assert geo["coordX"][1] == pytest.approx(-0.8)

    def test_marca_que_nodos_de_la_malla_fina_son_datos(self) -> None:
        ex = malla.eje("x", [1.0, 2.0, 3.0])
        geo = figuras3d._geometria_json(ex, ex)
        assert sum(geo["medidaXf"]) == 3
        assert len(geo["medidaXf"]) == len(geo["coordXf"])

    def test_las_etiquetas_son_los_valores_reales(self) -> None:
        ex = malla.eje("d", [512, 1024, 2048])
        assert figuras3d._geometria_json(ex, ex)["valoresX"] == ["512", "1024", "2048"]


class TestPieDeFigura:
    EX = malla.eje("F2_L1", [0.25, 0.5, 1.0])
    EY = malla.eje("F2_BETA", [1.0, 2.0, 3.0])

    def test_declara_la_escala_de_cada_eje(self) -> None:
        pie = figuras3d._pie_de_figura(self.EX, self.EY, [], "max", "max", {})
        assert "Ejes a escala" in pie
        assert "F2_L1" in pie and "F2_BETA" in pie

    def test_avisa_de_que_la_superficie_es_una_estimacion(self) -> None:
        """Sin este aviso, la figura se lee como si todos los puntos estuvieran
        medidos.
        """
        pie = figuras3d._pie_de_figura(self.EX, self.EY, [], "max", "max", {})
        assert "ESTIMACION" in pie
        assert "solo esos puntos son datos" in pie

    def test_sin_interpolar_no_se_anuncia_estimacion(self) -> None:
        ex = malla.eje("x", [1.0, 2.0], densidad=0)
        pie = figuras3d._pie_de_figura(ex, ex, [], "max", "max", {})
        assert "ESTIMACION" not in pie

    def test_declara_la_proyeccion_sobre_los_ejes_restantes(self) -> None:
        pie = figuras3d._pie_de_figura(self.EX, self.EY, ["F5_RFF_DIM"],
                                       "max", "max", {})
        assert "Proyeccion" in pie
        assert "no es un corte a valor fijo" in pie

    def test_la_reduccion_por_media_se_dice_como_tal(self) -> None:
        pie = figuras3d._pie_de_figura(self.EX, self.EY, ["otro"], "media", "max", {})
        assert "media sobre" in pie

    def test_declara_los_ejes_fijos(self) -> None:
        pie = figuras3d._pie_de_figura(self.EX, self.EY, [], "max", "max",
                                       {"F5_RFF_DIM": 512})
        assert "Fijos: F5_RFF_DIM=512" in pie


class TestUtilidades:
    def test_la_etiqueta_compacta_junta_los_tres_ejes(self) -> None:
        assert figuras3d._etiqueta_modelo("2", "jugador", "global") == "F2_jugador_global"

    def test_el_titulo_del_eje_declara_la_escala_si_no_es_lineal(self) -> None:
        """Un eje logaritmico cambia lo que significa la distancia entre ticks."""
        assert figuras3d._titulo_eje(malla.eje("x", [1.0, 2.0, 3.0])) == "x"
        assert "(log)" in figuras3d._titulo_eje(malla.eje("d", [10, 100, 1000]))

    def test_el_parseo_de_listas_admite_espacios(self) -> None:
        assert figuras3d._lista("top1, mrr ") == {"top1", "mrr"}

    def test_sin_filtro_devuelve_none(self) -> None:
        assert figuras3d._lista(None) is None
        assert figuras3d._lista("") is None

    def test_la_orientacion_es_la_misma_que_usa_el_resumen(self) -> None:
        """Las tablas y las figuras no pueden discrepar sobre el sentido de una
        metrica: las dos salen de `puntuacion.orientacion`.
        """
        assert figuras3d.orientacion is puntuacion.orientacion


class TestGenerar:
    def test_sin_dos_ejes_variables_no_hay_superficie(self, tmp_path: Path) -> None:
        d = escribir_barrido(
            tmp_path / "b",
            [fila_metrica("v01", "F2_equipo_global", "top1", 0.5),
             fila_metrica("v02", "F2_equipo_global", "top1", 0.7)],
            {"v01": {"F2_L1": 0.25}, "v02": {"F2_L1": 0.5}})
        with pytest.raises(SystemExit, match="necesita DOS ejes"):
            figuras3d.generar(d, tmp_path / "out")

    def test_un_filtro_que_no_deja_nada_aborta(self, barrido_2x2: Path) -> None:
        with pytest.raises(SystemExit, match="no deja ninguna serie"):
            figuras3d.generar(barrido_2x2, barrido_2x2 / "out",
                              metricas_sel={"inexistente"})

    def test_sin_metricas_puntuables_no_hay_score_pero_si_figuras(
        self, tmp_path: Path, capsys
    ) -> None:
        """El score agrega solo las metricas con peso declarado; que no haya
        ninguna no puede impedir dibujar las que si trae el barrido.
        """
        combis = {"v01": {"a": 1, "b": 1}, "v02": {"a": 2, "b": 1},
                  "v03": {"a": 1, "b": 2}, "v04": {"a": 2, "b": 2}}
        filas = [fila_metrica(n, "F2_equipo_global", "n_pureza", float(i))
                 for i, n in enumerate(combis, start=1)]
        d = escribir_barrido(tmp_path / "b", filas, combis)
        figuras3d.generar(d, d / "f", con_html=False)
        assert "sin score compuesto" in capsys.readouterr().out
        assert not (d / "f" / "score.csv").exists()
        assert list((d / "f").glob("*.png"))

    def test_con_un_solo_modelo_no_hay_comparativa_que_dibujar(
        self, tmp_path: Path
    ) -> None:
        """Superponer una unica superficie no compara nada."""
        combis = {"v01": {"a": 1, "b": 1}, "v02": {"a": 2, "b": 1},
                  "v03": {"a": 1, "b": 2}, "v04": {"a": 2, "b": 2}}
        filas = [fila_metrica(n, "F2_equipo_global", "top1", float(i))
                 for i, n in enumerate(combis, start=1)]
        d = escribir_barrido(tmp_path / "b", filas, combis)
        figuras3d.generar(d, d / "f", con_html=False)
        assert not list((d / "f").glob("comparativa_*.png"))

    def test_avisa_si_la_rejilla_es_demasiado_pobre(
        self, tmp_path: Path, capsys
    ) -> None:
        d = escribir_barrido(
            tmp_path / "b",
            [fila_metrica(c, "F2_equipo_global", "top1", 0.5)
             for c in ("v01", "v02", "v03")],
            {"v01": {"a": 1, "b": 1}, "v02": {"a": 2, "b": 1}, "v03": {"a": 1, "b": 2}})
        figuras3d.generar(d, d / "out", con_html=False, con_score=False)
        assert "la superficie sera" in capsys.readouterr().out

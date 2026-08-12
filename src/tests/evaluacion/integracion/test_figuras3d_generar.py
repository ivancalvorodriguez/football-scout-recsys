"""Dibujo completo de las superficies de un barrido ya generado.

`generar` encadena lectura del registro, rejilla, geometria (`malla`), score
(`puntuacion`), matplotlib y el visor HTML. Aqui se comprueba que ese encadenado
produce los entregables que documenta `docs/evaluacion.md` y que los CSV recogen
SOLO lo medido: la superficie interpolada es una ayuda de lectura, no un
resultado que deba viajar a un analisis externo como si se hubiera evaluado.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluacion import figuras3d, puntuacion
from src.tests.evaluacion.conftest import escribir_barrido, fila_metrica

pytestmark = [pytest.mark.integracion, pytest.mark.lento]


@pytest.fixture
def salida(barrido_2x2: Path) -> Path:
    out = barrido_2x2 / "figuras3d"
    figuras3d.generar(barrido_2x2, out)
    return out


class TestEntregables:
    def test_dibuja_un_png_por_metrica_y_modelo(self, salida: Path) -> None:
        for metrica in ("top1", "asimetria"):
            for modelo in ("F2_equipo_global", "F5_equipo_global"):
                assert (salida / f"{modelo}__{metrica}__F2_L1_x_F2_BETA.png").exists()

    def test_escribe_la_rejilla_medida_en_csv(self, salida: Path) -> None:
        assert (salida / "rejilla_3d.csv").exists()

    def test_el_csv_solo_trae_los_puntos_evaluados(self, salida: Path) -> None:
        """Cuatro casillas por (metrica, modelo): la malla fina (~30x30) no sale."""
        df = pd.read_csv(salida / "rejilla_3d.csv")
        por_serie = df.groupby(["metrica", "modelo"]).size()
        assert set(por_serie) == {4}

    def test_el_csv_declara_la_escala_de_cada_eje(self, salida: Path) -> None:
        df = pd.read_csv(salida / "rejilla_3d.csv")
        assert set(df["escala_x"]) == {"lineal"}
        assert {"eje_x", "eje_y", "valor_x", "valor_y", "z"} <= set(df.columns)

    def test_sin_ejes_sobrantes_no_se_anota_reduccion(self, salida: Path) -> None:
        """Anotar "max" ahi haria pensar que la celda es un maximo sobre algo,
        cuando es el valor de una unica combinacion.
        """
        df = pd.read_csv(salida / "rejilla_3d.csv")
        assert df["reduccion"].isna().all()

    def test_escribe_el_visor_interactivo(self, salida: Path) -> None:
        html = (salida / "superficies.html").read_text(encoding="utf-8")
        assert "F2_L1" in html
        assert "http" not in html

    def test_devuelve_cuantas_figuras_escribio(self, barrido_2x2: Path) -> None:
        n = figuras3d.generar(barrido_2x2, barrido_2x2 / "f")
        assert n == len(list((barrido_2x2 / "f").glob("*.png")))


class TestScoreCompuesto:
    def test_publica_las_z_que_lo_componen(self, salida: Path) -> None:
        """El numero se usa para ordenar modelos: tiene que ser auditable."""
        df = pd.read_csv(salida / "score.csv")
        assert puntuacion.NOMBRE in df.columns
        assert "z_top1" in df.columns

    def test_viaja_como_una_metrica_mas(self, salida: Path) -> None:
        df = pd.read_csv(salida / "rejilla_3d.csv")
        assert puntuacion.NOMBRE in set(df["metrica"])

    def test_dibuja_la_comparativa_entre_modelos_de_la_misma_entidad(
        self, salida: Path
    ) -> None:
        """La unica figura que superpone modelos, porque el score es lo unico
        comparable entre ellos (z-scoreado dentro de la entidad).
        """
        assert (salida /
                f"comparativa_equipo__{puntuacion.NOMBRE}__F2_L1_x_F2_BETA.png").exists()

    def test_imprime_el_ranking_por_entidad(self, barrido_2x2: Path, capsys) -> None:
        figuras3d.generar(barrido_2x2, barrido_2x2 / "f")
        salida = capsys.readouterr().out
        assert "Score compuesto" in salida
        assert "equipo:" in salida

    def test_se_puede_desactivar(self, barrido_2x2: Path) -> None:
        out = barrido_2x2 / "sin_score"
        figuras3d.generar(barrido_2x2, out, con_score=False)
        assert not (out / "score.csv").exists()
        assert puntuacion.NOMBRE not in set(
            pd.read_csv(out / "rejilla_3d.csv")["metrica"])

    def test_se_calcula_sobre_todas_las_metricas_y_no_sobre_el_filtro(
        self, barrido_2x2: Path
    ) -> None:
        """Si dependiera de `--metricas`, seria un numero distinto en cada ejecucion
        y no serviria para ordenar nada.
        """
        completo = barrido_2x2 / "completo"
        filtrado = barrido_2x2 / "filtrado"
        figuras3d.generar(barrido_2x2, completo, con_html=False)
        figuras3d.generar(barrido_2x2, filtrado, metricas_sel={"top1"},
                          con_html=False)
        a = pd.read_csv(completo / "score.csv").set_index(["modelo", "combinacion"])
        b = pd.read_csv(filtrado / "score.csv").set_index(["modelo", "combinacion"])
        assert a[puntuacion.NOMBRE].equals(b[puntuacion.NOMBRE])


class TestFiltrosYAvisos:
    def test_el_filtro_de_metricas_acota_los_entregables(
        self, barrido_2x2: Path
    ) -> None:
        out = barrido_2x2 / "solo_top1"
        figuras3d.generar(barrido_2x2, out, metricas_sel={"top1"}, con_score=False)
        assert not list(out.glob("*asimetria*.png"))
        assert list(out.glob("*top1*.png"))

    def test_el_filtro_de_modelos_acota_los_entregables(
        self, barrido_2x2: Path
    ) -> None:
        out = barrido_2x2 / "solo_f5"
        figuras3d.generar(barrido_2x2, out, modelos_sel={"F5_equipo_global"},
                          con_score=False)
        assert not list(out.glob("F2_*.png"))

    def test_se_puede_desactivar_el_visor(self, barrido_2x2: Path) -> None:
        out = barrido_2x2 / "sin_html"
        figuras3d.generar(barrido_2x2, out, con_html=False)
        assert not (out / "superficies.html").exists()

    def test_avisa_cuando_el_optimo_cae_en_el_borde(
        self, barrido_2x2: Path, capsys
    ) -> None:
        """Un maximo pegado al borde no es un maximo: es el limite del rango
        probado. Con dos valores por eje todo es borde, asi que la rejilla se
        amplia a tres para que el aviso signifique algo.
        """
        combis = {f"v{i:02d}": {"F2_L1": l1, "F2_BETA": b}
                  for i, (l1, b) in enumerate(
                      [(a, c) for a in (0.25, 0.5, 0.75) for c in (1.0, 2.0, 3.0)],
                      start=1)}
        # El maximo cae en la esquina (0.75, 3.0): borde en los dos ejes.
        filas = [fila_metrica(n, "F5_equipo_global", "top1",
                              c["F2_L1"] + c["F2_BETA"])
                 for n, c in combis.items()]
        d = escribir_barrido(barrido_2x2.parent / "borde", filas, combis)
        figuras3d.generar(d, d / "f", con_html=False, con_score=False)
        assert "optimo en el BORDE" in capsys.readouterr().out

    def test_una_serie_sin_ningun_valor_finito_se_omite_avisando(
        self, barrido_2x2: Path, capsys
    ) -> None:
        """La pureza posicional del equipo es el caso real: no es calculable, y
        dibujar una superficie vacia seria peor que decirlo.
        """
        df = pd.read_csv(barrido_2x2 / "barrido_metricas.csv")
        vacias = df[df["metrica"] == "top1"].copy()
        vacias["metrica"] = "pureza_top1"
        vacias["valor"] = float("nan")
        pd.concat([df, vacias]).to_csv(
            barrido_2x2 / "barrido_metricas.csv", index=False)
        figuras3d.generar(barrido_2x2, barrido_2x2 / "f", con_html=False,
                          con_score=False)
        assert "sin ningun valor finito" in capsys.readouterr().out


class TestDensidadYEscala:
    def test_la_densidad_no_cambia_ningun_numero_medido(
        self, barrido_2x2: Path
    ) -> None:
        """Solo controla cuanto se nota el facetado del dibujo."""
        fina = barrido_2x2 / "fina"
        cruda = barrido_2x2 / "cruda"
        figuras3d.generar(barrido_2x2, fina, densidad=32, con_html=False)
        figuras3d.generar(barrido_2x2, cruda, densidad=0, con_html=False)
        a = pd.read_csv(fina / "rejilla_3d.csv").sort_values(
            ["metrica", "modelo", "valor_x", "valor_y"]).reset_index(drop=True)
        b = pd.read_csv(cruda / "rejilla_3d.csv").sort_values(
            ["metrica", "modelo", "valor_x", "valor_y"]).reset_index(drop=True)
        assert a["z"].equals(b["z"])

    def test_la_escala_ordinal_recupera_el_reparto_uniforme(
        self, barrido_2x2: Path
    ) -> None:
        out = barrido_2x2 / "ordinal"
        figuras3d.generar(barrido_2x2, out, escala="ordinal", con_html=False)
        assert set(pd.read_csv(out / "rejilla_3d.csv")["escala_x"]) == {"ordinal"}

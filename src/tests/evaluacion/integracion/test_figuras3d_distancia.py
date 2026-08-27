"""La distancia se comporta como la normalizacion en las figuras, no como un eje.

Es la diferencia entre un HIPERPARAMETRO y un eje del MODELO, y aqui se fija:

- un hiperparametro (`F5_EASE_LAMBDA`) es un eje de la superficie: se dibuja en X
  o en Y y la metrica se lee como una funcion continua de el;
- la distancia NO. Cada geometria es otro modelo, con su propia superficie sobre
  sus propios hiperparametros — igual que `por_liga` y `global`. Ponerla como eje
  daria una «superficie» sobre cuatro casillas categoricas, que no es una
  superficie: son cuatro puntos.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from src.evaluacion import figuras3d
from src.tests.evaluacion.conftest import escribir_barrido, fila_metrica

pytestmark = pytest.mark.integracion


@pytest.fixture
def barrido_dos_distancias(tmp_path: Path) -> Path:
    """Mismo modelo y mismos ejes, medido con dos geometrias.

    Dos hiperparametros x dos valores (el minimo que da una superficie) por cada
    una de las dos distancias.
    """
    combis = {
        f"v{i:02d}": {"F5_EASE_LAMBDA": lam, "F5_RFF_DIM": dim,
                      "F5_METODO_EQUIPO": "mmd"}
        for i, (lam, dim) in enumerate(
            itertools.product((10.0, 50.0), (512, 1024)), start=1)
    }
    filas = []
    for i, combi in enumerate(combis):
        for j, distancia in enumerate(("euclidea", "manhattan")):
            filas.append(fila_metrica(combi, "F5_equipo_global", "top1",
                                      0.1 + 0.05 * i + 0.2 * j,
                                      distancia=distancia))
    return escribir_barrido(tmp_path / "barrido", filas, combis)


@pytest.fixture
def figuras(barrido_dos_distancias: Path) -> Path:
    out = barrido_dos_distancias / "figuras3d"
    figuras3d.generar(barrido_dos_distancias, out, con_score=False,
                      con_html=False, densidad=0)
    return out


class TestUnaFiguraPorGeometria:
    def test_cada_distancia_tiene_su_propio_png(self, figuras: Path) -> None:
        nombres = {p.name for p in figuras.glob("*.png")}
        assert "F5_equipo_global_euclidea__top1__F5_EASE_LAMBDA_x_F5_RFF_DIM.png" \
            in nombres
        assert "F5_equipo_global_manhattan__top1__F5_EASE_LAMBDA_x_F5_RFF_DIM.png" \
            in nombres

    def test_las_dos_se_dibujan_sobre_los_MISMOS_ejes_de_hiperparametro(
        self, figuras: Path
    ) -> None:
        """Lo que cambia entre las dos figuras es la geometria, no los ejes: por
        eso son comparables una contra otra."""
        ejes = {p.name.split("__")[2] for p in figuras.glob("*.png")}
        assert ejes == {"F5_EASE_LAMBDA_x_F5_RFF_DIM.png"}

    def test_la_distancia_no_es_un_eje_de_ninguna_superficie(
        self, figuras: Path
    ) -> None:
        """Ni en el nombre del fichero ni, por tanto, en los ejes dibujados."""
        assert not list(figuras.glob("*distancia*"))

    def test_la_rejilla_medida_separa_las_dos_geometrias(self, figuras: Path) -> None:
        import pandas as pd

        rejilla = pd.read_csv(figuras / "rejilla_3d.csv")
        assert set(rejilla["modelo"]) == {"F5_equipo_global_euclidea",
                                          "F5_equipo_global_manhattan"}
        # Y ninguna casilla mezcla las dos: cada modelo trae su rejilla entera.
        for _modelo, bloque in rejilla.groupby("modelo"):
            assert len(bloque) == 4          # 2 valores x 2 valores


class TestCarpetaAntigua:
    def test_un_barrido_sin_la_columna_se_dibuja_como_euclideo(
        self, tmp_path: Path
    ) -> None:
        """Las carpetas ya acumuladas no la tienen: se midieron con la euclidea,
        que era la unica. Deben seguir dibujandose, declarandolo."""
        import pandas as pd

        combis = {
            f"v{i:02d}": {"F5_EASE_LAMBDA": lam, "F5_RFF_DIM": dim}
            for i, (lam, dim) in enumerate(
                itertools.product((10.0, 50.0), (512, 1024)), start=1)
        }
        filas = [fila_metrica(c, "F5_equipo_global", "top1", 0.1 + 0.05 * i)
                 for i, c in enumerate(combis)]
        # Se quita la columna, como en una carpeta anterior a este eje.
        barrido_dir = escribir_barrido(tmp_path / "viejo", filas, combis)
        csv = barrido_dir / "barrido_metricas.csv"
        df = pd.read_csv(csv)
        df["modelo"] = "F5_equipo_global"
        df.drop(columns=["distancia"]).to_csv(csv, index=False)

        out = barrido_dir / "figuras3d"
        figuras3d.generar(barrido_dir, out, con_score=False, con_html=False,
                          densidad=0)
        assert list(out.glob("F5_equipo_global_euclidea__*.png"))

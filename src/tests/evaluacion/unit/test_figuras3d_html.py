"""Visor 3D autocontenido: un unico HTML sin servidor ni dependencias.

La razon de que exista es que el PNG deja la camara fija y en 3D el angulo decide
lo que se ve. Su contrato practico es "doble clic y funciona, aqui o en otra
maquina": nada de red, nada de ficheros hermanos, y los datos ya resueltos desde
Python (el visor no reimplementa la escala de los ejes ni la interpolacion).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from src.evaluacion import figuras3d, figuras3d_html, malla


def _vista() -> dict:
    ex = malla.eje("F2_L1", [0.25, 0.5])
    ey = malla.eje("F2_BETA", [1.0, 2.0])
    Z = np.array([[0.1, 0.2], [0.3, np.nan]])
    return {
        "metrica": "top1", "modelo": "F2_equipo_global",
        "ejes": "F2_L1 x F2_BETA", "mejor": "max", "pie": "pie de figura",
        **figuras3d._geometria_json(ex, ey),
        "superficies": [{"modelo": "F2_equipo_global",
                         "Z": figuras3d._z_json(Z),
                         "Zf": figuras3d._z_json(malla.interpolar(Z, ex, ey))}],
    }


def _datos(html: str) -> dict:
    """Recupera el JSON embebido para poder afirmar sobre lo que ve el visor."""
    m = re.search(r"const DATOS\s*=\s*(\{.*?\});", html, re.S)
    assert m, "el HTML no lleva los datos embebidos"
    return json.loads(m.group(1))


class TestEscribir:
    def test_escribe_el_fichero_y_devuelve_su_ruta(self, tmp_path: Path) -> None:
        destino = tmp_path / "superficies.html"
        assert figuras3d_html.escribir([_vista()], destino, "Titulo") == destino
        assert destino.is_file()

    def test_es_una_pagina_html_completa(self, tmp_path: Path) -> None:
        html = figuras3d_html.escribir(
            [_vista()], tmp_path / "s.html", "Titulo").read_text(encoding="utf-8")
        assert html.lstrip().startswith("<!doctype html>")
        assert "</html>" in html

    def test_lleva_el_titulo_pedido(self, tmp_path: Path) -> None:
        html = figuras3d_html.escribir(
            [_vista()], tmp_path / "s.html", "Barrido X").read_text(encoding="utf-8")
        assert "Barrido X" in html

    def test_no_pide_nada_a_la_red(self, tmp_path: Path) -> None:
        """Se puede mover a otra maquina o adjuntar a la memoria tal cual; por eso
        el renderer 3D esta escrito a mano sobre canvas 2D.
        """
        html = figuras3d_html.escribir(
            [_vista()], tmp_path / "s.html", "T").read_text(encoding="utf-8")
        assert "http://" not in html
        assert "https://" not in html

    def test_embebe_las_vistas_con_su_geometria(self, tmp_path: Path) -> None:
        html = figuras3d_html.escribir(
            [_vista()], tmp_path / "s.html", "T").read_text(encoding="utf-8")
        datos = _datos(html)
        assert datos["titulo"] == "T"
        vista = datos["vistas"][0]
        assert vista["metrica"] == "top1"
        assert vista["ejeX"] == "F2_L1"
        assert len(vista["coordXf"]) == len(vista["medidaXf"])

    def test_las_casillas_sin_medir_viajan_como_null(self, tmp_path: Path) -> None:
        """JSON no tiene NaN: un NaN literal dejaria el visor sin poder arrancar."""
        html = figuras3d_html.escribir(
            [_vista()], tmp_path / "s.html", "T").read_text(encoding="utf-8")
        assert "NaN" not in html
        assert _datos(html)["vistas"][0]["superficies"][0]["Z"][1][1] is None

    def test_lleva_la_superficie_medida_y_la_estimada(self, tmp_path: Path) -> None:
        """El HTML y los PNG tienen que enseñar exactamente lo mismo: la geometria
        llega ya resuelta desde Python.
        """
        html = figuras3d_html.escribir(
            [_vista()], tmp_path / "s.html", "T").read_text(encoding="utf-8")
        superficie = _datos(html)["vistas"][0]["superficies"][0]
        assert len(superficie["Z"]) == 2                      # rejilla medida
        assert len(superficie["Zf"]) > len(superficie["Z"])   # malla fina

    def test_admite_varias_vistas(self, tmp_path: Path) -> None:
        v1, v2 = _vista(), {**_vista(), "metrica": "mrr"}
        html = figuras3d_html.escribir(
            [v1, v2], tmp_path / "s.html", "T").read_text(encoding="utf-8")
        assert {v["metrica"] for v in _datos(html)["vistas"]} == {"top1", "mrr"}

    def test_avisa_en_la_propia_pagina_si_el_visor_no_puede_dibujar(
        self, tmp_path: Path
    ) -> None:
        """Los mismos numeros estan en los PNG y en rejilla_3d.csv: la pagina lo
        dice en vez de quedarse en blanco.
        """
        html = figuras3d_html.escribir(
            [_vista()], tmp_path / "s.html", "T").read_text(encoding="utf-8")
        assert "rejilla_3d.csv" in html

    def test_se_escribe_en_utf8(self, tmp_path: Path) -> None:
        vista = {**_vista(), "pie": "proyeccion sobre los demas ejes — señales"}
        destino = figuras3d_html.escribir([vista], tmp_path / "s.html", "T")
        assert "señales" in destino.read_text(encoding="utf-8")

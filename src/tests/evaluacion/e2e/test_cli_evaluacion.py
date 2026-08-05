"""Extremo a extremo: `build` -> `evaluar`, y `barrido` -> `figuras3d`.

Reproduce los dos flujos documentados en CLAUDE.md tal cual, en subprocesos. Es lo
unico que comprueba que los puntos de entrada encajan entre si: que `evaluar` lee
lo que `build` escribe con los mismos nombres de artefacto, y que `figuras3d`
entiende la carpeta que deja `barrido`.

El barrido se lanza con una lista `HIPERPARAMETROS` reducida (inyectada en el
subproceso), porque la del repo son cientos de combinaciones y cada una construye
su propia rejilla: aqui interesa el encadenado, no el tamaño de la rejilla.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.tests.conftest import RAIZ_REPO, ejecutar_modulo

pytestmark = [pytest.mark.e2e, pytest.mark.lento]

# Solo la F5 de equipo: 6 entidades y 16 observaciones, segundos por ajuste. Con
# la F2 de jugador el mismo flujo tardaria minutos sin cubrir nada nuevo.
EJES = [("F5_EASE_LAMBDA", [10.0, 50.0]), ("F5_RFF_DIM", [512, 1024])]


def ejecutar_barrido(*args: str, ejes: list = EJES) -> subprocess.CompletedProcess:
    """Lanza `barrido.main` en un subproceso con la rejilla de ejes reducida.

    No hay flag que elija los hiperparametros (a proposito: viven en el modulo),
    asi que la unica forma de acotar la rejilla desde fuera es inyectar la lista.
    """
    codigo = (
        "import sys\n"
        "from src.evaluacion import barrido\n"
        f"barrido.HIPERPARAMETROS = {ejes!r}\n"
        "barrido.main(sys.argv[1:])\n"
    )
    return subprocess.run(
        [sys.executable, "-c", codigo, *args],
        cwd=str(RAIZ_REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=600,
    )


@pytest.fixture(scope="module")
def modelos(bd_sintetica: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Rejilla de equipo construida por CLI, tal como la dejaria un usuario."""
    destino = tmp_path_factory.mktemp("modelos_eval_e2e")
    res = ejecutar_modulo(
        "src.similitud.build",
        "--db", str(bd_sintetica), "--out", str(destino),
        "--entidad", "equipo", "--formulacion", "ambas", "--normalizacion", "ambas",
    )
    assert res.returncode == 0, res.stderr
    return {"dir": destino}


@pytest.fixture(scope="module")
def evaluacion(modelos: dict, bd_sintetica: Path,
               tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    salida = tmp_path_factory.mktemp("evaluacion_e2e")
    res = ejecutar_modulo(
        "src.evaluacion.evaluar",
        "--db", str(bd_sintetica), "--model-dir", str(modelos["dir"]),
        "--out", str(salida), "--fases", "0,1,5", "--bootstrap", "0",
    )
    return {"dir": salida, "res": res}


class TestEvaluar:
    def test_termina_bien(self, evaluacion: dict) -> None:
        assert evaluacion["res"].returncode == 0, evaluacion["res"].stderr

    def test_escribe_el_informe_como_entregable_principal(
        self, evaluacion: dict
    ) -> None:
        texto = (evaluacion["dir"] / "resumen.md").read_text(encoding="utf-8")
        assert "Fase 1 - Auto-similitud" in texto
        assert "Fase 6" in texto

    def test_escribe_un_csv_por_fase_ejecutada(self, evaluacion: dict) -> None:
        for nombre in ("fase0_sanity", "fase1_autosimilitud", "fase5_downstream"):
            assert (evaluacion["dir"] / f"{nombre}.csv").exists()

    def test_no_escribe_los_csv_de_las_fases_omitidas(self, evaluacion: dict) -> None:
        """Un CSV vacio se leeria como "la fase corrio y no dio nada"."""
        assert not (evaluacion["dir"] / "fase2_estabilidad.csv").exists()
        assert not (evaluacion["dir"] / "fase3_triangulacion.csv").exists()

    def test_dibuja_las_figuras_clave(self, evaluacion: dict) -> None:
        assert (evaluacion["dir"] / "figuras" /
                "fase1_autosimilitud_top1.png").exists()

    def test_informa_del_progreso_y_del_tiempo(self, evaluacion: dict) -> None:
        """La evaluacion completa tarda: sin ETA no se sabe si quedan segundos o
        una hora.
        """
        salida = evaluacion["res"].stdout
        assert "pasos cronometrados" in salida
        assert "Listo en" in salida

    def test_avisa_de_los_modelos_de_jugador_que_faltan(self, evaluacion: dict) -> None:
        assert "falta el modelo" in evaluacion["res"].stdout

    def test_sin_ningun_modelo_aborta_diciendolo(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        res = ejecutar_modulo(
            "src.evaluacion.evaluar",
            "--db", str(bd_sintetica), "--model-dir", str(tmp_path),
            "--out", str(tmp_path / "out"), "--fases", "0",
        )
        assert res.returncode != 0
        assert "No hay modelos en" in res.stderr

    def test_la_ayuda_documenta_las_opciones(self) -> None:
        res = ejecutar_modulo("src.evaluacion.evaluar", "--help")
        assert res.returncode == 0
        for opcion in ("--db", "--model-dir", "--out", "--fases", "--bootstrap",
                       "--sin-figuras"):
            assert opcion in res.stdout

    def test_sin_figuras_no_dibuja(self, modelos: dict, bd_sintetica: Path,
                                   tmp_path: Path) -> None:
        res = ejecutar_modulo(
            "src.evaluacion.evaluar",
            "--db", str(bd_sintetica), "--model-dir", str(modelos["dir"]),
            "--out", str(tmp_path), "--fases", "0", "--bootstrap", "0",
            "--sin-figuras",
        )
        assert res.returncode == 0, res.stderr
        assert not (tmp_path / "figuras").exists()


@pytest.fixture(scope="module")
def barrido(bd_sintetica: Path,
            tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    salida = tmp_path_factory.mktemp("barrido_e2e")
    res = ejecutar_barrido(
        "--db", str(bd_sintetica), "--out", str(salida),
        "--formulaciones", "5", "--entidades", "equipo",
        "--fases", "0,1", "--bootstrap", "0", "--sin-figuras",
    )
    return {"dir": salida, "res": res}


class TestBarrido:
    def test_construye_y_evalua_sin_build_previo(self, barrido: dict) -> None:
        """Es la diferencia con `evaluar`: se puede lanzar de cero."""
        assert barrido["res"].returncode == 0, barrido["res"].stderr
        assert "[construir] rejilla de modelos" in barrido["res"].stdout

    def test_escribe_los_tres_entregables_de_la_carpeta(self, barrido: dict) -> None:
        for nombre in ("resumen_barrido.md", "barrido_metricas.csv",
                       "combinaciones.json"):
            assert (barrido["dir"] / nombre).exists()

    def test_deja_una_subcarpeta_por_combinacion(self, barrido: dict) -> None:
        assert sorted(p.name for p in barrido["dir"].glob("v*")) == [
            "v01", "v02", "v03", "v04"]

    def test_cada_combinacion_lleva_sus_csv_y_sus_modelos(self, barrido: dict) -> None:
        v01 = barrido["dir"] / "v01"
        assert (v01 / "fase0_sanity.csv").exists()
        assert list((v01 / "modelo").glob("formulacion5_equipo_*.npz"))

    def test_imprime_la_correspondencia_nombre_configuracion(
        self, barrido: dict
    ) -> None:
        """El `vNN` identifica una configuracion: hay que poder leer cual."""
        assert "F5_EASE_LAMBDA=" in barrido["res"].stdout
        assert "[nueva]" in barrido["res"].stdout

    def test_dice_de_donde_sale_cada_artefacto(self, barrido: dict) -> None:
        """Los dos ejes barridos entran en el alcance de la celda F5, asi que las
        cuatro combinaciones tienen huellas distintas y se construyen todas: el
        log dice ademas por que no se reutilizo nada.
        """
        salida = barrido["res"].stdout
        assert "[construye]" in salida
        assert "no hay ningun artefacto con esta huella" in salida

    def test_remite_a_las_superficies_3d(self, barrido: dict) -> None:
        assert "src.evaluacion.figuras3d" in barrido["res"].stdout

    def test_relanzarlo_no_reconstruye_ni_pierde_combinaciones(
        self, barrido: dict, bd_sintetica: Path
    ) -> None:
        """La carpeta es acumulativa: es lo que permite barrer en tandas."""
        res = ejecutar_barrido(
            "--db", str(bd_sintetica), "--out", str(barrido["dir"]),
            "--formulaciones", "5", "--entidades", "equipo",
            "--fases", "0", "--bootstrap", "0", "--sin-figuras",
        )
        assert res.returncode == 0, res.stderr
        assert "0 nuevas" in res.stdout
        assert "[construye]" not in res.stdout

    def test_una_formulacion_desconocida_aborta(self, bd_sintetica: Path,
                                                tmp_path: Path) -> None:
        res = ejecutar_barrido("--db", str(bd_sintetica), "--out", str(tmp_path),
                               "--formulaciones", "9")
        assert res.returncode != 0
        assert "formulaciones desconocidas" in res.stderr

    def test_una_entidad_desconocida_aborta(self, bd_sintetica: Path,
                                            tmp_path: Path) -> None:
        res = ejecutar_barrido("--db", str(bd_sintetica), "--out", str(tmp_path),
                               "--entidades", "arbitro")
        assert res.returncode != 0
        assert "entidades desconocidas" in res.stderr

    def test_un_hiperparametro_inexistente_aborta_antes_de_construir(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        """Antes de gastar horas: un nombre que no existe no sobrescribe nada y el
        barrido acabaria comparando combinaciones identicas en silencio.
        """
        res = ejecutar_barrido("--db", str(bd_sintetica), "--out", str(tmp_path),
                               ejes=[("F5_NO_EXISTE", [1.0])])
        assert res.returncode != 0
        assert "hiperparametro desconocido" in res.stderr

    def test_la_ayuda_documenta_las_opciones(self) -> None:
        res = ejecutar_modulo("src.evaluacion.barrido", "--help")
        assert res.returncode == 0
        for opcion in ("--formulaciones", "--entidades", "--fases", "--rehacer",
                       "--adoptar-existentes"):
            assert opcion in res.stdout


class TestFiguras3d:
    def test_dibuja_las_superficies_del_barrido(self, barrido: dict) -> None:
        res = ejecutar_modulo("src.evaluacion.figuras3d",
                              "--barrido", str(barrido["dir"]))
        assert res.returncode == 0, res.stderr
        figuras = barrido["dir"] / "figuras3d"
        assert list(figuras.glob("*.png"))
        assert (figuras / "rejilla_3d.csv").exists()
        assert (figuras / "score.csv").exists()
        assert (figuras / "superficies.html").exists()

    def test_declara_los_ejes_y_su_escala(self, barrido: dict) -> None:
        res = ejecutar_modulo("src.evaluacion.figuras3d",
                              "--barrido", str(barrido["dir"]), "--sin-html")
        assert "Ejes barridos:" in res.stdout
        assert "Escala de los ejes:" in res.stdout

    def test_sin_un_barrido_previo_el_error_lo_dice(self, tmp_path: Path) -> None:
        res = ejecutar_modulo("src.evaluacion.figuras3d", "--barrido", str(tmp_path))
        assert res.returncode != 0
        assert "src.evaluacion.barrido" in res.stderr

    def test_la_ayuda_documenta_las_opciones(self) -> None:
        res = ejecutar_modulo("src.evaluacion.figuras3d", "--help")
        assert res.returncode == 0
        for opcion in ("--barrido", "--metricas", "--modelos", "--escala",
                       "--densidad", "--sin-score", "--sin-html"):
            assert opcion in res.stdout

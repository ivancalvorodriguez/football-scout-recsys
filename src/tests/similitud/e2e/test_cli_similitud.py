"""Extremo a extremo del modelo: `build` -> `probar` -> `comparar`.

Reproduce el flujo documentado en CLAUDE.md tal cual, en subprocesos: es lo unico
que comprueba que los tres puntos de entrada encajan entre si (que `probar` lee
lo que `build` escribe, con los mismos nombres de artefacto).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.tests.conftest import ejecutar_modulo

pytestmark = [pytest.mark.e2e, pytest.mark.lento]


@pytest.fixture(scope="module")
def modelos(bd_sintetica: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Construye de verdad, por CLI, las dos formulaciones para equipo.

    Se usa `equipo` (6 entidades, 16 observaciones): la formulacion 2 resuelve un
    elastic-net por observacion, asi que sobre jugadores el e2e seria lento sin
    aportar cobertura nueva.
    """
    destino = tmp_path_factory.mktemp("modelos_e2e")
    res = ejecutar_modulo(
        "src.similitud.build",
        "--db", str(bd_sintetica), "--out", str(destino),
        "--entidad", "equipo", "--formulacion", "ambas", "--normalizacion", "ambas",
    )
    return {"dir": destino, "res": res}


class TestBuild:
    def test_termina_bien(self, modelos: dict) -> None:
        assert modelos["res"].returncode == 0, modelos["res"].stderr

    def test_genera_las_cuatro_combinaciones(self, modelos: dict) -> None:
        """2 formulaciones x 2 normalizaciones, cada una con .npz y .json."""
        for form in ("2", "5"):
            for modo in ("por_liga", "global"):
                assert (modelos["dir"] / f"formulacion{form}_equipo_{modo}.npz").exists()
                assert (modelos["dir"] / f"formulacion{form}_equipo_{modo}.json").exists()

    def test_informa_del_tamano_del_problema(self, modelos: dict) -> None:
        salida = modelos["res"].stdout
        assert "observaciones" in salida
        assert "entidades" in salida
        assert "guardado en" in salida

    def test_la_ayuda_documenta_las_opciones(self) -> None:
        res = ejecutar_modulo("src.similitud.build", "--help")
        assert res.returncode == 0
        for opcion in ("--db", "--out", "--formulacion", "--entidad", "--normalizacion"):
            assert opcion in res.stdout


class TestProbar:
    def _nombre(self, resumen_bd: dict) -> str:
        return resumen_bd["equipos"][0][1]

    @pytest.mark.parametrize("formulacion", ["2", "5"])
    def test_consulta_el_top_k(self, modelos: dict, resumen_bd: dict, formulacion: str) -> None:
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--modelo", str(modelos["dir"]), "--entidad", "equipo",
            "--formulacion", formulacion, "--nombre", self._nombre(resumen_bd), "--k", "3",
        )
        assert res.returncode == 0, res.stderr
        assert f"Formulacion {formulacion}" in res.stdout
        assert "Referencia:" in res.stdout
        assert "Top-3 mas similares" in res.stdout

    def test_muestra_los_scores_y_las_metricas_que_explican(
        self, modelos: dict, resumen_bd: dict
    ) -> None:
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--modelo", str(modelos["dir"]), "--entidad", "equipo",
            "--formulacion", "5", "--nombre", self._nombre(resumen_bd), "--k", "3",
        )
        assert "score=" in res.stdout
        assert "coinciden en:" in res.stdout

    def test_admite_nombre_parcial(self, modelos: dict, resumen_bd: dict) -> None:
        completo = self._nombre(resumen_bd)          # "Equipo 100"
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--modelo", str(modelos["dir"]), "--entidad", "equipo",
            "--formulacion", "5", "--nombre", completo.split()[-1], "--k", "2",
        )
        assert res.returncode == 0, res.stderr
        assert completo in res.stdout

    def test_elige_el_modelo_segun_la_normalizacion(
        self, modelos: dict, resumen_bd: dict
    ) -> None:
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--modelo", str(modelos["dir"]), "--entidad", "equipo",
            "--formulacion", "5", "--normalizacion", "global",
            "--nombre", self._nombre(resumen_bd),
        )
        assert res.returncode == 0, res.stderr
        assert ("=== Formulacion 5 | equipo | global | distancia euclidea ==="
                in res.stdout)

    def test_elige_el_modelo_segun_la_distancia(
        self, modelos: dict, resumen_bd: dict
    ) -> None:
        """Sin `--distancia` no habria forma de consultar desde la terminal un
        modelo construido con otra geometria: viven en ficheros con sufijo."""
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--modelo", str(modelos["dir"]), "--entidad", "equipo",
            "--formulacion", "5", "--normalizacion", "global",
            "--distancia", "coseno", "--nombre", self._nombre(resumen_bd),
        )
        # El artefacto coseno no se construye en esta fixture: lo que se fija es
        # que la flag llega hasta la carga y que el error dice QUE fichero falta.
        assert "coseno" in (res.stdout + res.stderr)

    def test_un_nombre_inexistente_aborta_con_mensaje(self, modelos: dict) -> None:
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--modelo", str(modelos["dir"]), "--entidad", "equipo",
            "--formulacion", "5", "--nombre", "Equipo Que No Existe",
        )
        assert res.returncode != 0
        assert "No se encontro" in res.stderr

    def test_sin_modelo_construido_el_error_lo_dice(self, tmp_path: Path) -> None:
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--modelo", str(tmp_path), "--entidad", "jugador",
            "--formulacion", "2", "--nombre", "Messi",
        )
        assert res.returncode != 0
        assert "No se encontro modelo" in res.stderr


class TestComparar:
    def test_compara_las_normalizaciones_y_escribe_las_salidas(
        self, modelos: dict, resumen_bd: dict, tmp_path: Path
    ) -> None:
        nombres = ",".join(n for _, n in resumen_bd["equipos"][:2])
        res = ejecutar_modulo(
            "src.similitud.comparar",
            "--entidad", "equipo", "--formulacion", "5", "--equipos", nombres,
            "--k", "3", "--modelo-dir", str(modelos["dir"]), "--out-dir", str(tmp_path),
        )
        assert res.returncode == 0, res.stderr
        assert "Jaccard=" in res.stdout
        assert (tmp_path / "comparativa_form5_equipo.csv").exists()
        assert (tmp_path / "heatmap_form5_equipo.png").exists()
        assert (tmp_path / "jaccard_form5_equipo.png").exists()

    def test_exige_la_lista_de_referencias(self, modelos: dict, tmp_path: Path) -> None:
        res = ejecutar_modulo(
            "src.similitud.comparar",
            "--entidad", "equipo", "--formulacion", "5",
            "--modelo-dir", str(modelos["dir"]), "--out-dir", str(tmp_path),
        )
        assert res.returncode != 0
        assert "--equipos" in res.stderr


class TestFlujoCompleto:
    def test_extraer_construir_y_consultar(self, tmp_path: Path) -> None:
        """El camino real de punta a punta: eventos StatsBomb -> BD -> modelo ->
        recomendacion, sin tocar nada a mano por el medio.
        """
        from src.tests import factories as fac

        # 1) Un dataset con varios partidos para que haya con que ajustar.
        partidos, eventos, alineaciones = [], {}, {}
        for i in range(4):
            mid = 8000 + i
            m, ev, lu = fac.partido_minimo(mid)
            m["match_week"], m["match_date"] = i + 1, f"2020-01-{i + 1:02d}"
            partidos.append(m)
            eventos[mid], alineaciones[mid] = ev, lu
        raiz = fac.escribir_open_data(
            tmp_path / "open-data" / "data",
            competiciones=[fac.fila_competicion(
                1, 100, competition_name="Liga E2E", season_name="2019/2020")],
            partidos={(1, 100): partidos}, eventos=eventos, alineaciones=alineaciones,
        )
        bd = tmp_path / "scouting.db"
        modelos = tmp_path / "modelo"

        # 2) Extraer.
        res = ejecutar_modulo(
            "src.extraccion.extract",
            "--competition", "Liga E2E", "--season", "2019/2020", "--pct", "100",
            "--db", str(bd), "--data-root", str(raiz),
        )
        assert res.returncode == 0, res.stderr

        # 3) Construir el modelo sobre lo extraido.
        res = ejecutar_modulo(
            "src.similitud.build",
            "--db", str(bd), "--out", str(modelos),
            "--entidad", "jugador", "--formulacion", "5", "--normalizacion", "por_liga",
        )
        assert res.returncode == 0, res.stderr

        # 4) Consultar una recomendacion.
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--modelo", str(modelos), "--entidad", "jugador",
            "--formulacion", "5", "--nombre", "Jugador 10", "--k", "2",
        )
        assert res.returncode == 0, res.stderr
        assert "Referencia: Jugador 10" in res.stdout
        assert "score=" in res.stdout

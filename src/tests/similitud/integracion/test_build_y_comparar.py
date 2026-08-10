"""Integracion de los orquestadores: `build` y `comparar`.

`build` se prueba con la formulacion 5 (mucho mas barata que la 2, que resuelve
un elastic-net por observacion); la 2 ya se cubre en `test_formulaciones.py` y en
el e2e.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.similitud import build, comparar
from src.similitud.modelo import cargar_modelo

pytestmark = [pytest.mark.integracion, pytest.mark.lento]


class TestBuild:
    def test_construye_y_guarda_un_modelo(
        self, bd_sintetica: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        build._construir_uno(bd_sintetica, tmp_path, "5", "jugador", "por_liga")
        assert (tmp_path / "formulacion5_jugador_por_liga.npz").exists()
        assert (tmp_path / "formulacion5_jugador_por_liga.json").exists()

    def test_informa_del_tamano_del_problema(
        self, bd_sintetica: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        build._construir_uno(bd_sintetica, tmp_path, "5", "jugador", "por_liga")
        salida = capsys.readouterr().out
        assert "observaciones" in salida
        assert "entidades" in salida
        assert "guardado en" in salida

    def test_el_modelo_guardado_se_puede_volver_a_cargar(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        build._construir_uno(bd_sintetica, tmp_path, "5", "equipo", "global")
        modelo = cargar_modelo(tmp_path, "5", "equipo", normalizacion="global")
        assert modelo.entidad == "equipo"
        assert modelo.meta["normalizacion"] == "global"

    def test_main_recorre_todas_las_combinaciones_pedidas(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        build.main([
            "--db", str(bd_sintetica), "--out", str(tmp_path),
            "--formulacion", "5", "--entidad", "equipo", "--normalizacion", "ambas",
        ])
        assert (tmp_path / "formulacion5_equipo_por_liga.npz").exists()
        assert (tmp_path / "formulacion5_equipo_global.npz").exists()

    def test_main_admite_una_sola_combinacion(
        self, bd_sintetica: Path, tmp_path: Path
    ) -> None:
        build.main([
            "--db", str(bd_sintetica), "--out", str(tmp_path),
            "--formulacion", "5", "--entidad", "equipo", "--normalizacion", "global",
        ])
        # Solo esa combinacion; el `.warm.npz` que la acompana es el estado
        # reutilizable del mismo modelo, no otra combinacion (ver `similitud.warm`).
        assert sorted(p.name for p in tmp_path.glob("*.npz")) == [
            "formulacion5_equipo_global.npz",
            "formulacion5_equipo_global.warm.npz",
        ]

    def test_no_toca_la_base_de_datos(self, bd_sintetica: Path, tmp_path: Path) -> None:
        """`build` es de solo lectura sobre la BD del extractor."""
        antes = bd_sintetica.read_bytes()
        build._construir_uno(bd_sintetica, tmp_path, "5", "equipo", "por_liga")
        assert bd_sintetica.read_bytes() == antes


@pytest.fixture(scope="module")
def modelos_equipo(bd_sintetica: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Los dos modos de normalizacion para equipo, ya entrenados."""
    destino = tmp_path_factory.mktemp("modelos")
    for modo in ("por_liga", "global"):
        build._construir_uno(bd_sintetica, destino, "5", "equipo", modo)
    return destino


class TestComparar:
    def _referencias(self, modelos_equipo: Path) -> list[str]:
        modelo = cargar_modelo(modelos_equipo, "5", "equipo", normalizacion="por_liga")
        return modelo.entity_names[:2]

    def test_genera_csv_y_figuras(self, modelos_equipo: Path, tmp_path: Path) -> None:
        refs = self._referencias(modelos_equipo)
        comparar._comparar_para(
            entidad="equipo", formulacion="5",
            normalizaciones=["por_liga", "global"], referencias=refs, k=3,
            model_dir=modelos_equipo, out_dir=tmp_path,
        )
        assert (tmp_path / "comparativa_form5_equipo.csv").exists()
        assert (tmp_path / "heatmap_form5_equipo.png").exists()
        assert (tmp_path / "jaccard_form5_equipo.png").exists()
        assert len(list(tmp_path.glob("ranking_form5_equipo_*.png"))) == len(refs)

    def test_el_csv_lleva_los_rankings_paralelos(
        self, modelos_equipo: Path, tmp_path: Path
    ) -> None:
        import pandas as pd

        refs = self._referencias(modelos_equipo)
        comparar._comparar_para(
            entidad="equipo", formulacion="5",
            normalizaciones=["por_liga", "global"], referencias=refs, k=3,
            model_dir=modelos_equipo, out_dir=tmp_path,
        )
        df = pd.read_csv(tmp_path / "comparativa_form5_equipo.csv")
        assert set(df.columns) == {"referencia", "modo", "rank", "candidato", "score"}
        assert set(df["modo"]) == {"por_liga", "global"}
        assert set(df["referencia"]) == set(refs)

    def test_informa_del_jaccard_por_referencia(
        self, modelos_equipo: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        comparar._comparar_para(
            entidad="equipo", formulacion="5",
            normalizaciones=["por_liga", "global"],
            referencias=self._referencias(modelos_equipo), k=3,
            model_dir=modelos_equipo, out_dir=tmp_path,
        )
        assert "Jaccard=" in capsys.readouterr().out

    def test_una_referencia_desconocida_se_omite_sin_abortar(
        self, modelos_equipo: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A diferencia de `probar` (una sola entidad), aqui saltarse una
        referencia mala y seguir con el resto es lo util.
        """
        refs = [*self._referencias(modelos_equipo), "Equipo Inventado"]
        comparar._comparar_para(
            entidad="equipo", formulacion="5",
            normalizaciones=["por_liga", "global"], referencias=refs, k=3,
            model_dir=modelos_equipo, out_dir=tmp_path,
        )
        assert "referencia omitida" in capsys.readouterr().out
        assert (tmp_path / "comparativa_form5_equipo.csv").exists()

    def test_una_referencia_duplicada_se_omite(
        self, modelos_equipo: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Dos consultas pueden resolver a la misma entidad: procesarla dos veces
        duplicaria su fila en el CSV y su barra en las figuras.
        """
        nombre = self._referencias(modelos_equipo)[0]
        comparar._comparar_para(
            entidad="equipo", formulacion="5",
            normalizaciones=["por_liga", "global"], referencias=[nombre, nombre], k=3,
            model_dir=modelos_equipo, out_dir=tmp_path,
        )
        assert "referencia duplicada omitida" in capsys.readouterr().out

    def test_sin_referencias_validas_no_escribe_nada(
        self, modelos_equipo: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        salida = tmp_path / "vacio"
        comparar._comparar_para(
            entidad="equipo", formulacion="5",
            normalizaciones=["por_liga", "global"], referencias=["No Existe"], k=3,
            model_dir=modelos_equipo, out_dir=salida,
        )
        assert "nada que escribir" in capsys.readouterr().out
        assert not salida.exists()

    def test_hace_falta_mas_de_una_normalizacion(
        self, modelos_equipo: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit, match="al menos 2"):
            comparar._comparar_para(
                entidad="equipo", formulacion="5", normalizaciones=["por_liga"],
                referencias=["X"], k=3, model_dir=modelos_equipo, out_dir=tmp_path,
            )

    def test_una_normalizacion_desconocida_aborta(
        self, modelos_equipo: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit, match="desconocida"):
            comparar._comparar_para(
                entidad="equipo", formulacion="5",
                normalizaciones=["por_liga", "minmax"],
                referencias=["X"], k=3, model_dir=modelos_equipo, out_dir=tmp_path,
            )

    def test_main_exige_jugadores_para_entidad_jugador(self) -> None:
        with pytest.raises(SystemExit, match="--jugadores"):
            comparar.main(["--entidad", "jugador"])

    def test_main_exige_equipos_para_entidad_equipo(self) -> None:
        with pytest.raises(SystemExit, match="--equipos"):
            comparar.main(["--entidad", "equipo"])

    def test_main_cablea_los_argumentos(self, modelos_equipo: Path, tmp_path: Path) -> None:
        nombres = ",".join(self._referencias(modelos_equipo))
        comparar.main([
            "--entidad", "equipo", "--formulacion", "5", "--equipos", nombres,
            "--k", "3", "--modelo-dir", str(modelos_equipo), "--out-dir", str(tmp_path),
        ])
        assert (tmp_path / "comparativa_form5_equipo.csv").exists()

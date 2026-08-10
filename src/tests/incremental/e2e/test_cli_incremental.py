"""Extremo a extremo del flujo incremental: `ingesta` -> `reentrenar` -> `probar`.

Reproduce en subprocesos lo que haria un usuario con partidos nuevos, incluido el
ultimo eslabon: que la consulta de siempre encuentra a un jugador que no existia
antes de ingerir. Es lo unico que comprueba que las tres entradas encajan (que
`reentrenar` lee lo que `ingesta` escribio y produce artefactos que `probar` sabe
abrir).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.tests import factories
from src.tests.conftest import ejecutar_modulo
from src.tests.incremental.conftest import escribir_paquete

pytestmark = [pytest.mark.e2e, pytest.mark.lento]


@pytest.fixture(scope="module")
def flujo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """BD sintetica + modelos en frio + ingesta de un paquete + reentrenamiento.

    Se usa `equipo`: la formulacion 2 resuelve un elastic-net por observacion y
    sobre jugadores el e2e seria lento sin cubrir nada nuevo.
    """
    base = tmp_path_factory.mktemp("incremental_e2e")
    db_path = base / "scouting.db"
    factories.crear_bd_sintetica(db_path)
    modelos = base / "modelo"
    paquete = escribir_paquete(base / "paquete", competition_id=9, season_id=900,
                               competition_name="Liga Nueva")

    construccion = ejecutar_modulo(
        "src.similitud.build",
        "--db", str(db_path), "--out", str(modelos),
        "--entidad", "equipo", "--formulacion", "5", "--normalizacion", "por_liga",
    )
    ingesta = ejecutar_modulo(
        "src.incremental.ingesta",
        "--paquete", str(paquete["raiz"]), "--db", str(db_path), "--sin-copia",
    )
    reentrenamiento = ejecutar_modulo(
        "src.incremental.reentrenar",
        "--db", str(db_path), "--modelos", str(modelos),
        "--entidad", "equipo", "--formulacion", "5", "--normalizacion", "por_liga",
    )
    return {
        "base": base, "db": db_path, "modelos": modelos, "paquete": paquete,
        "construccion": construccion, "ingesta": ingesta,
        "reentrenamiento": reentrenamiento,
    }


class TestIngesta:
    def test_termina_bien(self, flujo: dict) -> None:
        assert flujo["ingesta"].returncode == 0, flujo["ingesta"].stderr

    def test_informa_de_lo_que_ha_entrado(self, flujo: dict) -> None:
        salida = flujo["ingesta"].stdout
        assert "jugadores nuevos: 5" in salida
        assert "equipos nuevos: 2" in salida
        assert "ligas nuevas: 1" in salida

    def test_deja_el_informe_junto_a_la_bd(self, flujo: dict) -> None:
        assert (flujo["db"].parent / "ingesta.json").exists()

    def test_apunta_al_siguiente_paso(self, flujo: dict) -> None:
        assert "src.incremental.reentrenar" in flujo["ingesta"].stdout

    def test_solo_validar_no_escribe_nada(self, tmp_path: Path) -> None:
        paquete = escribir_paquete(tmp_path / "p")
        db_path = tmp_path / "no-deberia-existir.db"
        res = ejecutar_modulo(
            "src.incremental.ingesta", "--paquete", str(paquete["raiz"]),
            "--db", str(db_path), "--solo-validar")
        assert res.returncode == 0, res.stderr
        assert not db_path.exists()
        assert "procesables" in res.stdout

    def test_un_paquete_roto_falla_sin_tocar_la_bd(self, tmp_path: Path) -> None:
        paquete = escribir_paquete(tmp_path / "p")
        (paquete["raiz"] / "events" / f'{paquete["match_id"]}.json').unlink()
        db_path = tmp_path / "scouting.db"
        factories.crear_bd_sintetica(db_path)
        antes = db_path.read_bytes()

        res = ejecutar_modulo(
            "src.incremental.ingesta", "--paquete", str(paquete["raiz"]),
            "--db", str(db_path), "--sin-copia")

        assert res.returncode == 2
        assert "--omitir-invalidos" in res.stdout
        assert db_path.read_bytes() == antes

    def test_la_ayuda_documenta_las_opciones(self) -> None:
        res = ejecutar_modulo("src.incremental.ingesta", "--help")
        assert res.returncode == 0
        for opcion in ("--paquete", "--partidos", "--competicion", "--temporada",
                       "--solo-validar", "--omitir-invalidos", "--sin-copia"):
            assert opcion in res.stdout


class TestReentrenar:
    def test_termina_bien(self, flujo: dict) -> None:
        assert flujo["reentrenamiento"].returncode == 0, flujo["reentrenamiento"].stderr

    def test_informa_de_lo_reutilizado(self, flujo: dict) -> None:
        salida = flujo["reentrenamiento"].stdout
        assert "reutilizadas" in salida
        assert "entidades nuevas" in salida

    def test_deja_el_informe_y_el_estado(self, flujo: dict) -> None:
        assert (flujo["modelos"] / "reentrenamiento.json").exists()
        assert (flujo["modelos"] / "formulacion5_equipo_por_liga.warm.npz").exists()

    def test_sin_bd_falla_con_mensaje(self, tmp_path: Path) -> None:
        res = ejecutar_modulo("src.incremental.reentrenar",
                              "--db", str(tmp_path / "nada.db"))
        assert res.returncode == 2
        assert "No existe la BD" in res.stderr

    def test_la_ayuda_documenta_las_opciones(self) -> None:
        res = ejecutar_modulo("src.incremental.reentrenar", "--help")
        assert res.returncode == 0
        for opcion in ("--db", "--modelos", "--out", "--frio", "--verificar",
                       "--refrescar-kernel"):
            assert opcion in res.stdout


class TestConsultaDespues:
    """El objetivo de todo el flujo: que lo nuevo se pueda recomendar."""

    def test_probar_encuentra_un_equipo_recien_llegado(self, flujo: dict) -> None:
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--nombre", factories.EQUIPO_A["name"], "--entidad", "equipo",
            "--formulacion", "5", "--normalizacion", "por_liga",
            "--modelo", str(flujo["modelos"]),
        )
        assert res.returncode == 0, res.stderr
        assert factories.EQUIPO_A["name"] in res.stdout

    def test_un_equipo_antiguo_recibe_candidatos(self, flujo: dict) -> None:
        res = ejecutar_modulo(
            "src.similitud.probar",
            "--nombre", "Equipo 100", "--entidad", "equipo",
            "--formulacion", "5", "--normalizacion", "por_liga",
            "--modelo", str(flujo["modelos"]), "--k", "5",
        )

        assert res.returncode == 0, res.stderr
        assert "Equipo 100" in res.stdout

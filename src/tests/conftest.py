"""Fixtures compartidas por toda la suite."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.tests import factories

# Raiz del repo: `src/tests/conftest.py` -> subir 3 niveles.
RAIZ_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def raiz_repo() -> Path:
    return RAIZ_REPO


@pytest.fixture(scope="session")
def bd_sintetica(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """BD SQLite con dos ligas, poblada una sola vez por sesion.

    De solo lectura para todo lo que la consume (`src.similitud` nunca escribe en
    la BD), asi que compartirla entre pruebas es seguro y ahorra repoblarla.
    """
    ruta = tmp_path_factory.mktemp("bd") / "scouting.db"
    factories.crear_bd_sintetica(ruta)
    return ruta


@pytest.fixture(scope="session")
def resumen_bd(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Como `bd_sintetica` pero devuelve tambien el resumen de lo creado."""
    ruta = tmp_path_factory.mktemp("bd_resumen") / "scouting.db"
    return factories.crear_bd_sintetica(ruta)


@pytest.fixture
def open_data_minimo(tmp_path: Path) -> dict[str, Any]:
    """Arbol `open-data/data` sintetico con una competicion y un partido."""
    match, eventos, alineaciones = factories.partido_minimo()
    raiz = factories.escribir_open_data(
        tmp_path / "open-data" / "data",
        competiciones=[factories.fila_competicion(1, 100)],
        partidos={(1, 100): [match]},
        eventos={match["match_id"]: eventos},
        alineaciones={match["match_id"]: alineaciones},
    )
    return {
        "data_root": raiz,
        "match": match,
        "eventos": eventos,
        "alineaciones": alineaciones,
    }


def ejecutar_modulo(modulo: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Lanza `python -m <modulo> ...` desde la raiz del repo, como un usuario.

    Se ejecuta con `cwd` en la raiz para que `-m` ponga el repo en `sys.path` y
    para que las rutas relativas por defecto (`open-data/data`, `outputs/`)
    signifiquen lo mismo que en uso real.

    Se fija `PYTHONIOENCODING=utf-8` en el hijo: sin consola real, Python hereda
    la codificacion de la locale (cp1252 en este Windows) y los nombres con
    caracteres fuera de latin-1 harian fallar el print. Los scripts de
    `similitud` ya se defienden con `configurar_consola()`; esto iguala el trato
    para `extract`, que no lo hace, y hace la captura determinista.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, "-m", modulo, *args],
        cwd=str(cwd or RAIZ_REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=600,
    )

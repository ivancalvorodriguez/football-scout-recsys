"""Extremo a extremo de la sección «Datos»: incorporar y reentrenar por HTTP.

Es el único sitio donde el circuito entero corre de verdad: el navegador envía el
formulario, la app lanza `python -m src.incremental...` como subproceso, el
trabajo escribe en la base de datos y en los modelos, y la propia app —sin
reiniciarse— empieza a servir lo que acaba de entrar.

Sobre copias en `tmp_path`, no sobre los fixtures compartidos: aquí sí se
escribe.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import pytest

from src.app.catalogo import ClaveModelo
from src.app.config import (
    FICHERO_CONJUNTO,
    FICHERO_VARIANTE,
    SUBDIR_CONJUNTOS,
    SUBDIR_VARIANTES,
)
from src.tests.conftest import RAIZ_REPO
from src.tests.incremental.conftest import escribir_paquete

pytestmark = [pytest.mark.e2e, pytest.mark.lento]

ARRANQUE_MAX_S = 30.0
TAREA_MAX_S = 180.0


def _puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pedir(url: str, datos: dict[str, str] | None = None,
           timeout: float = 30.0) -> tuple[int, str]:
    cuerpo = urllib.parse.urlencode(datos).encode() if datos is not None else None
    try:
        with urllib.request.urlopen(url, data=cuerpo, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


@pytest.fixture(scope="module")
def entorno(tmp_path_factory: pytest.TempPathFactory, dir_modelos: Path,
            bd: Path) -> dict[str, Any]:
    """Copias escribibles de la BD y de los modelos, más un paquete de un partido."""
    base = tmp_path_factory.mktemp("datos_e2e")
    copia_bd = base / "scouting.db"
    shutil.copy2(bd, copia_bd)
    copia_modelos = base / "modelo"
    shutil.copytree(dir_modelos, copia_modelos)
    paquete = escribir_paquete(base / "paquete", competition_id=99, season_id=990,
                               competition_name="Liga Recien Llegada")
    return {"bd": copia_bd, "modelos": copia_modelos, "paquete": paquete}


@pytest.fixture(scope="module")
def servidor(entorno: dict) -> Iterator[str]:
    puerto = _puerto_libre()
    proceso = subprocess.Popen(
        [sys.executable, "-m", "src.app",
         "--modelo", str(entorno["modelos"]), "--bd", str(entorno["bd"]),
         "--host", "127.0.0.1", "--puerto", str(puerto)],
        cwd=str(RAIZ_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    base = f"http://127.0.0.1:{puerto}"
    limite = time.monotonic() + ARRANQUE_MAX_S
    try:
        while time.monotonic() < limite:
            if proceso.poll() is not None:
                pytest.fail(f"el servidor murio al arrancar:\n{proceso.communicate()[0]}")
            try:
                _pedir(base + "/datos/", timeout=1.0)
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail(f"el servidor no respondio en {ARRANQUE_MAX_S:.0f}s")
        yield base
    finally:
        proceso.terminate()
        try:
            proceso.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proceso.kill()


def esperar_tarea(base: str) -> dict[str, Any]:
    """Consulta el estado hasta que la tarea deje de estar en curso."""
    limite = time.monotonic() + TAREA_MAX_S
    while time.monotonic() < limite:
        _, cuerpo = _pedir(base + "/datos/tarea")
        tarea = json.loads(cuerpo)["tarea"]
        if tarea is not None and tarea["estado"] != "en_curso":
            return tarea
        time.sleep(0.4)
    pytest.fail(f"la tarea no termino en {TAREA_MAX_S:.0f}s")


class TestFlujoCompleto:
    """Las pruebas comparten servidor y van en orden: incorporar y luego reentrenar."""

    def test_01_el_panel_se_sirve(self, servidor: str) -> None:
        codigo, cuerpo = _pedir(servidor + "/datos/")
        assert codigo == 200
        assert "Añadir partidos" in cuerpo
        assert "tareas.js" in cuerpo

    def test_02_sirve_el_javascript_de_seguimiento(self, servidor: str) -> None:
        codigo, cuerpo = _pedir(servidor + "/static/tareas.js")
        assert codigo == 200 and cuerpo.strip()

    def test_03_validar_no_escribe_en_la_base_de_datos(
        self, servidor: str, entorno: dict
    ) -> None:
        antes = entorno["bd"].stat().st_mtime_ns
        codigo, _ = _pedir(servidor + "/datos/validar",
                           {"paquete": str(entorno["paquete"]["raiz"])})
        assert codigo == 200      # la redirección la sigue urllib
        tarea = esperar_tarea(servidor)
        assert tarea["estado"] == "terminada", tarea["lineas"]
        assert entorno["bd"].stat().st_mtime_ns == antes

    def test_04_incorporar_partidos_crea_un_conjunto_de_datos(
        self, servidor: str, entorno: dict
    ) -> None:
        antes = entorno["bd"].stat().st_mtime_ns
        codigo, _ = _pedir(servidor + "/datos/ingerir", {
            "paquete": str(entorno["paquete"]["raiz"]),
            "nombre_datos": "Con liga recien llegada",
        })
        assert codigo == 200
        tarea = esperar_tarea(servidor)
        assert tarea["estado"] == "terminada", tarea["lineas"]

        # El conjunto nuevo tiene su BD y sus metadatos, y el base NO se ha tocado.
        carpeta = entorno["bd"].parent / SUBDIR_CONJUNTOS / "con-liga-recien-llegada"
        assert (carpeta / entorno["bd"].name).is_file()
        assert (carpeta / FICHERO_CONJUNTO).is_file()
        assert entorno["bd"].stat().st_mtime_ns == antes

        # El panel enseña los dos: el base con 4 partidos y el nuevo con 5.
        _, panel = _pedir(servidor + "/datos/")
        assert "Liga Recien Llegada" in panel
        assert "<strong>4</strong> partidos" in panel
        assert "<strong>5</strong> partidos" in panel

    def test_05_entrenar_un_modelo_nuevo_lo_deja_servible(
        self, servidor: str, entorno: dict
    ) -> None:
        """Se entrena con nombre, en su carpeta, sin tocar el modelo de origen."""
        antes = {
            ruta.name: ruta.stat().st_mtime_ns
            for ruta in entorno["modelos"].glob("*.npz")
        }
        codigo, _ = _pedir(servidor + "/datos/reentrenar", {
            "nombre": "Con liga recien llegada",
            "datos_origen": "con-liga-recien-llegada",
        })
        assert codigo == 200
        tarea = esperar_tarea(servidor)
        assert tarea["estado"] == "terminada", tarea["lineas"]

        carpeta = entorno["modelos"] / SUBDIR_VARIANTES / "con-liga-recien-llegada"
        assert (carpeta / FICHERO_VARIANTE).is_file()
        # Las dos entidades, aunque no se haya elegido ninguna, y su estado warm.
        for entidad in ("jugador", "equipo"):
            stem = ClaveModelo(entidad).stem
            assert (carpeta / f"{stem}.npz").exists()
            assert (carpeta / f"{stem}.warm.npz").exists()
        # El de fábrica queda intacto: se puede volver a él desde el buscador.
        assert antes == {
            ruta.name: ruta.stat().st_mtime_ns
            for ruta in entorno["modelos"].glob("*.npz")
        }

    def test_06_los_equipos_nuevos_ya_se_pueden_consultar(self, servidor: str) -> None:
        """Sin reiniciar la app: los catálogos se releen al cambiar BD y modelos.

        Con el modelo nuevo, que es el que conoce a los equipos recién llegados;
        el de fábrica sigue respondiendo con el universo de antes.
        """
        codigo, cuerpo = _pedir(
            servidor + "/api/sugerencias?q=Equipo&entidad=equipo"
            "&modelo=con-liga-recien-llegada")
        assert codigo == 200
        sugerencias = json.loads(cuerpo)["sugerencias"]
        nombres = [s["nombre"] for s in sugerencias]
        assert "Equipo A" in nombres, nombres
        # Y con el nombre de su liga, que solo está en el conjunto nuevo: el
        # modelo se sirve con los adornos de la BD sobre la que se entrenó.
        nuevo = next(s for s in sugerencias if s["nombre"] == "Equipo A")
        assert "Liga Recien Llegada" in " ".join(nuevo["ligas_nombre"])

    def test_06b_el_modelo_nuevo_se_puede_elegir_en_el_buscador(
        self, servidor: str
    ) -> None:
        codigo, cuerpo = _pedir(servidor + "/")
        assert codigo == 200
        assert 'name="modelo"' in cuerpo
        assert "Con liga recien llegada" in cuerpo

    def test_07_una_carpeta_inexistente_devuelve_400(self, servidor: str) -> None:
        codigo, cuerpo = _pedir(servidor + "/datos/ingerir", {"paquete": "no/existe"})
        assert codigo == 400
        assert "No existe la carpeta" in cuerpo

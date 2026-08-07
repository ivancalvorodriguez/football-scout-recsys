"""Extremo a extremo de la app: `python -m src.app` sirviendo de verdad.

El cliente de pruebas de Flask (integración) no ejercita el arranque real: ni el
parseo de flags, ni que las plantillas y el CSS se encuentren desde el paquete
instalado, ni que el servidor levante. Esto lo hace por HTTP contra el proceso.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import pytest

from src.tests.conftest import RAIZ_REPO, ejecutar_modulo

pytestmark = [pytest.mark.e2e, pytest.mark.lento]

ARRANQUE_MAX_S = 30.0


def _puerto_libre() -> int:
    """Puerto que el SO da por libre ahora mismo.

    Hay una ventana de carrera entre cerrarlo aquí y que lo abra el servidor,
    pero es la forma habitual de hacerlo sin coordinar con el proceso hijo.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pedir(url: str, timeout: float = 10.0) -> tuple[int, str]:
    """GET que devuelve (código, cuerpo) también para respuestas de error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


@pytest.fixture(scope="module")
def servidor(dir_modelos: Path, bd: Path) -> Iterator[dict[str, Any]]:
    """Levanta el servidor en un puerto libre y lo para al acabar el modulo."""
    puerto = _puerto_libre()
    proceso = subprocess.Popen(
        [
            sys.executable, "-m", "src.app",
            "--modelo", str(dir_modelos),
            # Explicita a proposito: sin --bd caeria en la BD real del repo, si
            # existe, y el e2e dejaria de ser reproducible.
            "--bd", str(bd),
            "--host", "127.0.0.1", "--puerto", str(puerto),
        ],
        cwd=str(RAIZ_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    base = f"http://127.0.0.1:{puerto}"
    limite = time.monotonic() + ARRANQUE_MAX_S
    try:
        while time.monotonic() < limite:
            if proceso.poll() is not None:
                pytest.fail(f"el servidor murio al arrancar:\n{proceso.communicate()[0]}")
            try:
                _pedir(base + "/", timeout=1.0)
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail(f"el servidor no respondio en {ARRANQUE_MAX_S:.0f}s")
        yield {"base": base, "proceso": proceso}
    finally:
        proceso.terminate()
        try:
            proceso.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proceso.kill()


class TestServidor:
    def test_sirve_el_buscador(self, servidor: dict) -> None:
        codigo, cuerpo = _pedir(servidor["base"] + "/")
        assert codigo == 200
        assert "Buscar similares" in cuerpo

    def test_sirve_los_estaticos(self, servidor: dict) -> None:
        """El CSS y el JS viven en el paquete: si la ruta estática no encaja,
        la página se sirve igual pero sin estilo ni autocompletado."""
        for fichero in ("estilo.css", "buscador.js", "campo.js"):
            codigo, cuerpo = _pedir(f"{servidor['base']}/static/{fichero}")
            assert codigo == 200 and cuerpo.strip()

    def test_consulta_completa_por_http(self, servidor: dict) -> None:
        codigo, cuerpo = _pedir(
            servidor["base"] + "/similares?entidad=jugador&nombre=Messi&k=3"
        )
        assert codigo == 200
        assert "Lionel Messi" in cuerpo
        assert "Top 3 más similares" in cuerpo

    def test_la_api_responde_json(self, servidor: dict) -> None:
        codigo, cuerpo = _pedir(
            servidor["base"] + "/api/similares?entidad=equipo&nombre=Barcelona&k=2"
        )
        assert codigo == 200
        datos = json.loads(cuerpo)
        assert datos["referencia"]["nombre"] == "Barcelona"
        assert len(datos["candidatos"]) == 2

    def test_la_ficha_se_sirve_con_sus_figuras(self, servidor: dict) -> None:
        """El SVG del heptagono y el del campo se generan en el servidor."""
        codigo, cuerpo = _pedir(servidor["base"] + "/jugador/10")
        assert codigo == 200
        assert "radar-area" in cuerpo and "campo-juego" in cuerpo

    def test_un_nombre_inexistente_no_tumba_el_servidor(self, servidor: dict) -> None:
        codigo, _ = _pedir(servidor["base"] + "/similares?entidad=jugador&nombre=Nadie")
        assert codigo == 404
        assert _pedir(servidor["base"] + "/")[0] == 200


class TestAyuda:
    def test_la_ayuda_documenta_las_opciones(self) -> None:
        res = ejecutar_modulo("src.app", "--help")
        assert res.returncode == 0
        for opcion in ("--modelo", "--bd", "--host", "--puerto", "--debug"):
            assert opcion in res.stdout

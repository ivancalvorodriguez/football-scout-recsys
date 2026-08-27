"""Login y modo produccion contra un servidor de verdad.

Lo que aqui se ejercita y no cubre la integracion: que la cookie de sesion viaja
de verdad entre peticiones HTTP (no es el atajo del cliente de Flask), que
`--produccion` arranca con waitress y exige la clave de firma, y que sin entrar
no se llega a nada.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import pytest

from src.app import config as config_app
from src.tests.app.e2e.sesion import CONTRASENA, NOMBRE, USUARIO, Sesion, escribir_usuarios
from src.tests.conftest import RAIZ_REPO

pytestmark = [pytest.mark.e2e, pytest.mark.lento]

ARRANQUE_MAX_S = 30.0


def _puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _responde(base: str) -> bool:
    try:
        urllib.request.urlopen(base + "/login", timeout=1.0)
        return True
    except urllib.error.HTTPError:
        return True          # responde, aunque sea con un codigo de error
    except OSError:
        return False


def _levantar(argumentos: list[str], entorno: dict[str, str] | None = None):
    return subprocess.Popen(
        [sys.executable, "-m", "src.app", *argumentos],
        cwd=str(RAIZ_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env={**os.environ, **(entorno or {})},
    )


def _esperar(proceso, base: str) -> None:
    limite = time.monotonic() + ARRANQUE_MAX_S
    while time.monotonic() < limite:
        if proceso.poll() is not None:
            pytest.fail(f"el servidor murio al arrancar:\n{proceso.communicate()[0]}")
        if _responde(base):
            return
        time.sleep(0.2)
    pytest.fail(f"el servidor no respondio en {ARRANQUE_MAX_S:.0f}s")


@pytest.fixture(scope="module")
def cuentas(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return escribir_usuarios(tmp_path_factory.mktemp("cuentas") / "usuarios.json")


@pytest.fixture(scope="module")
def servidor(dir_modelos: Path, bd: Path, cuentas: Path) -> Iterator[str]:
    puerto = _puerto_libre()
    proceso = _levantar([
        "--modelo", str(dir_modelos), "--bd", str(bd),
        "--usuarios", str(cuentas),
        "--host", "127.0.0.1", "--puerto", str(puerto),
    ])
    base = f"http://127.0.0.1:{puerto}"
    try:
        _esperar(proceso, base)
        yield base
    finally:
        proceso.terminate()
        try:
            proceso.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proceso.kill()


# --- El recomendador, sin cuenta ----------------------------------------------
def test_la_portada_es_el_recomendador_y_se_sirve_sin_cuenta(servidor: str):
    sesion = Sesion(servidor)
    codigo, cuerpo = sesion.get("/")
    assert codigo == 200
    assert "Buscar similares" in cuerpo
    # Y ofrece las dos puertas en la cabecera.
    assert "Iniciar sesión" in cuerpo and "Registrarse" in cuerpo


def test_una_busqueda_sin_cuenta_por_http(servidor: str):
    codigo, cuerpo = Sesion(servidor).get(
        "/similares?entidad=jugador&nombre=Messi&k=3")
    assert codigo == 200
    assert "Lionel Messi" in cuerpo


def test_la_gestion_sin_cuenta_lleva_al_login(servidor: str):
    codigo, cuerpo = Sesion(servidor).get("/datos/")
    assert codigo == 200            # urllib sigue la redireccion
    assert "Iniciar sesión" in cuerpo
    assert "Añadir partidos" not in cuerpo


# --- Sesion por HTTP ----------------------------------------------------------
def test_entrar_y_seguir_dentro_entre_peticiones(servidor: str):
    """La cookie de sesion tiene que viajar de verdad, no solo dentro de Flask."""
    sesion = Sesion(servidor)
    sesion.entrar()
    assert sesion.get("/datos/")[0] == 200
    assert "Añadir partidos" in sesion.get("/datos/")[1]
    assert NOMBRE in sesion.get("/")[1]


def test_registrarse_por_http_deja_usar_la_gestion(servidor: str):
    sesion = Sesion(servidor)
    assert "Añadir partidos" not in sesion.get("/datos/")[1]
    codigo, _ = sesion.post("/registro", {
        "usuario": "reciennacida", "nombre": "Recién Nacida",
        "contrasena": "contrasena-nueva-1", "contrasena2": "contrasena-nueva-1",
    })
    assert codigo == 200            # urllib sigue la redireccion a la portada
    assert "Añadir partidos" in sesion.get("/datos/")[1]
    assert "Recién Nacida" in sesion.get("/")[1]


def test_las_credenciales_malas_no_entran(servidor: str):
    sesion = Sesion(servidor)
    codigo, _ = sesion.post("/login", {"usuario": USUARIO, "contrasena": "no-es"})
    assert codigo == 401
    assert "Añadir partidos" not in sesion.get("/datos/")[1]


def test_salir_cierra_la_sesion(servidor: str):
    sesion = Sesion(servidor)
    sesion.entrar()
    sesion.post("/logout")
    assert "Añadir partidos" not in sesion.get("/datos/")[1]
    # Pero el recomendador se le sigue sirviendo.
    assert "Buscar similares" in sesion.get("/")[1]


def test_un_post_sin_token_se_rechaza_por_http(servidor: str):
    """El agujero verificado en la auditoria: un POST de otro origen se procesaba."""
    sesion = Sesion(servidor)
    sesion.entrar()
    codigo, _ = sesion.post("/datos/validar", {"paquete": "."}, con_csrf=False)
    assert codigo == 400


def test_las_cabeceras_de_seguridad_llegan_por_http(servidor: str):
    with urllib.request.urlopen(servidor + "/login", timeout=10) as r:
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in r.headers["Content-Security-Policy"]


def test_el_glosario_se_sirve_sin_sesion(servidor: str):
    codigo, cuerpo = Sesion(servidor).get("/glosario")
    assert codigo == 200
    assert "Entrar" in cuerpo or "Glosario" in cuerpo


# --- Modo produccion ----------------------------------------------------------
def test_produccion_sin_clave_no_arranca(dir_modelos: Path, cuentas: Path):
    """Y lo dice: la alternativa seria un servidor en pie con sesiones rotas."""
    entorno = {k: v for k, v in os.environ.items()}
    entorno.pop(config_app.ENV_CLAVE, None)
    proceso = subprocess.Popen(
        [sys.executable, "-m", "src.app", "--produccion",
         "--modelo", str(dir_modelos), "--usuarios", str(cuentas),
         "--puerto", str(_puerto_libre())],
        cwd=str(RAIZ_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=entorno,
    )
    salida, _ = proceso.communicate(timeout=60)
    assert proceso.returncode == 1
    assert config_app.ENV_CLAVE in salida


def test_produccion_con_clave_sirve_con_waitress(dir_modelos: Path, bd: Path,
                                                 cuentas: Path):
    """Levanta de verdad con waitress y sirve, con sus cabeceras.

    **No se comprueba el login aqui, y no es un olvido**: en produccion la cookie
    de sesion va marcada `Secure`, asi que el navegador no la devuelve por HTTP
    plano y ninguna sesion puede completarse. Es lo correcto —`--produccion` da
    por hecho HTTPS, normalmente detras de un proxy inverso— y es justo lo que
    esta prueba fija: que la cookie sale marcada. El circuito de sesion completo
    se ejercita en el resto del modulo, sobre el servidor de desarrollo.
    """
    puerto = _puerto_libre()
    proceso = _levantar(
        ["--produccion", "--modelo", str(dir_modelos), "--bd", str(bd),
         "--usuarios", str(cuentas), "--host", "127.0.0.1",
         "--puerto", str(puerto), "--hilos", "4"],
        entorno={config_app.ENV_CLAVE: "clave-de-prueba-e2e"},
    )
    base = f"http://127.0.0.1:{puerto}"
    try:
        _esperar(proceso, base)
        with urllib.request.urlopen(base + "/login", timeout=10) as r:
            assert r.status == 200
            assert r.headers["X-Frame-Options"] == "DENY"
            # La cookie de sesion (la lleva el token CSRF al pintar el formulario).
            galleta = r.headers.get("Set-Cookie", "")
            assert "Secure" in galleta and "HttpOnly" in galleta
        # Waitress se anuncia en la cabecera del servidor: es la comprobacion de
        # que NO es el de desarrollo de Werkzeug.
        with urllib.request.urlopen(base + "/glosario", timeout=10) as r:
            assert "waitress" in r.headers.get("Server", "").lower()
    finally:
        proceso.terminate()
        try:
            proceso.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proceso.kill()


def test_produccion_y_debug_juntos_se_rechazan(dir_modelos: Path):
    proceso = _levantar(["--produccion", "--debug", "--modelo", str(dir_modelos)])
    salida, _ = proceso.communicate(timeout=60)
    assert proceso.returncode == 2
    assert "incompatibles" in salida


def test_el_wsgi_exige_la_clave(monkeypatch):
    """`src.app.wsgi:app` se crea en modo produccion al importarlo."""
    entorno = {k: v for k, v in os.environ.items()}
    entorno.pop(config_app.ENV_CLAVE, None)
    resultado = subprocess.run(
        [sys.executable, "-c", "import src.app.wsgi"],
        cwd=str(RAIZ_REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=entorno, timeout=120,
    )
    assert resultado.returncode != 0
    assert config_app.ENV_CLAVE in resultado.stderr


def test_el_wsgi_expone_la_app_con_clave(dir_modelos: Path, cuentas: Path):
    entorno = {
        **os.environ,
        config_app.ENV_CLAVE: "clave-de-prueba-e2e",
        config_app.ENV_MODELO: str(dir_modelos),
        config_app.ENV_USUARIOS: str(cuentas),
    }
    resultado = subprocess.run(
        [sys.executable, "-c",
         "from src.app.wsgi import app; print(app.config['APP_SIMILITUD'].produccion)"],
        cwd=str(RAIZ_REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=entorno, timeout=120,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "True" in resultado.stdout

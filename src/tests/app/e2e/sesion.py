"""Cliente HTTP con sesion para los e2e, y el registro de cuentas que usan.

Los e2e hablan con un servidor de verdad por HTTP, asi que no pueden usar el
cliente de pruebas de Flask ni los fixtures de sesion. Necesitan lo mismo que un
navegador: guardar la cookie de sesion entre peticiones, entrar por el formulario
de login y adjuntar el token CSRF en cada POST.

Vive aparte de los dos ficheros de e2e porque los dos levantan un servidor y los
dos tienen que entrar; duplicarlo seria copiar la parte mas facil de equivocar.
"""

from __future__ import annotations

import http.cookiejar
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.app.auth import Usuarios, crear_usuario

USUARIO = "e2e"
NOMBRE = "Cuenta de prueba"
CONTRASENA = "contrasena-de-prueba-e2e"

_CAMPO_CSRF = re.compile(r'name="csrf_token" value="([^"]+)"')


def escribir_usuarios(destino: Path) -> Path:
    """Registro con una sola cuenta, para arrancar el servidor con `--usuarios`."""
    Usuarios(destino).guardar(crear_usuario(USUARIO, NOMBRE, CONTRASENA))
    return destino


class Sesion:
    """Cliente HTTP que se comporta como un navegador con sesion iniciada.

    Guarda las cookies, saca el token CSRF de la pagina que toque y lo adjunta a
    los POST. Devuelve `(codigo, cuerpo)` igual que los `_pedir` que habia antes,
    para que las pruebas cambien lo menos posible.
    """

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self._galletas = http.cookiejar.CookieJar()
        self._abridor = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._galletas))

    # --- Peticiones ----------------------------------------------------------
    def get(self, ruta: str, timeout: float = 30.0) -> tuple[int, str]:
        return self._abrir(ruta, None, timeout)

    def post(self, ruta: str, datos: dict[str, str] | None = None,
             timeout: float = 30.0, con_csrf: bool = True) -> tuple[int, str]:
        cuerpo = dict(datos or {})
        if con_csrf:
            cuerpo.setdefault("csrf_token", self.token_csrf())
        return self._abrir(ruta, cuerpo, timeout)

    def _abrir(self, ruta: str, datos: dict[str, str] | None,
               timeout: float) -> tuple[int, str]:
        url = ruta if ruta.startswith("http") else self.base + ruta
        cuerpo = urllib.parse.urlencode(datos).encode() if datos is not None else None
        try:
            with self._abridor.open(url, data=cuerpo, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    # --- Sesion --------------------------------------------------------------
    def token_csrf(self) -> str:
        """Token de esta sesion, sacado de la pagina que lo lleve.

        Con sesion iniciada `/login` redirige, asi que se busca primero ahi (que
        es lo que vale ANTES de entrar) y, si no aparece, en la portada, cuyo
        formulario de salir lo lleva.
        """
        for ruta in ("/login", "/", "/datos/"):
            _, cuerpo = self.get(ruta)
            encontrado = _CAMPO_CSRF.search(cuerpo)
            if encontrado:
                return encontrado.group(1)
        raise AssertionError("no se encontro ningun token CSRF en el servidor")

    def entrar(self, usuario: str = USUARIO, contrasena: str = CONTRASENA) -> None:
        codigo, cuerpo = self.post(
            "/login", {"usuario": usuario, "contrasena": contrasena})
        # `urllib` sigue la redireccion, asi que un login correcto acaba en 200
        # sobre la portada; uno incorrecto se queda en el formulario (401).
        assert codigo == 200, f"no se pudo entrar ({codigo}): {cuerpo[:300]}"
        assert "Entrar" not in cuerpo or "Salir" in cuerpo

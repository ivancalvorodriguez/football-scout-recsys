"""Protección CSRF de todos los POST, sin dependencias nuevas.

El agujero verificado: un POST a `/datos/validar` con `Origin` de otro dominio se
procesaba. Con el servidor levantado, cualquier página que el usuario visitase
podía disparar una ingesta o un entrenamiento en su nombre — el navegador manda
la cookie de sesión sola, y al servidor le llegaba una petición indistinguible de
una legítima.

Cómo se cierra, y por qué así:

- **Un token por sesión**, de `secrets.token_urlsafe`. Va DENTRO de la sesión
  firmada de Flask, no en un diccionario de módulo. Es la diferencia entre
  funcionar y no funcionar en cuanto haya más de un proceso: un diccionario vive
  en la memoria de UN intérprete, así que el token que emite un worker no lo
  reconoce el de al lado y la mitad de los envíos se rechazarían. La cookie
  firmada viaja con el usuario y la valida cualquiera que tenga la clave.
- **Se compara con `hmac.compare_digest`**, no con `==`. La comparación normal
  corta en el primer byte distinto y el tiempo de respuesta va filtrando el
  token.
- **Se valida en TODOS los POST, incluido `/login`**. Un login sin protección
  permite meter a la víctima en la cuenta del atacante (*login CSRF*) y que todo
  lo que haga después acabe registrado ahí.
- **Falta o no cuadra -> 400**, no 403: no es un problema de permisos, es una
  petición mal formada desde el punto de vista del servidor.

Lo que NO cubre: no reemplaza a `SameSite=Lax` en la cookie (que ya evita que la
cookie viaje en un POST de otro sitio), sino que lo dobla. Los dos a la vez es lo
razonable mientras haya navegadores que no apliquen el default de SameSite.
"""

from __future__ import annotations

import hmac
import secrets

from flask import request, session

# Clave del token dentro de la sesión firmada, nombre del campo oculto en los
# formularios y de la cabecera equivalente (la usa el JS de la página de datos).
CLAVE_SESION = "_csrf"
CAMPO = "csrf_token"
CABECERA = "X-CSRF-Token"

# Métodos que cambian estado. GET/HEAD/OPTIONS quedan fuera por definición: si
# alguno de ellos cambiase algo, el problema sería ese y no el token.
METODOS_PROTEGIDOS = {"POST", "PUT", "PATCH", "DELETE"}

MENSAJE = (
    "Petición sin token de seguridad válido. Vuelve a cargar la página y "
    "reenvía el formulario."
)


class CSRFInvalido(ValueError):
    """El POST no trae un token que cuadre con la sesión (se traduce a 400)."""


def token() -> str:
    """Token de esta sesión, creándolo la primera vez.

    Es estable mientras dure la sesión: regenerarlo en cada respuesta rompe las
    pestañas que el usuario tuviera abiertas, que es la forma más habitual de
    que una protección CSRF acabe desactivada por molesta. Al entrar y al salir
    se limpia la sesión entera (`auth`), así que el token también se renueva.
    """
    valor = session.get(CLAVE_SESION)
    if not valor:
        valor = secrets.token_urlsafe(32)
        session[CLAVE_SESION] = valor
    return valor


def _enviado() -> str:
    """Token que trae la petición, del formulario o de la cabecera."""
    return (request.form.get(CAMPO)
            or request.headers.get(CABECERA)
            or "")


def validar() -> None:
    """Comprueba el token del POST en curso. `CSRFInvalido` si no cuadra."""
    esperado = session.get(CLAVE_SESION) or ""
    recibido = _enviado()
    if not esperado or not recibido:
        raise CSRFInvalido(MENSAJE)
    if not hmac.compare_digest(str(esperado), str(recibido)):
        raise CSRFInvalido(MENSAJE)


def registrar(app) -> None:
    """Cablea la validación y el global `csrf_token()` de las plantillas."""

    @app.before_request
    def _validar_csrf():
        if request.method in METODOS_PROTEGIDOS:
            validar()
        return None

    # Las plantillas escriben el campo oculto con `{{ csrf_token() }}`. Como
    # global y no como variable de contexto: así ninguna vista tiene que
    # acordarse de pasarlo, que es exactamente el olvido que deja un formulario
    # sin proteger.
    app.jinja_env.globals["csrf_token"] = token

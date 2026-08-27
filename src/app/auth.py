"""Cuentas de usuario: alta, verificación y sesión.

**Qué se protege y qué no.** El recomendador es público: buscar, comparar, las
fichas, el glosario y la API se consultan sin cuenta, porque son consulta de un
modelo ya construido y no cuestan más que una petición. Lo que exige cuenta es
`/datos`, que lanza subprocesos, copia bases de datos y entrena modelos —o sea,
lo que escribe en el disco del servidor y consume su CPU durante minutos u horas.
El reparto está en `config.PREFIJOS_PRIVADOS`.

**El alta es abierta**: cualquiera puede registrarse desde `/registro`. Es una
decisión de producto, y conviene tenerla escrita porque debilita lo anterior: con
auto-registro, «hace falta cuenta para entrenar» equivale a «hace falta rellenar
un formulario para entrenar». Lo que sigue acotando el daño son las cuotas por
cuenta (`config.MAX_*_POR_USUARIO`) y que cada cual solo ve lo suyo. En un
despliegue expuesto a internet habría que añadir algo más —invitación, lista de
altas permitidas o moderación—; el CLI `src.app.usuarios` sigue existiendo para
administrar cuentas al margen del formulario.

**Sin dependencias nuevas.** El hash es PBKDF2-HMAC-SHA256 de `hashlib`, con sal
por usuario y el número de iteraciones guardado en cada registro (así se puede
subir el coste más adelante sin invalidar las contraseñas ya dadas de alta). La
comparación es `hmac.compare_digest`, en tiempo constante: comparar con `==`
filtra por temporización cuántos bytes iniciales coinciden. La contraseña en
claro no se guarda, no se devuelve y no se escribe en el log en ningún camino,
incluido el de error.

**La sesión es la de Flask**, una cookie firmada con `SECRET_KEY`. No hay tabla
de sesiones en el servidor, y eso es deliberado: el estado compartido entre
procesos es justo lo que este despliegue no tiene (ver `docs/app_web.md`). Como
contrapartida, cerrar sesión solo borra la cookie del navegador; una cookie
robada vale hasta que cambie la clave.

**No hay roles ni permisos.** Todas las cuentas pueden hacer lo mismo; lo que las
separa es de quién es cada modelo y cada conjunto de datos (ver `catalogo` y
`conjuntos`). Un usuario no ve lo de otro, pero no porque tenga menos permisos,
sino porque no es suyo.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    current_app,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from . import config

bp_auth = Blueprint("auth", __name__)

# Claves dentro de la sesión firmada.
CLAVE_SESION = "usuario"

# Mensaje único para «no existe» y «contraseña incorrecta». Distinguirlos
# convierte el formulario en un oráculo de qué cuentas existen.
ERROR_CREDENCIALES = "Usuario o contraseña incorrectos."

# Nombre de cuenta admitido: letras ASCII, dígitos, guion y guion bajo. Acaba
# escrito en `variante.json` y en la cabecera, así que se acota a algo que no
# necesite escaparse en ningún sitio.
_NOMBRE_CUENTA = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class UsuarioDesconocido(LookupError):
    """No hay ninguna cuenta con ese nombre."""


class RegistroInvalido(ValueError):
    """Los datos del formulario de alta no sirven (se le enseñan al usuario)."""


@dataclass(frozen=True)
class Usuario:
    """Una cuenta. `hash` y `sal` van en hexadecimal para que el JSON sea texto."""

    usuario: str
    nombre: str
    sal: str
    hash: str
    iteraciones: int = config.PBKDF2_ITERACIONES
    creado: str | None = None

    def como_dict(self) -> dict[str, Any]:
        return {
            "nombre": self.nombre, "sal": self.sal, "hash": self.hash,
            "iteraciones": self.iteraciones, "creado": self.creado,
        }


def normalizar(usuario: str) -> str:
    """Nombre de cuenta canónico: sin espacios alrededor y en minúsculas.

    Se normaliza para que «Ivan» e «ivan» no sean dos cuentas distintas, que es
    una fuente clásica de suplantación por confusión.
    """
    return usuario.strip().lower()


def _derivar(contrasena: str, sal: bytes, iteraciones: int) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", contrasena.encode("utf-8"), sal, iteraciones
    )


def validar_alta(usuario: str, contrasena: str, repetida: str) -> str:
    """Comprueba los datos de un alta y devuelve el nombre de cuenta normalizado.

    Vive aquí y no en la vista porque el CLI aplica las mismas reglas: dos sitios
    que crean cuentas no pueden aceptar cosas distintas, o la que entra por un
    lado no se podría usar por el otro.

    `RegistroInvalido` con un mensaje pensado para enseñárselo al usuario tal
    cual: aquí no hay nada que ocultar —todavía no existe ninguna cuenta— y decir
    exactamente qué falla es lo que evita el bucle de reintentos a ciegas.
    """
    cuenta = normalizar(usuario)
    if not cuenta:
        raise RegistroInvalido("Escribe un nombre de usuario.")
    if len(cuenta) > config.MAX_LARGO_USUARIO:
        raise RegistroInvalido(
            f"El usuario no puede pasar de {config.MAX_LARGO_USUARIO} caracteres.")
    if not _NOMBRE_CUENTA.match(cuenta):
        raise RegistroInvalido(
            "El usuario solo admite letras sin acentos, números, punto, guion y "
            "guion bajo, y tiene que empezar por letra o número."
        )
    if len(contrasena) < config.MIN_LARGO_CONTRASENA:
        raise RegistroInvalido(
            f"La contraseña debe tener al menos {config.MIN_LARGO_CONTRASENA} "
            "caracteres."
        )
    if contrasena != repetida:
        raise RegistroInvalido("Las dos contraseñas no coinciden.")
    return cuenta


def crear_usuario(usuario: str, nombre: str, contrasena: str) -> Usuario:
    """Construye la cuenta con una sal nueva. No toca el disco."""
    sal = secrets.token_bytes(config.PBKDF2_SAL_BYTES)
    iteraciones = config.PBKDF2_ITERACIONES
    return Usuario(
        usuario=normalizar(usuario),
        nombre=(nombre or "").strip() or normalizar(usuario),
        sal=sal.hex(),
        hash=_derivar(contrasena, sal, iteraciones).hex(),
        iteraciones=iteraciones,
        creado=datetime.now().isoformat(timespec="seconds"),
    )


def verificar(cuenta: Usuario | None, contrasena: str) -> bool:
    """¿Es esa la contraseña de esa cuenta?

    Con `cuenta` a None (usuario inexistente) se deriva igualmente contra una sal
    ficticia antes de devolver False: si se devolviera en seco, responder a un
    usuario que no existe sería mucho más rápido que a uno que sí, y el tiempo de
    respuesta delataría qué cuentas hay.
    """
    if cuenta is None:
        _derivar(contrasena, b"\x00" * config.PBKDF2_SAL_BYTES,
                 config.PBKDF2_ITERACIONES)
        return False
    try:
        sal = bytes.fromhex(cuenta.sal)
        esperado = bytes.fromhex(cuenta.hash)
    except ValueError:      # registro corrupto: no autentica a nadie
        return False
    return hmac.compare_digest(_derivar(contrasena, sal, cuenta.iteraciones), esperado)


class Usuarios:
    """Las cuentas, leídas del JSON en cada consulta.

    No se cachea a propósito: dar de alta a alguien con `python -m
    src.app.usuarios` tiene que surtir efecto sin reiniciar el servidor, y el
    fichero tiene un puñado de líneas. Un fichero ausente o ilegible es «no hay
    cuentas», que deja la app cerrada — el fallo seguro es el que no deja entrar.
    """

    def __init__(self, ruta: Path) -> None:
        self.ruta = Path(ruta)

    # --- Lectura -------------------------------------------------------------
    def _leer(self) -> dict[str, dict]:
        try:
            # `utf-8-sig`: el fichero puede editarse a mano y en Windows casi
            # cualquier editor le mete un BOM.
            datos = json.loads(self.ruta.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        cuentas = datos.get("usuarios") if isinstance(datos, dict) else None
        return cuentas if isinstance(cuentas, dict) else {}

    def todos(self) -> list[Usuario]:
        salida = []
        for nombre_cuenta, bruto in sorted(self._leer().items()):
            if not isinstance(bruto, dict):
                continue
            salida.append(Usuario(
                usuario=nombre_cuenta,
                nombre=str(bruto.get("nombre") or nombre_cuenta),
                sal=str(bruto.get("sal") or ""),
                hash=str(bruto.get("hash") or ""),
                iteraciones=int(bruto.get("iteraciones")
                                or config.PBKDF2_ITERACIONES),
                creado=bruto.get("creado"),
            ))
        return salida

    def obtener(self, usuario: str) -> Usuario | None:
        clave = normalizar(usuario)
        return next((u for u in self.todos() if u.usuario == clave), None)

    @property
    def hay_cuentas(self) -> bool:
        return bool(self._leer())

    # --- Escritura -----------------------------------------------------------
    def guardar(self, cuenta: Usuario) -> None:
        """Da de alta o reemplaza una cuenta.

        Se escribe el fichero entero (son unas pocas cuentas) a través de un
        temporal y un `os.replace`: si el proceso muere a mitad, el fichero
        anterior sigue completo en vez de quedar truncado y dejar a todo el mundo
        fuera.
        """
        cuentas = self._leer()
        cuentas[cuenta.usuario] = cuenta.como_dict()
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = self.ruta.with_suffix(self.ruta.suffix + ".tmp")
        temporal.write_text(
            json.dumps({"version": 1, "usuarios": cuentas},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporal, self.ruta)

    def borrar(self, usuario: str) -> None:
        cuentas = self._leer()
        if cuentas.pop(normalizar(usuario), None) is None:
            raise UsuarioDesconocido(f"No existe la cuenta {usuario!r}.")
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = self.ruta.with_suffix(self.ruta.suffix + ".tmp")
        temporal.write_text(
            json.dumps({"version": 1, "usuarios": cuentas},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporal, self.ruta)


# --- Integración con Flask ----------------------------------------------------

def _usuarios() -> Usuarios:
    return current_app.extensions["usuarios"]


def usuario_actual() -> str | None:
    """Cuenta de la petición en curso, o None si no hay sesión.

    Se comprueba contra el fichero, no solo contra la cookie: borrar una cuenta
    tiene que echar fuera a quien ya estuviera dentro, y sin esto su cookie
    firmada seguiría valiendo hasta caducar.
    """
    if hasattr(g, "_usuario_verificado"):
        return g._usuario_verificado
    nombre = session.get(CLAVE_SESION)
    verificado = None
    if nombre and _usuarios().obtener(nombre) is not None:
        verificado = normalizar(nombre)
    g._usuario_verificado = verificado
    return verificado


def exigir_usuario() -> str:
    """La cuenta de la petición. Solo se llama detrás de `before_request`."""
    nombre = usuario_actual()
    if nombre is None:      # no debería ocurrir: la petición no habría llegado
        raise PermissionError("petición sin sesión")
    return nombre


def exige_sesion(ruta: str) -> bool:
    """¿Hace falta cuenta para esta ruta? Solo la gestión de datos y modelos.

    El prefijo tiene que cerrar en barra o coincidir entero: con un `startswith`
    a secas, `/datos-publicos` quedaría protegido sin que nadie lo decidiera —y,
    peor en la otra dirección, un prefijo mal escrito dejaría `/datos` abierto.
    """
    return any(ruta == p or ruta.startswith(p + "/")
               for p in config.PREFIJOS_PRIVADOS)


def registrar(app) -> None:
    """Cablea el guardia de sesión y las vistas de entrada, alta y salida."""

    @app.before_request
    def _exigir_sesion():
        if not exige_sesion(request.path):
            return None
        if usuario_actual() is not None:
            return None
        # `siguiente` devuelve al usuario a donde iba tras identificarse. Solo se
        # acepta una ruta del propio sitio (ver `_destino_seguro`). `full_path`
        # deja un '?' colgando cuando no hay query: molesta en la barra del
        # navegador y no aporta nada.
        return redirect(url_for("auth.login",
                                siguiente=request.full_path.rstrip("?")))

    @app.context_processor
    def _inyectar_usuario():
        """La cabecera enseña quién eres —o los dos botones si no hay sesión."""
        nombre = usuario_actual()
        cuenta = _usuarios().obtener(nombre) if nombre else None
        return {
            "usuario": nombre,
            "usuario_nombre": cuenta.nombre if cuenta else nombre,
        }

    app.register_blueprint(bp_auth)


def _destino_seguro(destino: str | None) -> str:
    """Ruta a la que volver tras entrar, saneada.

    Solo se admiten rutas relativas del propio sitio: sin esto, un enlace del
    tipo `/login?siguiente=https://otro-sitio` convierte el formulario en un
    redirector abierto, que es la mitad de un phishing convincente. Se rechaza
    también `//otro-sitio`, que el navegador interpreta como absoluta.
    """
    if not destino or not destino.startswith("/") or destino.startswith("//"):
        return url_for("web.inicio")
    return destino


@bp_auth.route("/login", methods=["GET", "POST"])
def login():
    """Formulario de entrada y su envío.

    El POST pasa igualmente por la validación CSRF (`csrf.registrar`): un
    formulario de login sin protección permite forzar la sesión de la víctima a
    una cuenta del atacante (*login CSRF*) y que lo que haga después acabe en
    ella.
    """
    destino = _destino_seguro(request.values.get("siguiente"))
    if request.method == "GET":
        if usuario_actual() is not None:
            return redirect(destino)
        return render_template("login.html", siguiente=destino, error="")

    nombre = normalizar(request.form.get("usuario") or "")
    contrasena = request.form.get("contrasena") or ""
    cuenta = _usuarios().obtener(nombre) if nombre else None
    if not verificar(cuenta, contrasena):
        # Se registra el intento SIN la contraseña y sin decir si la cuenta
        # existe: el log es para detectar fuerza bruta, no para reconstruirla.
        current_app.logger.warning("login fallido para %r desde %s",
                                   nombre, request.remote_addr)
        return render_template(
            "login.html", siguiente=destino, error=ERROR_CREDENCIALES,
            usuario_escrito=nombre,
        ), 401

    # Sesión nueva para el que entra: conservar la anterior permitiría fijarla
    # desde fuera (*session fixation*), y de paso el token CSRF se renueva.
    session.clear()
    session[CLAVE_SESION] = cuenta.usuario
    session.permanent = False
    current_app.logger.info("sesión iniciada: %s", cuenta.usuario)
    return redirect(destino)


@bp_auth.route("/registro", methods=["GET", "POST"])
def registro():
    """Alta de cuenta desde el navegador.

    Solo pide lo imprescindible: usuario, nombre visible (opcional) y contraseña
    por duplicado. Al crearla se **entra directamente**, porque obligar a repetir
    las credenciales recién escritas no aporta nada.

    El nombre ocupado se dice tal cual. Es información que el formulario filtra
    de todos modos —hay que decir por qué no se puede usar ese nombre— y
    disimularlo daría un alta que parece correcta y luego no deja entrar.
    """
    destino = _destino_seguro(request.values.get("siguiente"))
    escrito = {
        "usuario": (request.values.get("usuario") or "").strip(),
        "nombre": (request.values.get("nombre") or "").strip(),
    }
    if request.method == "GET":
        if usuario_actual() is not None:
            return redirect(destino)
        return render_template("registro.html", siguiente=destino, error="",
                               escrito=escrito)

    contrasena = request.form.get("contrasena") or ""
    almacen = _usuarios()
    try:
        cuenta = validar_alta(
            escrito["usuario"], contrasena, request.form.get("contrasena2") or "")
        if almacen.obtener(cuenta) is not None:
            raise RegistroInvalido(
                f"Ya hay una cuenta llamada «{cuenta}». Elige otro nombre o "
                "inicia sesión.")
        nuevo = crear_usuario(cuenta, escrito["nombre"], contrasena)
        almacen.guardar(nuevo)
    except RegistroInvalido as exc:
        return render_template("registro.html", siguiente=destino,
                               error=str(exc), escrito=escrito), 400
    except OSError as exc:
        current_app.logger.error("no se pudo escribir el registro de cuentas: %s", exc)
        return render_template(
            "registro.html", siguiente=destino, escrito=escrito,
            error="No se pudo crear la cuenta. Inténtalo de nuevo.",
        ), 500

    # Sesión nueva, igual que en el login: nunca se reutiliza la anterior.
    session.clear()
    session[CLAVE_SESION] = nuevo.usuario
    session.permanent = False
    current_app.logger.info("cuenta creada: %s", nuevo.usuario)
    return redirect(destino)


@bp_auth.post("/logout")
def logout():
    """Cierra la sesión. Es POST porque cambia estado: con un GET, cualquier
    `<img src="/logout">` de otra página echaría al usuario fuera."""
    session.clear()
    return redirect(url_for("auth.login"))

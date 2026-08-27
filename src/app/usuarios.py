"""Alta y cambio de contraseña de las cuentas, desde la terminal.

    python -m src.app.usuarios --alta ivan --nombre "Iván Calvo"
    python -m src.app.usuarios --contrasena ivan
    python -m src.app.usuarios --listar
    python -m src.app.usuarios --baja ivan

No hay registro desde el navegador, y es deliberado: una app con auto-registro
abierto es una app sin autenticación con pasos de más. Las cuentas las da de alta
quien administra el servidor, que es quien tiene acceso a esta línea de comandos.

La contraseña se pide con `getpass` (sin eco) y se confirma. Nunca llega por
argumento: lo que se escribe en la línea de comandos queda en el historial del
shell y es visible en la lista de procesos de la máquina. Tampoco se imprime ni
se registra en ningún sitio; lo único que se escribe en disco es su hash PBKDF2
con sal (ver `auth`).
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from . import config
from .auth import RegistroInvalido, Usuarios, crear_usuario, normalizar, validar_alta

# Mismo mínimo que el formulario de registro: dos sitios que crean cuentas no
# pueden aceptar cosas distintas, o la que entra por uno no valdría por el otro.
MIN_LARGO = config.MIN_LARGO_CONTRASENA


class Cancelado(RuntimeError):
    """El usuario abortó la operación (Ctrl+C o contraseñas que no coinciden)."""


def pedir_contrasena(usuario: str, pedir=getpass.getpass) -> str:
    """Pide la contraseña dos veces y comprueba que coinciden.

    `pedir` es un parámetro para poder probar esto sin una terminal: `getpass`
    exige un tty y en la suite no lo hay.
    """
    primera = pedir(f"Contraseña para «{usuario}»: ")
    if len(primera) < MIN_LARGO:
        raise Cancelado(f"La contraseña debe tener al menos {MIN_LARGO} caracteres.")
    if primera != pedir("Repítela: "):
        raise Cancelado("Las contraseñas no coinciden; no se ha cambiado nada.")
    return primera


def alta(almacen: Usuarios, usuario: str, nombre: str, contrasena: str) -> None:
    """Da de alta una cuenta nueva. Falla si ya existe (para eso está `--contrasena`).

    Pasa por la MISMA validación que el formulario de registro (`validar_alta`):
    una cuenta creada aquí con un nombre que el formulario rechazaría sería una
    cuenta que después no se puede administrar desde la web.
    """
    try:
        clave = validar_alta(usuario, contrasena, contrasena)
    except RegistroInvalido as exc:
        raise Cancelado(str(exc)) from exc
    if almacen.obtener(clave) is not None:
        raise Cancelado(
            f"Ya existe la cuenta «{clave}». Para cambiarle la contraseña: "
            f"--contrasena {clave}"
        )
    almacen.guardar(crear_usuario(clave, nombre, contrasena))


def cambiar(almacen: Usuarios, usuario: str, contrasena: str) -> None:
    """Cambia la contraseña de una cuenta que ya existe, conservando su nombre."""
    cuenta = almacen.obtener(usuario)
    if cuenta is None:
        raise Cancelado(f"No existe la cuenta «{normalizar(usuario)}».")
    almacen.guardar(crear_usuario(cuenta.usuario, cuenta.nombre, contrasena))


def main(argv: list[str] | None = None, pedir=getpass.getpass) -> int:
    p = argparse.ArgumentParser(
        description="Cuentas de la app web (alta, contraseña, baja, listado).")
    p.add_argument("--fichero", type=Path, default=config.RUTA_USUARIOS,
                   help="Dónde vive el registro de cuentas.")
    p.add_argument("--alta", metavar="USUARIO", help="Crea una cuenta nueva.")
    p.add_argument("--nombre", default="",
                   help="Nombre visible en la cabecera (por defecto, el usuario).")
    p.add_argument("--contrasena", metavar="USUARIO",
                   help="Cambia la contraseña de una cuenta existente.")
    p.add_argument("--baja", metavar="USUARIO", help="Elimina una cuenta.")
    p.add_argument("--listar", action="store_true", help="Lista las cuentas.")
    args = p.parse_args(argv)

    almacen = Usuarios(args.fichero)
    try:
        if args.alta:
            alta(almacen, args.alta, args.nombre,
                 pedir_contrasena(normalizar(args.alta), pedir))
            print(f"Cuenta «{normalizar(args.alta)}» creada en {args.fichero}.")
        elif args.contrasena:
            cambiar(almacen, args.contrasena,
                    pedir_contrasena(normalizar(args.contrasena), pedir))
            print(f"Contraseña de «{normalizar(args.contrasena)}» cambiada.")
        elif args.baja:
            cuenta = almacen.obtener(args.baja)
            if cuenta is None:
                raise Cancelado(f"No existe la cuenta «{normalizar(args.baja)}».")
            almacen.borrar(cuenta.usuario)
            # Lo suyo NO se borra: sus modelos y conjuntos siguen en disco a su
            # nombre. Borrarlos aquí sería destruir horas de entrenamiento por un
            # comando de administración de cuentas.
            print(f"Cuenta «{cuenta.usuario}» eliminada. Sus modelos y conjuntos "
                  f"siguen en disco a su nombre.")
        elif args.listar:
            cuentas = almacen.todos()
            if not cuentas:
                print(f"No hay ninguna cuenta en {args.fichero}.")
            for c in cuentas:
                print(f"  {c.usuario:<20} {c.nombre}"
                      f"{'  · ' + c.creado if c.creado else ''}")
        else:
            p.print_help()
            return 2
    except Cancelado as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelado.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error escribiendo {args.fichero}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

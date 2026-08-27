"""Punto de entrada de la app web.

    python -m src.app [--modelo DIR] [--bd RUTA] [--host H] [--puerto P]
                      [--debug | --produccion] [--hilos N] [--log NIVEL]

Sin `--produccion` levanta el servidor de desarrollo de Flask, que es lo
razonable en local (recarga automática con `--debug`, trazas en el navegador) y
lo que el propio Werkzeug avisa de no usar de cara al público.

Con `--produccion` sirve con **waitress**, un servidor WSGI de verdad, y en
modo endurecido: exige `SCOUTING_SECRET_KEY` y deja de enseñar rutas absolutas
del servidor y líneas de comandos en `/datos`. Se sirve con varios **hilos** y un
solo proceso: el estado de las tareas vive en memoria (ver `docs/app_web.md`).

Los dos modos son excluyentes: `--debug` enciende el depurador interactivo de
Werkzeug, que ejecuta código arbitrario desde el navegador. Combinarlo con
«producción» no es una configuración rara, es una puerta abierta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.similitud import config as config_similitud
from src.similitud.consulta import configurar_consola

from . import config as config_app
from .factoria import crear_app

configurar_consola()

# Hilos de waitress. Uno solo bastaría para el tráfico de esta app, pero las
# tareas largas se consultan por sondeo mientras corren: con un único hilo, una
# petición lenta (cargar la matriz S por primera vez, 51 MB) congelaría el
# sondeo y la barra de progreso se quedaría clavada.
HILOS_DEFECTO = 8


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Servidor web del recomendador por similitud.")
    p.add_argument("--modelo", type=Path, default=config_similitud.DEFAULT_MODEL_DIR,
                   help="Directorio con los modelos guardados.")
    p.add_argument("--bd", type=Path, default=config_similitud.DEFAULT_DB_PATH,
                   help="Base de datos de la extracción (solo lectura). Aporta los "
                        "nombres de liga, el equipo de cada jugador y su reparto de "
                        "minutos por posición; si falta, la app sirve sin esos bloques.")
    p.add_argument("--usuarios", type=Path, default=config_app.RUTA_USUARIOS,
                   help="Registro de cuentas (lo escribe python -m src.app.usuarios).")
    p.add_argument("--host", default="127.0.0.1",
                   help="Interfaz de escucha (127.0.0.1 = solo esta máquina).")
    p.add_argument("--puerto", type=int, default=5000)
    p.add_argument("--debug", action="store_true",
                   help="Recarga automática y trazas en el navegador (solo en local).")
    p.add_argument("--produccion", action="store_true",
                   help="Sirve con waitress y endurece la configuración. Exige "
                        f"{config_app.ENV_CLAVE} y da por hecho HTTPS (la cookie "
                        "de sesión va marcada Secure, así que por HTTP plano "
                        "nadie podría entrar): normalmente, detrás de un proxy "
                        "inverso que termina TLS.")
    p.add_argument("--hilos", type=int, default=HILOS_DEFECTO,
                   help=f"Hilos de waitress con --produccion (por defecto {HILOS_DEFECTO}).")
    p.add_argument("--cola", default=None,
                   help="URL de redis donde encolar las tareas largas de /datos, "
                        "que pasa a ejecutar el worker (python -m src.app.worker). "
                        f"Por defecto, ${config_app.ENV_COLA}; sin ninguna de las "
                        "dos, las ejecuta este mismo proceso.")
    p.add_argument("--log", default=None,
                   help="Nivel de log: DEBUG, INFO, WARNING, ERROR.")
    return p


def main(argv: list[str] | None = None) -> int:
    p = _parser()
    args = p.parse_args(argv)

    if args.produccion and args.debug:
        # `error` sale por stderr con código 2, que es lo que espera quien
        # automatiza el arranque.
        p.error(
            "--produccion y --debug son incompatibles: el depurador de Werkzeug "
            "ejecuta código arbitrario desde el navegador."
        )

    try:
        app = crear_app(
            args.modelo, db_path=args.bd, ruta_usuarios=args.usuarios,
            url_cola=args.cola, produccion=args.produccion, nivel_log=args.log,
        )
    except RuntimeError as exc:      # falta la clave de firma en producción
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Modelos: {args.modelo.resolve()}")
    if args.bd.is_file():
        print(f"Base de datos: {args.bd.resolve()}")
    else:
        print(f"Sin base de datos en {args.bd.resolve()} "
              f"(se omiten equipo, posiciones y nombres de liga).")
    cola = app.config["APP_SIMILITUD"].url_cola
    if cola:
        # Se dice al arrancar porque cambia quién ejecuta: si el worker no está
        # en pie, «Añadir partidos» y «Entrenar» aceptarán y no pasará nada
        # visible. Mejor saber desde el principio que la app ya no ejecuta.
        print(f"Cola de tareas: {cola} (las ejecuta el worker, no este proceso)")
    if not app.extensions["usuarios"].hay_cuentas:
        print("AVISO: no hay ninguna cuenta dada de alta, así que no se puede "
              "entrar. Crea una con:\n"
              '  python -m src.app.usuarios --alta <usuario> --nombre "<Nombre>"')

    if args.produccion:
        return _servir_waitress(app, args)
    print(f"Sirviendo en http://{args.host}:{args.puerto}  (Ctrl+C para parar)")
    app.run(host=args.host, port=args.puerto, debug=args.debug)
    return 0


def _servir_waitress(app, args) -> int:
    """Sirve con waitress. Un proceso, varios hilos.

    El import va dentro y no arriba porque waitress está en
    `requirements-app.txt` como dependencia del despliegue: quien solo levanta la
    app en local con `python -m src.app` no tiene por qué tenerlo instalado para
    que el módulo se pueda importar.
    """
    try:
        from waitress import serve
    except ImportError:
        print("Error: falta waitress. Instálalo con:\n"
              "  pip install -r requirements-app.txt", file=sys.stderr)
        return 1
    print(f"Sirviendo con waitress en http://{args.host}:{args.puerto} "
          f"({args.hilos} hilos, un solo proceso)  (Ctrl+C para parar)")
    serve(app, host=args.host, port=args.puerto, threads=args.hilos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

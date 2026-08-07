"""Punto de entrada de la app web.

    python -m src.app [--modelo DIR] [--bd RUTA] [--host H] [--puerto P] [--debug]

Levanta el servidor de desarrollo de Flask sobre los modelos ya construidos por
`python -m src.similitud.build`. Para producción se serviría la misma factoría
(`src.app:crear_app`) desde un WSGI (waitress/gunicorn); no es el caso de este
TFG, así que aquí basta el servidor integrado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.similitud import config as config_similitud
from src.similitud.consulta import configurar_consola

from .factoria import crear_app

configurar_consola()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Servidor web del recomendador por similitud.")
    p.add_argument("--modelo", type=Path, default=config_similitud.DEFAULT_MODEL_DIR,
                   help="Directorio con los modelos guardados.")
    p.add_argument("--bd", type=Path, default=config_similitud.DEFAULT_DB_PATH,
                   help="Base de datos de la extracción (solo lectura). Aporta los "
                        "nombres de liga, el equipo de cada jugador y su reparto de "
                        "minutos por posición; si falta, la app sirve sin esos bloques.")
    p.add_argument("--host", default="127.0.0.1",
                   help="Interfaz de escucha (127.0.0.1 = solo esta máquina).")
    p.add_argument("--puerto", type=int, default=5000)
    p.add_argument("--debug", action="store_true",
                   help="Recarga automática y trazas en el navegador (solo en local).")
    args = p.parse_args(argv)

    app = crear_app(args.modelo, db_path=args.bd)
    print(f"Modelos: {args.modelo.resolve()}")
    if args.bd.is_file():
        print(f"Base de datos: {args.bd.resolve()}")
    else:
        print(f"Sin base de datos en {args.bd.resolve()} "
              f"(se omiten equipo, posiciones y nombres de liga).")
    print(f"Sirviendo en http://{args.host}:{args.puerto}  (Ctrl+C para parar)")
    app.run(host=args.host, port=args.puerto, debug=args.debug)


if __name__ == "__main__":
    main()

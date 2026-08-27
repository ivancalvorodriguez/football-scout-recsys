"""HEALTHCHECK del contenedor de la app.

Va en un fichero y no en un `python -c` dentro del Dockerfile porque la
comprobación tiene una rama: `urlopen` lanza `HTTPError` en cualquier código >=
400, y un 404 significa que el servidor está vivo y contestando. Lo que hace
«no sana» a esta app es un 5xx o no contestar en absoluto.

La portada sirve aunque no haya ningún modelo construido (explica en pantalla
cómo construirlos), así que vale como sonda sin depender del estado de los datos.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/"
TIMEOUT = 4.0


def main() -> int:
    try:
        codigo = urllib.request.urlopen(URL, timeout=TIMEOUT).status
    except urllib.error.HTTPError as exc:
        codigo = exc.code
    except OSError as exc:                 # conexión rechazada, DNS, timeout
        print(f"sin respuesta: {exc}", file=sys.stderr)
        return 1
    return 0 if codigo < 500 else 1


if __name__ == "__main__":
    raise SystemExit(main())

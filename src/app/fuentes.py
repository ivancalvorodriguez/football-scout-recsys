"""Los catálogos que salen de la BD, uno por conjunto de datos.

`ligas`, `contexto` y `crudos` leen una base de datos concreta y la cachean en
memoria. Mientras hubo una sola BD, la app creaba uno de cada al arrancar. Con
varios conjuntos de datos ya no vale: **cada modelo se sirve con los adornos de
la BD sobre la que se entrenó**. Servir el modelo nuevo con la BD vieja dejaría
sin equipo, sin posiciones y sin valores reales justo a los jugadores que aporta
el conjunto nuevo — que son el motivo de haberlo creado.

Este módulo guarda una instancia por ruta y la reutiliza: son caros de construir
(el de `crudos` recalcula las features de toda la BD) y baratos de conservar.
Cuál toca en cada petición lo decide la vista, que resuelve el modelo y deja su
ruta en `g` con `usar`; los filtros de plantilla (`| ligas`, `| trayectoria`) la
leen de ahí en vez de arrastrar el catálogo por todos los contextos.
"""

from __future__ import annotations

import threading
from pathlib import Path

from flask import g

from .contexto import CatalogoContexto
from .crudos import CatalogoCrudos
from .ligas import CatalogoLigas

# Clave en `flask.g` de la BD de la petición en curso.
_CLAVE = "_fuente_bd"


def usar(db_path: Path) -> None:
    """Fija la BD de esta petición (la del conjunto del modelo que se sirve)."""
    setattr(g, _CLAVE, Path(db_path))


class Fuentes:
    """Catálogos de BD por ruta, con la de defecto como respaldo."""

    def __init__(self, db_defecto: Path) -> None:
        self.db_defecto = Path(db_defecto)
        self._ligas: dict[str, CatalogoLigas] = {}
        self._contexto: dict[str, CatalogoContexto] = {}
        self._crudos: dict[str, CatalogoCrudos] = {}
        self._lock = threading.Lock()

    def actual(self) -> Path:
        """BD de la petición en curso, o la de defecto fuera de una petición.

        `g` solo existe dentro de un contexto de aplicación; fuera de él (o en
        una vista que todavía no ha resuelto el modelo, como las páginas de
        error) se cae al conjunto base, que es lo que había antes de que
        hubiera varios.
        """
        try:
            return getattr(g, _CLAVE, self.db_defecto)
        except RuntimeError:      # sin contexto de aplicación
            return self.db_defecto

    def _obtener(self, cache: dict, clase, db_path: Path | None):
        ruta = Path(db_path) if db_path is not None else self.actual()
        clave = str(ruta)
        with self._lock:
            catalogo = cache.get(clave)
            if catalogo is None:
                catalogo = cache[clave] = clase(ruta)
        return catalogo

    def ligas(self, db_path: Path | None = None) -> CatalogoLigas:
        return self._obtener(self._ligas, CatalogoLigas, db_path)

    def contexto(self, db_path: Path | None = None) -> CatalogoContexto:
        return self._obtener(self._contexto, CatalogoContexto, db_path)

    def crudos(self, db_path: Path | None = None) -> CatalogoCrudos:
        return self._obtener(self._crudos, CatalogoCrudos, db_path)

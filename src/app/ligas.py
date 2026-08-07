"""Etiquetas legibles de las ligas, leídas de la BD (opcional).

Los modelos guardan la liga de cada entidad como `"<competition_id>-<season_id>"`
(p. ej. `"11-27"`), que es identidad suficiente para el pipeline pero ilegible en
una interfaz. Este módulo traduce esa clave a `"La Liga 2015/2016"` consultando
`competitions` y `seasons` en la BD, **en solo lectura**.

Es un adorno, no una dependencia: si la BD no está (la app solo necesita los
artefactos de `outputs/modelo/`), se devuelve la clave cruda y todo lo demás
sigue funcionando.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from pathlib import Path


class CatalogoLigas:
    """Traduce claves de liga a nombre; cachea la tabla en memoria."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._nombres: dict[str, str] | None = None
        self._huella: tuple[float, int] | None = None
        self._lock = threading.Lock()

    def _cargar(self) -> dict[str, str]:
        """`{"11-27": "La Liga 2015/2016"}` a partir de la BD.

        Cualquier fallo (BD ausente, esquema antiguo, fichero corrupto) se
        traduce en un catálogo vacío: la etiqueta cae a la clave cruda y la app
        no se cae por no poder adornar un nombre.
        """
        if not self.db_path.is_file():
            return {}
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        sql = """
            SELECT c.competition_id, s.season_id, c.competition_name, s.season_name
            FROM competitions c, seasons s
        """
        try:
            with closing(sqlite3.connect(uri, uri=True)) as conn:
                filas = conn.execute(sql).fetchall()
        except sqlite3.Error:
            return {}
        return {
            f"{comp_id}-{season_id}": f"{comp_name} {season_name}".strip()
            for comp_id, season_id, comp_name, season_name in filas
        }

    def _tabla(self) -> dict[str, str]:
        """Catálogo cacheado, releído si la BD ha cambiado.

        Misma huella (mtime, tamaño) que usa el catálogo de modelos; volver a
        extraer con el servidor levantado se refleja sin reiniciarlo.
        """
        huella: tuple[float, int] | None = None
        if self.db_path.is_file():
            estado = self.db_path.stat()
            huella = (estado.st_mtime, estado.st_size)
        with self._lock:
            if self._nombres is None or self._huella != huella:
                self._nombres = self._cargar()
                self._huella = huella
            return self._nombres

    def nombre(self, clave: str) -> str:
        """Nombre de la liga, o la propia clave si no se puede traducir."""
        return self._tabla().get(clave, clave)

    def nombres(self, claves: tuple[str, ...] | list[str]) -> list[str]:
        return [self.nombre(c) for c in claves]

    def texto(self, claves: tuple[str, ...] | list[str]) -> str:
        """Ligas de una entidad como texto para la interfaz."""
        if not claves:
            return "sin liga registrada"
        return ", ".join(self.nombres(claves))

"""Coherencia entre un modelo servido y la base de datos de su conjunto.

Un artefacto de `outputs/modelo/` es una foto: la S y los `entity_ids` que había
cuando se ajustó. La BD, en cambio, sigue creciendo (una extracción más, una
ingesta desde `/datos`). Nada obliga a que las dos cosas cuadren, y cuando no
cuadran **la app no se rompe**: sirve el modelo tal cual y las entidades que solo
están en la BD sencillamente no aparecen en ninguna búsqueda.

Ese silencio es el problema que resuelve este módulo. Se dio en producción: los
artefactos servidos cubrían 2176 jugadores y 80 equipos mientras la BD ya tenía
2640 y 98 (faltaba la Bundesliga 2015/2016 entera). La página de datos anunciaba
los 2640 y el buscador respondía «no encontrado» a los 464 restantes, sin que
nada relacionara las dos cifras.

Dos decisiones deliberadas:

- **Avisa, no impide.** Un modelo que cubre parte de la BD sigue siendo un modelo
  válido, y de hecho es el estado NORMAL justo después de incorporar partidos y
  antes de reentrenar. Lo que no es aceptable es que no se note.
- **El universo de la BD se mide como lo mide el pipeline**, no con
  `COUNT(*) FROM players`: `similitud.data.cargar_jugadores` parte de
  `player_match_stats` con JOIN a `players` y `matches`, así que un jugador
  fichado en la tabla pero sin estadísticas no entraría en el modelo y contarlo
  daría un aviso falso permanente.

El universo se cachea por BD con la misma huella (mtime, tamaño) que usan
`ligas`, `contexto` y `crudos`, de modo que ampliar la BD con el servidor
levantado se refleja sin reiniciarlo. La comparación en sí es una diferencia de
conjuntos de unos miles de enteros: no vale la pena cachearla.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

# Cómo se cuenta el universo de cada entidad. Es la traducción a SQL de lo que
# hace `src.similitud.data`: la tabla de estadísticas manda, y los JOIN son los
# mismos (un partido sin fila en `matches` tampoco llegaría al modelo).
#
# Se pide también el NOMBRE, y no solo el id, porque el universo tiene dos
# consumidores: contar (esto es «cuántos hay») y BUSCAR (el buscador ofrece
# también las entidades que solo están en la BD, para poder proyectarlas). Sacar
# las dos cosas de la misma consulta garantiza que lo que se puede encontrar es
# exactamente lo que se ha contado.
CONSULTAS: dict[str, str] = {
    "jugador": """
        SELECT DISTINCT p.player_id, pl.player_name
        FROM player_match_stats p
        JOIN players pl ON pl.player_id = p.player_id
        JOIN matches m ON m.match_id = p.match_id
    """,
    "equipo": """
        SELECT DISTINCT t.team_id, te.team_name
        FROM team_match_stats t
        JOIN teams te ON te.team_id = t.team_id
        JOIN matches m ON m.match_id = t.match_id
    """,
}


@dataclass(frozen=True)
class Cobertura:
    """Qué parte de la BD cubre un modelo, y qué parte de un modelo no está en la BD.

    Las dos direcciones importan y no son la misma:

    - `faltan` son entidades de la BD que el modelo no conoce. Es el caso que
      duele: el usuario las ve en `/datos` y el buscador no las encuentra.
    - `ajenas` son entidades del modelo que ya no están en la BD. Pasa al servir
      un modelo con el conjunto de datos equivocado; se sirven igual (el
      artefacto trae su nombre), pero sin equipo, posiciones ni valores reales.

    `medible` a False significa que no se ha podido leer la BD (no está, o el
    esquema no es el esperado). Entonces no se afirma nada: sin universo con el
    que comparar, cualquier aviso sería inventado.
    """

    entidad: str
    en_modelo: int
    en_datos: int
    cubiertas: int
    ajenas: int
    medible: bool = True

    @property
    def faltan(self) -> int:
        return self.en_datos - self.cubiertas

    @property
    def incompleta(self) -> bool:
        """True si hay entidades en la BD que el modelo no puede recomendar.

        Es el caso que se avisa en el BUSCADOR, y no `coherente` a secas: que el
        modelo traiga alguna entidad de más no le quita nada a quien busca (esas
        salen igual, con menos adornos), pero que le falten sí — son búsquedas
        que van a fallar sin explicación.
        """
        return self.medible and self.faltan > 0

    @property
    def coherente(self) -> bool:
        """True si el modelo cubre la BD entera y no arrastra entidades de otra."""
        return not self.medible or (self.faltan == 0 and self.ajenas == 0)

    @property
    def plural(self) -> str:
        return config.ETIQUETA_PLURAL.get(self.entidad, self.entidad)

    @property
    def texto(self) -> str:
        """Aviso en una línea, con las dos cifras que hay que poder comparar.

        Se escriben siempre las dos (N de M) en vez de solo las que faltan: «el
        modelo cubre 2176 de 2640 jugadores» dice a la vez que falta algo y
        cuánto es ese algo, que es lo que decide si toca reentrenar.
        """
        if not self.medible:
            return ""
        partes = []
        if self.faltan:
            partes.append(
                f"el modelo cubre {self.cubiertas} de {self.en_datos} "
                f"{self.plural} de estos datos"
            )
        if self.ajenas:
            partes.append(
                f"{self.ajenas} {self.plural} del modelo no están en estos datos"
            )
        return "; ".join(partes)


def ids_de_artefacto(ruta_npz: Path) -> list[int] | None:
    """Solo los `entity_ids` de un artefacto, sin cargar su matriz S.

    Medir la cobertura no necesita la S, y cargarla sí se nota: la de jugador
    ocupa 51 MB comprimidos. Con esto la portada y la página de datos pueden
    avisar de TODOS los modelos sin traerse ninguno a memoria — `np.load` sobre
    un `.npz` es perezoso y descomprime solo el miembro que se pide (unos 2 ms).

    None si el fichero no está o no es un `.npz` legible: sin ids no se afirma
    nada, igual que sin BD.
    """
    ruta_npz = Path(ruta_npz)
    if not ruta_npz.is_file():
        return None
    try:
        with np.load(ruta_npz, allow_pickle=False) as z:
            return [int(i) for i in z["entity_ids"]]
    except (OSError, ValueError, KeyError):
        return None


def medir(entidad: str, ids_modelo, ids_datos: frozenset[int] | None) -> Cobertura:
    """Compara los `entity_ids` de un artefacto con el universo de una BD.

    None en cualquiera de los dos lados es «no se ha podido leer» y se propaga
    como `medible=False`; un conjunto vacío, en cambio, es un dato (BD sin
    estadísticas de esa entidad) y sí se compara.
    """
    del_modelo = {int(i) for i in (ids_modelo or ())}
    if ids_datos is None or ids_modelo is None:
        return Cobertura(
            entidad=entidad, en_modelo=len(del_modelo), en_datos=0,
            cubiertas=0, ajenas=0, medible=False,
        )
    return Cobertura(
        entidad=entidad,
        en_modelo=len(del_modelo),
        en_datos=len(ids_datos),
        cubiertas=len(del_modelo & ids_datos),
        ajenas=len(del_modelo - ids_datos),
    )


class CatalogoUniverso:
    """Entidades que una BD aportaría al pipeline (id y nombre), cacheadas.

    Mismo contrato que `ligas` y `contexto`: una BD ausente, con esquema antiguo
    o corrupta devuelve None (no vacío) y la interfaz calla, en vez de dar por
    hecho que la BD no tiene a nadie y avisar de una incoherencia inexistente.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._cache: dict[str, dict[int, str] | None] = {}
        self._huella: tuple[float, int] | None = None
        self._lock = threading.Lock()

    def _consultar(self, entidad: str) -> dict[int, str] | None:
        sql = CONSULTAS.get(entidad)
        if sql is None or not self.db_path.is_file():
            return None
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as conn:
                return {int(i): str(n) for i, n in conn.execute(sql)}
        except sqlite3.Error:
            return None

    def nombres(self, entidad: str) -> dict[int, str] | None:
        """Mapa id -> nombre de esa entidad, releído si la BD ha cambiado."""
        huella: tuple[float, int] | None = None
        if self.db_path.is_file():
            estado = self.db_path.stat()
            huella = (estado.st_mtime, estado.st_size)
        with self._lock:
            if self._huella != huella:
                self._cache.clear()
                self._huella = huella
            if entidad not in self._cache:
                self._cache[entidad] = self._consultar(entidad)
            return self._cache[entidad]

    def ids(self, entidad: str) -> frozenset[int] | None:
        """Universo de esa entidad (solo los ids), para medir la cobertura."""
        nombres = self.nombres(entidad)
        return None if nombres is None else frozenset(nombres)

"""Conjuntos de datos: qué bases de datos hay y cómo se crea una nueva.

Mismo planteamiento que los modelos (`catalogo`), un nivel más abajo. Un
**conjunto de datos** es una base de datos de la extracción con un nombre:

- el **base**, el que produce `src.extraccion.extract`, en `outputs/db/scouting.db`;
- los creados **añadiendo partidos desde la interfaz**, cada uno en
  `outputs/db/conjuntos/<slug>/scouting.db` con su `conjunto.json` (nombre,
  fecha, de qué conjunto salió y qué paquete se incorporó).

Un conjunto nuevo **parte de una copia** del que se elija: incorporar partidos no
modifica nunca el de origen. Es lo que hace que se pueda volver atrás —el modelo
base sigue casando con el conjunto base— y lo que convierte la copia de seguridad
de `ingesta` en innecesaria (la copia ES el conjunto anterior, intacto).

Todas las bases se llaman igual dentro de su carpeta, así que los CLI
(`--db <ruta>`) y los catálogos que leen la BD funcionan con cualquiera sin saber
si es el base o uno con nombre.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import config
from .nombres import NombreInvalido, preparar


class ConjuntoNoDisponible(LookupError):
    """No existe el conjunto de datos pedido."""


@dataclass(frozen=True)
class Conjunto:
    """Un conjunto de datos con nombre y dónde vive su BD."""

    slug: str
    nombre: str
    ruta: Path
    creado: str | None = None
    origen: str | None = None
    paquete: str | None = None

    @property
    def es_base(self) -> bool:
        return self.slug == config.VARIANTE_BASE

    @property
    def existe(self) -> bool:
        """True si su fichero de BD está ahí.

        Puede ser False mientras la incorporación corre (la carpeta se reserva
        antes) o si falló a mitad.
        """
        return self.ruta.is_file()

    @property
    def fecha(self) -> datetime | None:
        """Cuándo se creó, o cuándo se escribió la BD si no lo lleva apuntado.

        El conjunto base no tiene `conjunto.json` —lo deja la extracción, no
        esta app—, así que su fecha es la del fichero.
        """
        if self.creado:
            try:
                return datetime.fromisoformat(self.creado)
            except ValueError:
                pass
        if self.existe:
            return datetime.fromtimestamp(self.ruta.stat().st_mtime)
        return None


class CatalogoDatos:
    """Conjuntos de datos disponibles a partir de la ruta de la BD base."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    # --- Layout --------------------------------------------------------------

    @property
    def dir_conjuntos(self) -> Path:
        """Carpeta que agrupa los conjuntos con nombre."""
        return self.db_path.parent / config.SUBDIR_CONJUNTOS

    def ruta(self, slug_pedido: str) -> Path:
        """Ruta de la BD de un conjunto (el nombre del fichero no cambia)."""
        if slug_pedido == config.VARIANTE_BASE:
            return self.db_path
        return self.dir_conjuntos / slug_pedido / self.db_path.name

    # --- Descubrimiento ------------------------------------------------------

    def _metadatos(self, carpeta: Path) -> dict:
        try:
            # `utf-8-sig`: lo escribe `crear` sin BOM, pero es texto que se puede
            # reescribir a mano y casi cualquier editor de Windows le mete uno.
            datos = json.loads(
                (carpeta / config.FICHERO_CONJUNTO).read_text(encoding="utf-8-sig")
            )
        except (OSError, json.JSONDecodeError):
            return {}
        return datos if isinstance(datos, dict) else {}

    def conjuntos(self) -> list[Conjunto]:
        """Los conjuntos declarados, el base primero y el resto por fecha.

        El base se lista siempre, exista o no su fichero: es el destino de
        `src.extraccion.extract` y el punto de partida de todo lo demás.
        """
        salida = [Conjunto(
            slug=config.VARIANTE_BASE,
            nombre=config.NOMBRE_BASE,
            ruta=self.db_path,
        )]
        if self.dir_conjuntos.is_dir():
            nombrados = []
            for carpeta in sorted(self.dir_conjuntos.iterdir()):
                if not carpeta.is_dir():
                    continue
                meta = self._metadatos(carpeta)
                nombrados.append(Conjunto(
                    slug=carpeta.name,
                    nombre=str(meta.get("nombre") or carpeta.name),
                    ruta=carpeta / self.db_path.name,
                    creado=meta.get("creado"),
                    origen=meta.get("origen"),
                    paquete=meta.get("paquete"),
                ))
            nombrados.sort(key=lambda c: (c.creado or "", c.slug))
            salida += nombrados
        return salida

    def disponibles(self) -> list[Conjunto]:
        """Solo los que tienen BD: los que se pueden consultar o ampliar."""
        return [c for c in self.conjuntos() if c.existe]

    def conjunto(self, slug_pedido: str) -> Conjunto:
        """El conjunto de ese slug. `ConjuntoNoDisponible` si no existe."""
        for c in self.conjuntos():
            if c.slug == slug_pedido:
                return c
        raise ConjuntoNoDisponible(
            f"No existe ningún conjunto de datos llamado {slug_pedido!r}."
        )

    def resolver(self, slug_pedido: str | None) -> Conjunto:
        """Conjunto pedido, o el base si no se pide ninguno o no se reconoce.

        Que un slug desconocido caiga al base (en vez de fallar) es deliberado:
        lo pide un MODELO, que apunta al conjunto con el que se entrenó. Si esa
        carpeta ya no está, se sirve igual con los adornos del base —perderá
        algún nombre de equipo— en vez de dejar de servir el modelo entero.
        """
        if slug_pedido:
            for c in self.conjuntos():
                if c.slug == slug_pedido:
                    return c
        return self.conjuntos()[0]

    # --- Alta de un conjunto nuevo -------------------------------------------

    def crear(self, nombre: str, origen: str = config.VARIANTE_BASE,
              paquete: str | None = None) -> Conjunto:
        """Reserva la carpeta de un conjunto nuevo con una COPIA del de origen.

        La copia es lo que hace que incorporar partidos sea seguro: la ingesta
        escribe solo sobre ella (por eso se lanza con `--sin-copia`) y el
        conjunto del que se partió queda intacto, con sus modelos todavía
        válidos.
        """
        limpio, destino = preparar(
            nombre, (c.slug for c in self.conjuntos()), "conjunto de datos")
        fuente = self.conjunto(origen)
        if not fuente.existe:
            raise NombreInvalido(
                f"El conjunto «{fuente.nombre}» todavía no tiene base de datos: "
                "no se puede partir de él."
            )
        carpeta = self.dir_conjuntos / destino
        carpeta.mkdir(parents=True)
        # `sqlite3.backup` y no `shutil.copy2`: copia una BD coherente aunque
        # alguien la esté leyendo, que es exactamente el caso (la app la tiene
        # abierta para los adornos de la ficha).
        with closing(sqlite3.connect(fuente.ruta)) as origen_conn:
            with closing(sqlite3.connect(carpeta / self.db_path.name)) as copia:
                origen_conn.backup(copia)
        conjunto = Conjunto(
            slug=destino,
            nombre=limpio,
            ruta=carpeta / self.db_path.name,
            creado=datetime.now().isoformat(timespec="seconds"),
            origen=origen,
            paquete=paquete,
        )
        (carpeta / config.FICHERO_CONJUNTO).write_text(
            json.dumps(
                {"nombre": conjunto.nombre, "slug": conjunto.slug,
                 "creado": conjunto.creado, "origen": conjunto.origen,
                 "paquete": conjunto.paquete},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        return conjunto

    def descartar(self, slug_pedido: str) -> None:
        """Borra la carpeta de un conjunto recién creado.

        Solo para deshacer un alta que no llegó a lanzarse (ver `rutas_datos`):
        sin esto el nombre quedaría ocupado por una copia que nadie va a ampliar.
        """
        if slug_pedido == config.VARIANTE_BASE:
            raise ValueError("el conjunto base no se descarta")
        shutil.rmtree(self.dir_conjuntos / slug_pedido, ignore_errors=True)

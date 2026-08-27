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


class CuotaSuperada(RuntimeError):
    """La cuenta ya tiene tantos conjuntos de datos como se le permiten."""


@dataclass(frozen=True)
class Conjunto:
    """Un conjunto de datos con nombre y dónde vive su BD."""

    slug: str
    nombre: str
    ruta: Path
    creado: str | None = None
    origen: str | None = None
    paquete: str | None = None
    # Cuenta que lo creó. Mismas reglas que en `catalogo.Variante`: `None` es
    # compartido (el base, y lo que existiera antes de haber cuentas), se ve
    # desde todas las cuentas, no lo borra nadie y no cuenta cuota.
    usuario: str | None = None

    @property
    def es_base(self) -> bool:
        return self.slug == config.VARIANTE_BASE

    @property
    def compartido(self) -> bool:
        return self.usuario is None

    def visible_para(self, usuario: str | None) -> bool:
        return self.compartido or self.usuario == usuario

    def borrable_por(self, usuario: str | None) -> bool:
        return not self.es_base and not self.compartido and self.usuario == usuario

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

    def todos_los_conjuntos(self) -> list[Conjunto]:
        """TODO lo que hay en disco, sin filtrar por dueño.

        De uso interno, igual que `Catalogo.todas_las_variantes`: contar cuota,
        comprobar nombres libres y resolver la BD que apunta un modelo. Las
        vistas usan `conjuntos(usuario)`.
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
                dueno = meta.get("usuario")
                nombrados.append(Conjunto(
                    slug=carpeta.name,
                    nombre=str(meta.get("nombre") or carpeta.name),
                    ruta=carpeta / self.db_path.name,
                    creado=meta.get("creado"),
                    origen=meta.get("origen"),
                    paquete=meta.get("paquete"),
                    usuario=str(dueno) if dueno else None,
                ))
            nombrados.sort(key=lambda c: (c.creado or "", c.slug))
            salida += nombrados
        return salida

    def conjuntos(self, usuario: str | None) -> list[Conjunto]:
        """Los que esa cuenta puede ver: los suyos y los compartidos.

        Sin valor por defecto, por el mismo motivo que en `catalogo`: olvidarse
        de filtrar tiene que ser un `TypeError`, no una fuga silenciosa.
        """
        return [c for c in self.todos_los_conjuntos() if c.visible_para(usuario)]

    def disponibles(self, usuario: str | None) -> list[Conjunto]:
        """Solo los que tienen BD: los que se pueden consultar o ampliar."""
        return [c for c in self.conjuntos(usuario) if c.existe]

    def conjunto(self, slug_pedido: str, usuario: str | None) -> Conjunto:
        """El conjunto de ese slug, si esa cuenta puede verlo."""
        for c in self.conjuntos(usuario):
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

        **No filtra por dueño y no debe hacerlo**: no es una respuesta a nadie,
        es la traducción de «este modelo se entrenó con estos datos» a una ruta
        de fichero. El control de quién ve qué está un paso antes, en el modelo:
        si el usuario no puede ver el modelo, nunca se llega aquí.
        """
        conjuntos = self.todos_los_conjuntos()
        if slug_pedido:
            for c in conjuntos:
                if c.slug == slug_pedido:
                    return c
        return conjuntos[0]

    # --- Cuota ---------------------------------------------------------------

    def propios(self, usuario: str | None) -> list[Conjunto]:
        """Los que ha creado esa cuenta (los que le cuentan cuota)."""
        return [c for c in self.todos_los_conjuntos() if c.usuario == usuario]

    def cuota(self, usuario: str | None) -> tuple[int, int]:
        """(cuántos tiene, cuántos puede tener) esa cuenta."""
        return len(self.propios(usuario)), config.MAX_DATASETS_POR_USUARIO

    def hay_hueco(self, usuario: str | None) -> bool:
        usados, tope = self.cuota(usuario)
        return usados < tope

    # --- Alta de un conjunto nuevo -------------------------------------------

    def crear(self, nombre: str, usuario: str | None,
              origen: str = config.VARIANTE_BASE,
              paquete: str | None = None) -> Conjunto:
        """Reserva la carpeta de un conjunto nuevo con una COPIA del de origen.

        La copia es lo que hace que incorporar partidos sea seguro: la ingesta
        escribe solo sobre ella (por eso se lanza con `--sin-copia`) y el
        conjunto del que se partió queda intacto, con sus modelos todavía
        válidos.

        Cada copia es una BD entera (12 MB con las cinco ligas de fábrica), así
        que la cuota se mira ANTES de copiar: comprobarla después habría gastado
        ya el disco que el tope pretende ahorrar.
        """
        usados, tope = self.cuota(usuario)
        if usados >= tope:
            raise CuotaSuperada(
                f"Has llegado al máximo de {tope} conjuntos de datos. Borra "
                f"alguno de los tuyos antes de crear otro."
            )
        limpio, destino = preparar(
            nombre, (c.slug for c in self.todos_los_conjuntos()), "conjunto de datos")
        fuente = self.conjunto(origen, usuario)
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
            usuario=usuario,
        )
        (carpeta / config.FICHERO_CONJUNTO).write_text(
            json.dumps(
                {"nombre": conjunto.nombre, "slug": conjunto.slug,
                 "creado": conjunto.creado, "origen": conjunto.origen,
                 "paquete": conjunto.paquete, "usuario": conjunto.usuario},
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

    def borrar(self, slug_pedido: str, usuario: str | None) -> Conjunto:
        """Borra un conjunto del usuario. Devuelve el borrado.

        Igual que en `catalogo`: lo que no es suyo da el mismo error que lo que
        no existe. Los modelos que apuntaban a este conjunto NO se borran —son
        otra cosa, con su propia cuota— y siguen sirviéndose; lo que pierden es
        el contexto de la BD (equipo, posiciones, valores reales), porque
        `resolver` cae al conjunto base cuando el suyo ya no está.
        """
        objetivo = next(
            (c for c in self.todos_los_conjuntos() if c.slug == slug_pedido), None)
        if objetivo is None or not objetivo.borrable_por(usuario):
            raise ConjuntoNoDisponible(
                f"No existe ningún conjunto de datos llamado {slug_pedido!r}.")
        shutil.rmtree(self.dir_conjuntos / objetivo.slug)
        return objetivo

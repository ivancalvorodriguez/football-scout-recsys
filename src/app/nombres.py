"""Nombres que escribe el usuario: normalización y validación.

Los **modelos** (`catalogo`) y los **conjuntos de datos** (`conjuntos`) se crean
de la misma forma: el usuario escribe un nombre, ese nombre acaba siendo una
carpeta en disco y un parámetro de URL, y dos con el mismo nombre serían
indistinguibles en un desplegable. Las reglas viven aquí una sola vez para que
las dos pantallas se comporten igual y digan lo mismo.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from . import config


class NombreInvalido(ValueError):
    """El nombre propuesto no sirve (vacío, reservado o repetido)."""


def slug(nombre: str) -> str:
    """Nombre escrito por el usuario -> nombre de carpeta seguro.

    El resultado acaba siendo un directorio y un parámetro de URL, así que se
    reduce a ASCII, minúsculas, dígitos y guiones: «Con la Bundesliga» ->
    `con-la-bundesliga`. Devuelve cadena vacía si no queda nada utilizable, y es
    quien llama el que decide qué hacer con eso.
    """
    plano = unicodedata.normalize("NFKD", nombre)
    plano = plano.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", plano)).strip("-")


def preparar(nombre: str, ocupados: Iterable[str], singular: str) -> tuple[str, str]:
    """Valida un nombre nuevo y devuelve (nombre limpio, slug).

    `ocupados` son los slugs que ya existen y `singular` cómo se llama en los
    mensajes lo que se está creando («modelo», «conjunto de datos»).
    """
    limpio = " ".join(nombre.split())[: config.MAX_LARGO_NOMBRE]
    destino = slug(limpio)
    if not destino:
        raise NombreInvalido(
            f"Ponle un nombre al {singular} (letras o números; los acentos y "
            "signos se simplifican para el nombre de la carpeta)."
        )
    if destino == config.VARIANTE_BASE:
        raise NombreInvalido(
            f"«{config.NOMBRE_BASE}» es el {singular} de fábrica: elige otro nombre."
        )
    if destino in set(ocupados):
        raise NombreInvalido(f"Ya hay un {singular} llamado «{limpio}».")
    return limpio, destino

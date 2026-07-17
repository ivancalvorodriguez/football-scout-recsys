"""Utilidades comunes a los scripts de consulta (`probar`, `comparar`, `build`).

Centraliza lo que antes estaba duplicado entre ellos: la configuracion de la
consola y la resolucion de una entidad por nombre (normalizando mayusculas y
acentos, con coincidencia exacta primero y parcial despues).

La resolucion lanza excepciones de dominio (`EntidadNoEncontrada`,
`EntidadAmbigua`, ambas `LookupError`) en vez de `SystemExit`: asi cada script
decide si abortar (consulta de una sola entidad) o saltarse la referencia y
seguir (comparador sobre una lista).
"""

from __future__ import annotations

import sys
import unicodedata


class ErrorResolucion(LookupError):
    """Base de los errores al resolver una entidad por nombre."""


class EntidadNoEncontrada(ErrorResolucion):
    """Ningun nombre del modelo contiene el texto buscado."""


class EntidadAmbigua(ErrorResolucion):
    """Varios nombres coinciden y ninguno es una coincidencia exacta."""


def configurar_consola() -> None:
    """Fuerza UTF-8 en stdout.

    La consola de Windows (cp1252) no imprime nombres con caracteres fuera de
    latin-1 (p. ej. la 'ğ' de algunos jugadores) y aborta o produce mojibake.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalizar(nombre: str) -> str:
    """Nombre -> forma comparable: sin acentos, en minusculas y sin espacios."""
    sin_acentos = (
        unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    )
    return sin_acentos.lower().strip()


def resolver(nombre: str, entity_names: list[str]) -> int:
    """Indice de la entidad cuyo nombre coincide con `nombre`.

    Primero busca una coincidencia exacta (normalizada); si no la hay, acepta una
    unica coincidencia parcial. Varias coincidencias parciales son ambiguas: es
    preferible fallar a elegir en silencio, porque en scouting devolver el
    jugador equivocado pasa desapercibido.
    """
    objetivo = normalizar(nombre)
    nombres_norm = [normalizar(n) for n in entity_names]
    for i, n in enumerate(nombres_norm):
        if n == objetivo:
            return i
    candidatos = [i for i, n in enumerate(nombres_norm) if objetivo in n]
    if len(candidatos) == 1:
        return candidatos[0]
    if not candidatos:
        raise EntidadNoEncontrada(
            f"No se encontro ninguna entidad que contenga {nombre!r}."
        )
    opciones = ", ".join(entity_names[i] for i in candidatos[:10])
    sufijo = " ..." if len(candidatos) > 10 else ""
    raise EntidadAmbigua(f"Ambiguo {nombre!r}; coincide con: {opciones}{sufijo}")

"""Catálogo de las 25 posiciones de StatsBomb para dibujarlas sobre un campo.

`src/extraccion/config.py` define el catálogo canónico (`POSITIONS_25`) con el
nombre en inglés de StatsBomb, que es el que viaja por la BD y por los nombres de
columna `pos_*`. Aquí se le añade lo que solo necesita la interfaz: nombre en
español, rol grueso (para colorear) y una coordenada donde pintarlo.

**El campo del dibujo es un esquema, no el sistema de coordenadas del proyecto.**
Las métricas geométricas usan StatsBomb 120x80 atacando hacia x=120; este módulo
usa un lienzo propio de 100x100 en orientación vertical, que es como se leen las
alineaciones: portería propia abajo (y=100), ataque hacia arriba (y=0), y la
banda izquierda del jugador a la izquierda de quien mira. Confundir ambos
sistemas no rompe nada porque de aquí no sale ninguna métrica: solo píxeles.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.extraccion.config import POSITIONS_25, position_slug


@dataclass(frozen=True)
class Posicion:
    """Una posición del catálogo, lista para pintar."""

    slug: str        # 'pos_left_back', igual que la columna del modelo
    nombre: str      # nombre StatsBomb original ('Left Back')
    etiqueta: str    # nombre en español para la interfaz
    rol: str         # GK | DEF | MID | ATT
    x: float         # 0 (banda izquierda) .. 100 (banda derecha)
    y: float         # 0 (portería rival) .. 100 (portería propia)


# (nombre StatsBomb, etiqueta, rol, x, y). El orden es el de `POSITIONS_25`.
_CATALOGO: tuple[tuple[str, str, str, float, float], ...] = (
    ("Goalkeeper",                  "Portero",                    "GK",  50.0, 92.0),
    ("Right Back",                  "Lateral derecho",            "DEF", 84.0, 74.0),
    ("Right Center Back",           "Central derecho",            "DEF", 65.0, 80.0),
    ("Center Back",                 "Central",                    "DEF", 50.0, 81.0),
    ("Left Center Back",            "Central izquierdo",          "DEF", 35.0, 80.0),
    ("Left Back",                   "Lateral izquierdo",          "DEF", 16.0, 74.0),
    ("Right Wing Back",             "Carrilero derecho",          "DEF", 87.0, 62.0),
    ("Left Wing Back",              "Carrilero izquierdo",        "DEF", 13.0, 62.0),
    ("Right Defensive Midfield",    "Pivote derecho",             "MID", 63.0, 64.0),
    ("Center Defensive Midfield",   "Pivote",                     "MID", 50.0, 66.0),
    ("Left Defensive Midfield",     "Pivote izquierdo",           "MID", 37.0, 64.0),
    ("Right Midfield",              "Medio derecho",              "MID", 84.0, 48.0),
    ("Right Center Midfield",       "Interior derecho",           "MID", 63.0, 50.0),
    ("Center Midfield",             "Mediocentro",                "MID", 50.0, 51.0),
    ("Left Center Midfield",        "Interior izquierdo",         "MID", 37.0, 50.0),
    ("Left Midfield",               "Medio izquierdo",            "MID", 16.0, 48.0),
    ("Right Wing",                  "Extremo derecho",            "ATT", 85.0, 28.0),
    ("Right Attacking Midfield",    "Mediapunta derecho",         "MID", 63.0, 34.0),
    ("Center Attacking Midfield",   "Mediapunta",                 "MID", 50.0, 35.0),
    ("Left Attacking Midfield",     "Mediapunta izquierdo",       "MID", 37.0, 34.0),
    ("Left Wing",                   "Extremo izquierdo",          "ATT", 15.0, 28.0),
    ("Right Center Forward",        "Delantero derecho",          "ATT", 62.0, 15.0),
    ("Center Forward",              "Delantero centro",           "ATT", 50.0, 12.0),
    ("Left Center Forward",         "Delantero izquierdo",        "ATT", 38.0, 15.0),
    ("Secondary Striker",           "Segundo delantero",          "ATT", 50.0, 24.0),
)

POSICIONES: tuple[Posicion, ...] = tuple(
    Posicion(slug=position_slug(nombre), nombre=nombre, etiqueta=etiqueta,
             rol=rol, x=x, y=y)
    for nombre, etiqueta, rol, x, y in _CATALOGO
)

POR_SLUG: dict[str, Posicion] = {p.slug: p for p in POSICIONES}
POR_NOMBRE: dict[str, Posicion] = {p.nombre: p for p in POSICIONES}

ETIQUETA_ROL = {
    "GK": "Portería",
    "DEF": "Defensa",
    "MID": "Centro del campo",
    "ATT": "Ataque",
}


def por_nombre(nombre: str) -> Posicion | None:
    """Posición a partir del nombre StatsBomb tal cual viene de la BD."""
    return POR_NOMBRE.get(nombre)


def etiqueta_de_slug(slug: str) -> str:
    """Etiqueta en español de una columna `pos_*`.

    Si el slug no está en el catálogo (una posición nueva de StatsBomb, o un
    nombre de columna ajeno) se devuelve una versión legible del propio slug en
    vez de fallar: es texto de interfaz, no una métrica.
    """
    posicion = POR_SLUG.get(slug)
    if posicion is not None:
        return posicion.etiqueta
    return slug.removeprefix("pos_").replace("_", " ").capitalize()


def _comprobar_catalogo() -> None:
    """Falla al importar si este catálogo y el de extracción se desalinean.

    Las 25 posiciones se declaran en dos sitios (allí el dato, aquí el dibujo);
    si StatsBomb añadiese una o cambiase un nombre, es mejor romper en el
    arranque que pintar un campo al que le faltan jugadores en silencio.
    """
    esperadas = set(POSITIONS_25.values())
    presentes = {p.nombre for p in POSICIONES}
    if esperadas != presentes:
        faltan = esperadas - presentes
        sobran = presentes - esperadas
        raise RuntimeError(
            "El catálogo de posiciones de la app no coincide con "
            f"extraccion.config.POSITIONS_25 (faltan: {sorted(faltan)}; "
            f"sobran: {sorted(sobran)})."
        )


_comprobar_catalogo()

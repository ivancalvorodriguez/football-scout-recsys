"""App web Flask del recomendador por similitud.

Capa de presentación sobre los modelos ya entrenados por `src.similitud`: NO
entrena y NO recalcula ninguna recomendación. Lee los artefactos de
`outputs/modelo/` (`.npz` + `.json`) y sirve el top-k por nombre. De la BD solo
lee, y solo contexto para leer mejor los resultados (nombres de liga, equipo del
jugador, minutos por posición); si no está, la app funciona sin esos bloques.

Estructura del paquete:

- `config`     — constantes de la app (rutas, defaults, límites).
- `catalogo`   — descubre qué modelos hay en disco y los cachea al cargarlos,
                 junto con su perfil por fases.
- `fases`      — taxonomía de fases de juego y geometría del radar.
- `glosario`   — nombres legibles de las métricas.
- `posiciones` — las 25 posiciones de StatsBomb, colocadas sobre un campo.
- `ligas`      — nombres de competición-temporada (desde la BD, opcional).
- `contexto`   — equipo y minutos por posición (desde la BD, opcional).
- `servicio`   — dominio: buscar entidades, armar el top-k explicado y las fichas.
                 No importa Flask (es testeable sin cliente HTTP).
- `rutas`      — blueprint con las vistas HTML y la API JSON.
- `__main__`   — `python -m src.app` para levantar el servidor de desarrollo.

La factoría `crear_app` es el único punto de entrada público.
"""

from __future__ import annotations

from .factoria import crear_app

__all__ = ["crear_app"]

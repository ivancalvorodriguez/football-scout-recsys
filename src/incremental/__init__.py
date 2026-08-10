"""Incorporacion incremental de partidos y reentrenamiento con warm start.

Dos piezas encadenadas, cada una con su punto de entrada:

- `ingesta`: mete en la BD un PAQUETE de partidos nuevos (formato StatsBomb, ver
  `paquete` y `docs/incremental.md`) reutilizando el motor de `src.extraccion`,
  e informa de que jugadores/equipos/ligas aparecen por primera vez.
- `reentrenar`: reconstruye los modelos de `src.similitud` incluyendo esas
  entidades nuevas, reaprovechando el ajuste anterior (warm start) en vez de
  partir de cero.

El contrato entre ambas es la propia BD mas el `estado` warm cacheado junto a
los artefactos de modelo.
"""

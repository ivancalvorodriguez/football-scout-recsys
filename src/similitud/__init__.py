"""Modelo de recomendación por similitud para scouting (SLIM).

Dos formulaciones fieles al análisis de `docs/como_usar_slim.pdf`, en ficheros
independientes para poder observar sus diferencias:

- `formulacion2`: SLIM instancia-instancia (partido-partido) + agregación de la
  matriz de pesos aprendida `W` a nivel entidad. Fiel a la letra de la
  restricción ("todas las filas, sin colapsar; la agregación cae sobre W").
- `formulacion5`: similitud distribucional entre las nubes de observaciones de
  cada entidad (MMD / OT) y despues EASE (SLIM de forma cerrada) como capa de
  re-ranking aprendido.

Puntos de entrada:

- `python -m src.similitud.build`   — construye y guarda los modelos.
- `python -m src.similitud.probar`  — consulta el top-k de una entidad por nombre.
- `python -m src.similitud.comparar` — compara los top-k entre modos de
  normalizacion (por_liga vs global) y genera CSV + figuras.
"""

from __future__ import annotations

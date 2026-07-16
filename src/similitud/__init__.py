"""Modelo de recomendación por similitud para scouting (SLIM).

Dos formulaciones fieles al análisis de `docs/como_usar_slim.pdf`, en ficheros
independientes para poder observar sus diferencias:

- `formulacion2`: SLIM instancia-instancia (partido-partido) + agregación de la
  matriz de pesos aprendida `W` a nivel entidad. Fiel a la letra de la
  restricción ("todas las filas, sin colapsar; la agregación cae sobre W").
- `formulacion5`: similitud distribucional entre las nubes de observaciones de
  cada entidad (MMD / OT) y despues EASE (SLIM de forma cerrada) como capa de
  re-ranking aprendido.

Punto de entrada de construccion: `python -m src.similitud.build`.
Script de prueba independiente: `python -m src.similitud.probar`.
"""

from __future__ import annotations

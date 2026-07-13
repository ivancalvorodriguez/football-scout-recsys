"""Paquete raíz del sistema de recomendación para scouting futbolístico.

Subpaquetes:
- `extraccion`  extracción de estadísticas por partido desde StatsBomb Open
                Data hacia SQLite (ver `src/extraccion/__init__.py`).
- `exploracion` scripts puntuales de exploración de datos, sin depender del
                pipeline de extracción.
"""

"""Paquete de extracción de estadísticas por partido a partir de StatsBomb Open Data.

Módulos:
- `config`        constantes de campo, catálogo de posiciones y rejilla xT.
- `statsbomb_io`  lectura de los JSON y listado de competiciones/temporadas/partidos.
- `selection`     selección interactiva (por nombre) y filtrado por % de jornadas.
- `features`      utilidades compartidas sobre eventos (geometría, pases, posesiones, xT).
- `minutes`       minutos jugados y posiciones por jugador en un partido.
- `player_stats`  métricas por (jugador, partido).
- `team_stats`    métricas por (equipo, partido).
- `database`      esquema SQLite y escritura (dimensiones + hechos, con UPSERT).
- `extract`       orquestación y punto de entrada (CLI).
"""

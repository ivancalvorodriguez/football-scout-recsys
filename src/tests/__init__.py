"""Suite de pruebas del proyecto (pytest).

Organizada por paquete de produccion y, dentro de cada uno, por tipo de prueba:

- ``unit/``: una funcion o modulo aislado, sin E/S (salvo `tmp_path`).
- ``integracion/``: varios modulos colaborando (p. ej. eventos -> metricas ->
  SQLite), con la BD real en un directorio temporal.
- ``e2e/``: los puntos de entrada lanzados como subproceso, igual que un usuario.

Los paquetes llevan `__init__.py` a proposito: hay ficheros con el mismo nombre
en varios subdirectorios (`test_features.py` existe para `extraccion` y para
`similitud`) y sin paquete pytest no podria importarlos a la vez.

Ninguna prueba lee `open-data/`: es un dataset externo vendored, esta en
`.gitignore` y no existe en un checkout limpio. Todo dato de entrada se fabrica
en `src/tests/factories.py`.
"""

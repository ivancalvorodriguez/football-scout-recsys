"""Evaluacion sin ground truth de los modelos de similitud (`src/similitud`).

Implementa el protocolo priorizado de `docs/Como_evaluar.pdf` sobre los
artefactos ya construidos en `outputs/modelo/` (2 formulaciones x 2 entidades x 2
normalizaciones) y sobre los modelos "virtuales" que solo existen para evaluar
(mitades pares/impares de cada entidad para la auto-similitud, remuestreos
bootstrap para la estabilidad). Escribe resultados en `outputs/evaluacion/`.
"""

"""Script de prueba del modelo guardado: consulta top-k por NOMBRE.

    python -m src.similitud.probar --nombre "Lionel Andres Messi Cuccittini"
    python -m src.similitud.probar --entidad equipo --nombre "Barcelona"
    python -m src.similitud.probar --formulacion 2 --nombre "..." --k 8
    python -m src.similitud.probar --formulacion 2 --normalizacion global --nombre "..."
    python -m src.similitud.probar --distancia coseno --nombre "..."

Carga lo guardado en `outputs/modelo/`, resuelve la entidad por nombre (coincidencia
parcial, sin distinguir mayusculas) e imprime las k mas similares con su puntuacion
y las metricas que mas separan a cada candidato de la entidad de referencia (para
interpretar el porque). Funciona para jugador->jugadores y equipo->equipos.

Con `--normalizacion` se elige el modo de estandarizacion del modelo a cargar
(`por_liga` por defecto; `global` para el modelo entrenado sin normalizacion por
liga), y con `--distancia` su geometria (`euclidea` por defecto, que es la del
grueso de lo construido; los artefactos SERVIDOS son `mahalanobis` y viven en
ficheros con sufijo, asi que para consultarlos hay que pedirla:
`--formulacion 5 --distancia mahalanobis`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config, distancias
from .consulta import ErrorResolucion, configurar_consola, resolver
from .modelo import ModeloSimilitud, cargar_modelo, top_k

configurar_consola()


def _metricas_clave(
    modelo: ModeloSimilitud, i: int, j: int, n: int = 4
) -> list[tuple[str, float, float]]:
    """Features (en z-score de liga) donde referencia y candidato mas coinciden.

    Devuelve las n features con menor |diferencia| que ademas son notables
    (|valor| alto) en la referencia: ayudan a explicar el parecido.
    """
    ref = modelo.feat_display[i]
    cand = modelo.feat_display[j]
    relevancia = np.abs(ref) - np.abs(ref - cand)  # notable y parecido
    orden = np.argsort(relevancia)[::-1][:n]
    return [(modelo.feat_names[f], float(ref[f]), float(cand[f])) for f in orden]


def _consultar(
    model_dir: Path, formulacion: str, entidad: str, normalizacion: str,
    nombre: str, k: int, distancia: str = distancias.POR_DEFECTO,
) -> None:
    modelo = cargar_modelo(model_dir, formulacion, entidad,
                           normalizacion=normalizacion, distancia=distancia)
    try:
        i = resolver(nombre, modelo.entity_names)
    except ErrorResolucion as e:
        raise SystemExit(str(e)) from e
    print(f"\n=== Formulacion {formulacion} | {entidad} | {normalizacion} "
          f"| distancia {distancia} ===")
    print(f"Referencia: {modelo.entity_names[i]}")
    print(f"({modelo.meta.get('descripcion','')})")
    print(f"Top-{k} mas similares:")
    for rango, (j, score) in enumerate(top_k(modelo, i, k), 1):
        print(f"  {rango:2d}. {modelo.entity_names[j]:<40s}  score={score:+.4f}")
        claves = _metricas_clave(modelo, i, j)
        detalle = "; ".join(
            f"{nm}(ref {rv:+.2f} vs {cv:+.2f})" for nm, rv, cv in claves
        )
        print(f"       coinciden en: {detalle}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Consulta top-k de un modelo de similitud.")
    p.add_argument("--modelo", type=Path, default=config.DEFAULT_MODEL_DIR,
                   help="Directorio con los modelos guardados.")
    p.add_argument("--formulacion", choices=["2", "5"], default="2")
    p.add_argument("--entidad", choices=["jugador", "equipo"], default="jugador")
    p.add_argument("--normalizacion", choices=["por_liga", "global"], default="por_liga",
                   help="Modo de normalizacion del modelo a cargar.")
    p.add_argument("--distancia", choices=list(distancias.DISTANCIAS_VALIDAS),
                   default=distancias.POR_DEFECTO,
                   help="Geometria del modelo a cargar. La euclidea es la unica "
                        "sin sufijo en el nombre del fichero; el resto solo existe "
                        "si se ha construido. Los artefactos servidos son "
                        "mahalanobis.")
    p.add_argument("--nombre", required=True, help="Nombre (o parte) de la entidad.")
    p.add_argument("--k", type=int, default=config.DEFAULT_TOP_K)
    args = p.parse_args(argv)
    _consultar(
        args.modelo, args.formulacion, args.entidad, args.normalizacion,
        args.nombre, args.k, distancia=args.distancia,
    )


if __name__ == "__main__":
    main()

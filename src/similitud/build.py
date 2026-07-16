"""Punto de entrada: construye y guarda los modelos de similitud.

    python -m src.similitud.build [--db RUTA] [--out DIR]
                                  [--formulacion {2,5,ambas}]
                                  [--entidad {jugador,equipo,ambas}]
                                  [--normalizacion {por_liga,global,ambas}]

Encadena: (1) carga desde SQLite -> (2) capa de features por-partido (derivadas +
z-score por liga o global) -> (3)/(4) ajuste de la formulacion elegida ->
guardado en `outputs/modelo/`. No modifica la BD (solo lectura).

Cada modo de normalizacion genera un artefacto independiente con sufijo en el
nombre (``_por_liga`` o ``_global``); con `--normalizacion ambas` se generan los
dos, lo que permite al comparador cruzarlos sin reentrenar.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config, data, features, formulacion2, formulacion5

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_FORMULACIONES = {"2": formulacion2, "5": formulacion5}
_NORMALIZACIONES_VALIDAS = ("por_liga", "global")


def _construir_uno(
    db_path: Path,
    out_dir: Path,
    formulacion: str,
    entidad: str,
    normalizacion: str,
) -> None:
    t0 = time.perf_counter()
    df = data.cargar(db_path, entidad)
    mf = features.construir(df, entidad, normalizacion=normalizacion)
    n_obs = mf.X.shape[0]
    n_ent = len(set(mf.entity_id.tolist()))
    # Verificacion del principio central: ninguna entidad colapsa a un punto.
    if n_obs < n_ent:
        raise RuntimeError("colapso detectado: menos observaciones que entidades")
    print(
        f"[{entidad} | formulacion {formulacion} | {normalizacion}] "
        f"{n_obs} observaciones -> {n_ent} entidades, {mf.X.shape[1]} features"
    )

    modelo = _FORMULACIONES[formulacion].construir(mf, entidad, normalizacion)
    path = modelo.guardar(out_dir)
    dt = time.perf_counter() - t0
    print(f"    guardado en {path}  ({dt:.1f}s)  meta={modelo.meta.get('descripcion','')}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Construye los modelos de similitud (SLIM).")
    p.add_argument("--db", type=Path, default=config.DEFAULT_DB_PATH,
                   help="Ruta a la BD SQLite (solo lectura).")
    p.add_argument("--out", type=Path, default=config.DEFAULT_MODEL_DIR,
                   help="Directorio de salida de los modelos.")
    p.add_argument("--formulacion", choices=["2", "5", "ambas"], default="ambas",
                   help="Formulacion SLIM a construir.")
    p.add_argument("--entidad", choices=["jugador", "equipo", "ambas"], default="ambas",
                   help="Tipo de entidad a modelar.")
    p.add_argument("--normalizacion",
                   choices=[*_NORMALIZACIONES_VALIDAS, "ambas"],
                   default="ambas",
                   help=("Modo de estandarizacion: 'por_liga' (z-score por competicion-temporada, "
                         "el comportamiento previo) o 'global' (z-score con todo el dataset, "
                         "habilita comparacion inter-liga)."))
    args = p.parse_args(argv)

    formulaciones = ["2", "5"] if args.formulacion == "ambas" else [args.formulacion]
    entidades = ["jugador", "equipo"] if args.entidad == "ambas" else [args.entidad]
    normalizaciones = (
        list(_NORMALIZACIONES_VALIDAS)
        if args.normalizacion == "ambas"
        else [args.normalizacion]
    )

    for entidad in entidades:
        for formulacion in formulaciones:
            for normalizacion in normalizaciones:
                _construir_uno(args.db, args.out, formulacion, entidad, normalizacion)


if __name__ == "__main__":
    main()

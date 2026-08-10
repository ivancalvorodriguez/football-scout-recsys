"""Punto de entrada: reentrena los modelos incluyendo lo nuevo, con warm start.

    python -m src.incremental.reentrenar [--db RUTA] [--modelos DIR] [--out DIR]
                                         [--servibles]
                                         [--formulacion {2,5,ambas}]
                                         [--entidad {jugador,equipo,ambas}]
                                         [--normalizacion {por_liga,global,ambas}]
                                         [--frio] [--verificar]

Despues de `src.incremental.ingesta`, la BD tiene jugadores y equipos que ningun
modelo conoce. Este comando reconstruye los artefactos de `outputs/modelo/` sobre
la BD completa, de modo que la S resultante incluye a los nuevos: aparecen en sus
propias recomendaciones Y como candidatos en las de los antiguos.

No es un ajuste desde cero. Dos mecanismos lo evitan:

1. **Estadisticas de normalizacion congeladas.** El z-score se aplica con las
   mu/sd guardadas en el modelo anterior, asi que una observacion antigua produce
   el MISMO vector que antes. Sin esto, meter una liga mas moveria a todas las
   entidades y no habria nada que reaprovechar (ni motivo para que las
   recomendaciones de un jugador que no ha jugado cambien).
2. **Warm start.** Cada formulacion arranca del estado que dejo el ajuste
   anterior (`<modelo>.warm.npz`): la W de la Formulacion 2 como punto de partida
   del coordinate descent, y el kernel congelado / los costes OT ya calculados en
   la Formulacion 5. Ver `src.similitud.warm`.

`--frio` desactiva ambos (util para comparar) y `--verificar` ajusta ademas en
frio y mide cuanto se separan las dos S: es la comprobacion de que el warm start
ahorra tiempo sin cambiar el resultado.

`--servibles` reentrena exactamente la pareja que sirve la app
(`config.MODELOS_SERVIBLES`: jugador con la F5 y equipo con la F2, las dos con
z-score global) en vez del producto cartesiano de las tres flags. Es lo que lanza
la seccion «Datos y modelos» de la interfaz: alli un modelo es una pareja
jugador+equipo con nombre, no una combinacion metodologica suelta.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.similitud import data, features, formulacion2, formulacion5, warm
from src.similitud.consulta import configurar_consola
from src.similitud.features import NORMALIZACIONES_VALIDAS, EstadisticasNorm
from src.similitud.modelo import ModeloSimilitud

from . import config

configurar_consola()

_FORMULACIONES = {"2": formulacion2, "5": formulacion5}


# --- Progreso -----------------------------------------------------------------
def _barra(etiqueta: str, cada: int = 5):
    """Callable progreso(hecho, total) que imprime cada `cada` por ciento."""
    ultimo = {"pct": -1}

    def progreso(hecho: int, total: int) -> None:
        pct = int(100 * hecho / max(1, total))
        if pct >= ultimo["pct"] + cada or hecho == total:
            ultimo["pct"] = pct
            print(f"\r    {etiqueta}: {pct:3d}%  ({hecho}/{total})", end="", flush=True)
            if hecho == total:
                print()

    return progreso


# --- Estado previo ------------------------------------------------------------
def _cargar_estado(model_dir: Path, formulacion: str, entidad: str,
                   normalizacion: str) -> warm.EstadoWarm | None:
    ruta = warm.ruta_estado(model_dir, formulacion, entidad, normalizacion)
    if not ruta.exists():
        return None
    try:
        return warm.EstadoWarm.cargar(ruta)
    except (KeyError, ValueError, OSError) as exc:
        print(f"    [aviso] estado previo ilegible ({exc}); se ajusta en frio")
        return None


def _cargar_estadisticas(model_dir: Path, formulacion: str, entidad: str,
                         normalizacion: str) -> EstadisticasNorm | None:
    """mu/sd del modelo anterior, si las lleva.

    Los artefactos construidos antes de existir el flujo incremental no las
    tienen: en ese caso no hay nada que congelar y se recalculan, lo que mueve
    todos los vectores y deja el warm start en una simple inicializacion.
    """
    path = Path(model_dir) / f"formulacion{formulacion}_{entidad}_{normalizacion}.json"
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
    except (json.JSONDecodeError, OSError):
        return None
    return EstadisticasNorm.desde_dict(meta.get("estadisticas_normalizacion"))


# --- Comparacion warm vs frio -------------------------------------------------
def _comparar(caliente: ModeloSimilitud, frio: ModeloSimilitud) -> dict:
    """Cuanto se separan dos ajustes del mismo problema (mismo orden de entidades)."""
    if not np.array_equal(caliente.entity_ids, frio.entity_ids):
        raise RuntimeError(
            "los dos ajustes no indexan las mismas entidades: la comparacion no "
            "significaria nada"
        )
    A, B = caliente.S, frio.S
    escala = float(np.abs(B).max()) or 1.0
    dif = float(np.abs(A - B).max())
    # Coincidencia del top-10 servible, que es lo que de verdad ve el usuario.
    k = min(10, A.shape[0] - 1)
    coincidencias = 0
    for i in range(A.shape[0]):
        fa, fb = A[i].copy(), B[i].copy()
        fa[i] = fb[i] = -np.inf
        coincidencias += len(
            set(np.argsort(fa)[::-1][:k]) & set(np.argsort(fb)[::-1][:k])
        )
    return {
        "max_dif_absoluta": dif,
        "max_dif_relativa": dif / escala,
        "solape_top10": coincidencias / max(1, A.shape[0] * k),
    }


# --- Reentrenamiento de un modelo --------------------------------------------
def reentrenar_uno(
    db_path: Path,
    model_dir: Path,
    out_dir: Path,
    formulacion: str,
    entidad: str,
    normalizacion: str,
    frio: bool = False,
    verificar: bool = False,
    refrescar_kernel: bool = False,
) -> dict:
    """Reentrena un (formulacion, entidad, normalizacion) y devuelve su informe."""
    etiqueta = f"formulacion {formulacion} | {entidad} | {normalizacion}"
    print(f"\n[{etiqueta}]")
    t0 = time.perf_counter()

    previo = None if frio else _cargar_estado(model_dir, formulacion, entidad, normalizacion)
    congeladas = None if frio else _cargar_estadisticas(
        model_dir, formulacion, entidad, normalizacion)
    if not frio:
        if previo is None:
            print("    sin estado previo: ajuste en frio (se guardara para la proxima)")
        if congeladas is None:
            print("    [aviso] el modelo anterior no guarda mu/sd: se recalculan, "
                  "asi que los vectores antiguos se moveran")

    df = data.cargar(db_path, entidad)
    mf = features.construir(df, entidad, normalizacion=normalizacion,
                            estadisticas=congeladas)
    n_ent = len(set(mf.entity_id.tolist()))
    print(f"    {mf.X.shape[0]} observaciones -> {n_ent} entidades, "
          f"{mf.X.shape[1]} features")

    modulo = _FORMULACIONES[formulacion]
    t_ajuste = time.perf_counter()
    modelo, estado = modulo.construir_con_estado(
        mf, entidad, normalizacion, previo=previo, progreso=_barra("ajuste"),
        congelar_kernel=not refrescar_kernel)
    ajuste = time.perf_counter() - t_ajuste
    segundos = time.perf_counter() - t0

    info_warm = modelo.meta.get("warm", {})
    if info_warm.get("motivo"):
        print(f"    [aviso] no se pudo reutilizar: {info_warm['motivo']}")
    print(f"    reutilizadas {info_warm.get('observaciones_reutilizadas', 0)}"
          f"/{info_warm.get('observaciones', 0)} observaciones "
          f"({info_warm.get('observaciones_intactas', 0)} intactas), "
          f"{info_warm.get('entidades_nuevas', 0)} entidades nuevas")

    path = modelo.guardar(out_dir)
    estado.guardar(warm.ruta_estado(out_dir, formulacion, entidad, normalizacion))
    print(f"    guardado en {path}  ({segundos:.1f}s, de los cuales {ajuste:.1f}s de ajuste)")

    informe = {
        "formulacion": formulacion,
        "entidad": entidad,
        "normalizacion": normalizacion,
        "segundos": round(segundos, 1),
        "segundos_ajuste": round(ajuste, 1),
        "n_observaciones": int(mf.X.shape[0]),
        "n_entidades": n_ent,
        "estadisticas_congeladas": congeladas is not None,
        "warm": info_warm,
    }

    if verificar:
        print("    verificando contra un ajuste en frio...")
        # Se reutiliza el `df` ya cargado y se cronometra SOLO el ajuste, para
        # que el tiempo sea comparable con `segundos_ajuste` de arriba.
        mf_frio = features.construir(df, entidad, normalizacion=normalizacion)
        t1 = time.perf_counter()
        modelo_frio, _ = modulo.construir_con_estado(mf_frio, entidad, normalizacion)
        ajuste_frio = time.perf_counter() - t1
        informe["frio"] = {
            "segundos_ajuste": round(ajuste_frio, 1),
            "ahorro": round(1.0 - ajuste / ajuste_frio, 3) if ajuste_frio else None,
            **_comparar(modelo, modelo_frio),
        }
        d = informe["frio"]
        print(f"    frio: {d['segundos_ajuste']}s de ajuste (warm {ajuste:.1f}s, "
              f"ahorro {100 * (d['ahorro'] or 0):.0f}%) | "
              f"max|dS| = {d['max_dif_absoluta']:.2e} "
              f"({d['max_dif_relativa']:.2e} relativo) | "
              f"solape top-10 = {d['solape_top10']:.3f}")
    return informe


# --- CLI ----------------------------------------------------------------------
def _combinaciones(args) -> list[tuple[str, str, str]]:
    """(entidad, formulacion, normalizacion) que hay que reentrenar.

    Con `--servibles` es la pareja de `config.MODELOS_SERVIBLES` —una entidad,
    una formulacion, una normalizacion— y no el producto cartesiano: cada entidad
    tiene su propia formulacion, asi que no se puede expresar con las tres flags.
    """
    if args.servibles:
        return [(e, f, n) for e, (f, n) in config.MODELOS_SERVIBLES.items()]
    formulaciones = ["2", "5"] if args.formulacion == "ambas" else [args.formulacion]
    entidades = ["jugador", "equipo"] if args.entidad == "ambas" else [args.entidad]
    normalizaciones = (
        list(NORMALIZACIONES_VALIDAS) if args.normalizacion == "ambas"
        else [args.normalizacion]
    )
    return [
        (entidad, formulacion, normalizacion)
        for entidad in entidades
        for formulacion in formulaciones
        for normalizacion in normalizaciones
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Reentrena los modelos de similitud incluyendo las entidades nuevas.")
    p.add_argument("--db", type=Path, default=config.DEFAULT_DB_PATH,
                   help="BD SQLite (solo lectura).")
    p.add_argument("--modelos", type=Path, default=config.DEFAULT_MODEL_DIR,
                   help="Directorio de los modelos ANTERIORES (de donde sale el estado "
                        "warm y las mu/sd congeladas).")
    p.add_argument("--out", type=Path, default=None,
                   help="Directorio de salida. Por defecto, el mismo que --modelos "
                        "(reentrenamiento en el sitio).")
    p.add_argument("--servibles", action="store_true",
                   help="Reentrena solo la pareja que sirve la app "
                        "(config.MODELOS_SERVIBLES) e ignora las tres flags "
                        "siguientes.")
    p.add_argument("--formulacion", choices=["2", "5", "ambas"], default="ambas")
    p.add_argument("--entidad", choices=["jugador", "equipo", "ambas"], default="ambas")
    p.add_argument("--normalizacion", choices=[*NORMALIZACIONES_VALIDAS, "ambas"],
                   default="ambas")
    p.add_argument("--frio", action="store_true",
                   help="Ignora el estado previo y las mu/sd guardadas: ajuste desde cero.")
    p.add_argument("--verificar", action="store_true",
                   help="Ajusta ademas en frio y mide la diferencia entre las dos S "
                        "(comprueba que el warm start no cambia el resultado).")
    p.add_argument("--refrescar-kernel", action="store_true",
                   help="Formulacion 5 con MMD: recalcula el ancho del kernel RBF con "
                        "todos los datos en vez de heredarlo. Da el mismo resultado que "
                        "un ajuste en frio, pero mueve tambien a las entidades que no "
                        "han jugado.")
    args = p.parse_args(argv)

    out_dir = args.out or args.modelos
    if not Path(args.db).exists():
        print(f"No existe la BD {args.db}.", file=sys.stderr)
        return 2
    # Con `--out` a un directorio nuevo (es lo que hace la app al crear un modelo
    # con nombre) el informe final se escribiria antes de que nadie lo cree.
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    combinaciones = _combinaciones(args)

    print(f"BD: {args.db}\nModelos previos: {args.modelos}\nSalida: {out_dir}")
    if args.frio:
        print("Modo FRIO: no se reutiliza nada.")

    informes = []
    t0 = time.perf_counter()
    for entidad, formulacion, normalizacion in combinaciones:
        informes.append(reentrenar_uno(
            args.db, args.modelos, out_dir, formulacion, entidad,
            normalizacion, frio=args.frio, verificar=args.verificar,
            refrescar_kernel=args.refrescar_kernel))

    destino = Path(out_dir) / "reentrenamiento.json"
    destino.write_text(
        json.dumps({"total_segundos": round(time.perf_counter() - t0, 1),
                    "modelos": informes}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nHecho en {time.perf_counter() - t0:.1f}s. Informe: {destino}")
    print("Consulta el resultado con:  python -m src.similitud.probar --nombre \"...\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

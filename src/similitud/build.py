"""Punto de entrada: construye y guarda los modelos de similitud.

    python -m src.similitud.build [--db RUTA] [--out DIR]
                                  [--formulacion {2,5,ambas}]
                                  [--entidad {jugador,equipo,ambas}]
                                  [--normalizacion {por_liga,global,ambas}]
                                  [--distancia {euclidea,mahalanobis,coseno,manhattan}]

Encadena: (1) carga desde SQLite -> (2) capa de features por-partido (derivadas +
z-score por liga o global) -> (3)/(4) ajuste de la formulacion elegida ->
guardado en `outputs/modelo/`. No modifica la BD (solo lectura).

Cada modo de normalizacion genera un artefacto independiente con sufijo en el
nombre (``_por_liga`` o ``_global``); con `--normalizacion ambas` se generan los
dos, lo que permite al comparador cruzarlos sin reentrenar.

Junto a cada artefacto se deja su ESTADO warm (`<stem>.warm.npz`): lo que el
ajuste ha aprendido por debajo de la S servible. `build` siempre ajusta en frio;
ese estado es lo que despues permite a `src.incremental.reentrenar` incorporar
partidos nuevos sin repetir el trabajo. Borrarlo no rompe nada: solo obliga al
siguiente reentrenamiento a arrancar de cero.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import config, data, distancias, features, formulacion2, formulacion5, warm
from .config import hiperparametros_servibles
from .consulta import configurar_consola
from .distancias import DISTANCIAS_VALIDAS
from .features import NORMALIZACIONES_VALIDAS

configurar_consola()

_FORMULACIONES = {"2": formulacion2, "5": formulacion5}


def _construir_uno(
    db_path: Path,
    out_dir: Path,
    formulacion: str,
    entidad: str,
    normalizacion: str,
    excluir_ligas: tuple[str, ...] = (),
    distancia: str = distancias.POR_DEFECTO,
) -> None:
    """Ajusta y guarda un artefacto (+ su estado warm).

    ``excluir_ligas`` deja fuera del ajuste las observaciones de esas
    competiciones (claves `competition-season`), ANTES de estandarizar: la liga
    excluida no entra ni en el modelo ni en las mu/sd del z-score. Lo usa el
    hold-out de la evaluacion (`src.evaluacion.generalizacion`), que necesita un
    modelo que de verdad no haya visto esa liga. Vacio = todo, que es el
    comportamiento normal de `build`.

    Ajusta con la configuracion que haya VIGENTE en `config`, sin tocarla. Esto
    importa: `src.evaluacion.trabajo` construye cada celda del barrido llamando
    aqui dentro de su propio `config_temporal`, asi que si esta funcion
    sobrescribiera algun hiperparametro, el artefacto no seria el que declara su
    huella y la cache del barrido acabaria sirviendo un modelo que no es. Los
    hiperparametros del modelo SERVIDO los pone `main`, un escalon mas arriba.
    """
    t0 = time.perf_counter()
    df = data.cargar(db_path, entidad)
    if excluir_ligas:
        df = df[~df[data.LEAGUE_KEY].astype(str).isin(set(excluir_ligas))]
        df = df.reset_index(drop=True)
        if df.empty:
            raise RuntimeError(
                f"excluir {', '.join(excluir_ligas)} no deja ninguna observacion")
    mf = features.construir(df, entidad, normalizacion=normalizacion)
    n_obs = mf.X.shape[0]
    n_ent = len(set(mf.entity_id.tolist()))
    # Verificacion del principio central: ninguna entidad colapsa a un punto.
    if n_obs < n_ent:
        raise RuntimeError("colapso detectado: menos observaciones que entidades")
    sin = f" (sin {', '.join(excluir_ligas)})" if excluir_ligas else ""
    print(
        f"[{entidad} | formulacion {formulacion} | {normalizacion} | {distancia}]{sin} "
        f"{n_obs} observaciones -> {n_ent} entidades, {mf.X.shape[1]} features"
    )

    modelo, estado = _FORMULACIONES[formulacion].construir_con_estado(
        mf, entidad, normalizacion, distancia=distancia)
    path = modelo.guardar(out_dir)
    estado.guardar(
        warm.ruta_estado(out_dir, formulacion, entidad, normalizacion, distancia))
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
                   choices=[*NORMALIZACIONES_VALIDAS, "ambas"],
                   default="ambas",
                   help=("Modo de estandarizacion: 'por_liga' (z-score por competicion-temporada, "
                         "el comportamiento previo) o 'global' (z-score con todo el dataset, "
                         "habilita comparacion inter-liga)."))
    p.add_argument("--distancia", choices=[*DISTANCIAS_VALIDAS, "todas"],
                   default=distancias.POR_DEFECTO,
                   help=("Distancia entre observaciones con la que se eligen los "
                         "vecinos y se mide el residuo del ajuste. 'euclidea' (por "
                         "defecto) es la de siempre; la que SIRVE la app es manhattan; "
                         "'mahalanobis' esta descartada (su blanqueo deshace la "
                         "ponderacion del bloque pos_*); el resto "
                         "genera artefactos con sufijo, sin tocar los existentes. "
                         "Ojo con 'manhattan' en la F2 de jugador: no tiene la via "
                         "rapida del producto de matrices y el kNN sobre ~50.000 "
                         "observaciones se va a horas."))
    args = p.parse_args(argv)

    formulaciones = ["2", "5"] if args.formulacion == "ambas" else [args.formulacion]
    entidades = ["jugador", "equipo"] if args.entidad == "ambas" else [args.entidad]
    normalizaciones = (
        list(NORMALIZACIONES_VALIDAS)
        if args.normalizacion == "ambas"
        else [args.normalizacion]
    )
    distancias_sel = (
        list(DISTANCIAS_VALIDAS) if args.distancia == "todas" else [args.distancia]
    )

    celdas = [
        (entidad, formulacion, normalizacion, distancia)
        for entidad in entidades
        for formulacion in formulaciones
        for normalizacion in normalizaciones
        for distancia in distancias_sel
    ]
    for entidad, formulacion, normalizacion, distancia in celdas:
        # La celda que SIRVE la app para esa entidad se ajusta con los
        # hiperparametros del artefacto servido y no con los defaults del
        # modulo: jugador y equipo comparten formulacion con valores
        # distintos (ver `config.HIPERPARAMETROS_SERVIBLES`), asi que los
        # defaults solo pueden reproducir uno de los dos. Las demas
        # celdas —el brazo de comparacion del TFG— salen con los defaults.
        # La distancia entra en esa condicion: solo la euclidea reproduce el
        # artefacto servido, asi que un `--distancia manhattan` sale con los
        # defaults del modulo aunque coincida la formulacion y la normalizacion.
        #
        # Va aqui, en el CLI, y no en `_construir_uno`: ahi dentro
        # entraria tambien el barrido, que fija sus propios valores.
        with hiperparametros_servibles(
                entidad, formulacion, normalizacion, distancia) as puestos:
            if puestos:
                print("[celda servible] "
                      + ", ".join(f"{k}={v}" for k, v in puestos.items()))
            _construir_uno(args.db, args.out, formulacion, entidad,
                           normalizacion, distancia=distancia)


if __name__ == "__main__":
    main()

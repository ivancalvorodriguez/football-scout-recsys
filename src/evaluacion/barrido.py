"""Punto de entrada: construye Y evalua un modelo por cada COMBINACION de hiperparametros.

    python -m src.evaluacion.barrido [--db RUTA] [--out DIR]
                                     [--formulaciones 2,5] [--entidades jugador,equipo]
                                     [--fases 0,1,5] [--bootstrap B] [--sin-figuras]
                                     [--rehacer] [--adoptar-existentes]

A diferencia de `src.evaluacion.evaluar` (que exige que los modelos ya esten
construidos en `outputs/modelo/`), este script CONSTRUYE los modelos sobre la
marcha antes de evaluarlos: por cada combinacion de hiperparametros reconstruye
la rejilla (formulacion 2/5 x jugador/equipo x por_liga/global) en su propia
carpeta y despues corre el protocolo de evaluacion. Asi se puede lanzar de cero,
sin `build` previo.

Los hiperparametros que se barren NO se piden por linea de comandos: estan
escritos en `HIPERPARAMETROS`, aqui abajo. El barrido recorre el PRODUCTO
CARTESIANO de esa lista — cada combinacion fija todos los ejes a la vez — y
construye/evalua cada una en `<out>/vNN/`. La linea de comandos no elige valores
de hiperparametro: solo decide QUE modelos entran en la rejilla
(`--formulaciones`, `--entidades`), COMO se evaluan (`--fases`, `--bootstrap`,
`--sin-figuras`) y que hace la caché (`--rehacer`, `--adoptar-existentes`).

Como todo el pipeline lee los valores de `src.similitud.config` en tiempo de
llamada, basta con fijarlos antes de construir/evaluar y restaurarlos despues. El
objetivo es observar el EFECTO de cada hiperparametro comparando las
combinaciones entre si.

**La carpeta de salida es ACUMULATIVA** (ver `registro.py`): relanzar el barrido
sobre la misma `--out` despues de editar `HIPERPARAMETROS` no borra lo anterior.
Cada combinacion se identifica por su CONFIGURACION, no por su posicion en el
producto cartesiano, asi que conserva su nombre `vNN` y su carpeta entre
corridas; las que ya se evaluaron y no vuelven a tocarse mantienen sus metricas
en `barrido_metricas.csv`, y las nuevas se suman. Ampliar la rejilla es
simplemente añadir valores a `HIPERPARAMETROS` y volver a lanzar: el resumen y
las superficies 3D salen con TODOS los puntos acumulados en la carpeta, que es lo
que permite localizar el optimo. Para empezar un barrido limpio, apunta `--out` a
una carpeta nueva.

Salida por combinacion en `<out>/<nombre>/` (CSV + figuras). Los deliverables de
la carpeta son `<out>/resumen_barrido.md` (comparativa de las metricas clave
entre TODAS las combinaciones acumuladas), `<out>/barrido_metricas.csv` (las
mismas metricas en formato largo) y `<out>/combinaciones.json` (el indice
nombre -> configuracion, que es lo que hace estables los nombres).
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.similitud import build as sbuild
from src.similitud import config as scfg

from . import config as ecfg
from . import datos, evaluar, huella, puntuacion, registro
from .evaluar import _fmt


# --------------------------------------------------------------------------- #
# Hiperparametros del barrido (editable): atributo de config -> valores a probar #
# --------------------------------------------------------------------------- #
# El barrido recorre el PRODUCTO CARTESIANO de esta lista, asi que cada eje
# MULTIPLICA el coste: N ejes de 2 valores son 2^N combinaciones, y cada una
# construye (o reutiliza de la caché) su propia rejilla de modelos. Los F2 de
# jugador son los caros (>10 min cada uno), de modo que anadir un eje que los
# afecte se nota en horas. Para barrer un eje concreto sin pagar el resto,
# acota la rejilla con `--formulaciones` / `--entidades`.
#
# Los nombres tienen que existir en `src.similitud.config` (se valida al
# arrancar). Cada entrada es (ATRIBUTO, [valores a probar]).

HIPERPARAMETROS: list[tuple[str, list[object]]] = [
    # Formulacion 2 (SLIM instancia-instancia): penalizacion L1 (dispersion de W)
    # y ridge L2.
    ("F2_L1", [0.45, 0.5, 0.55, 0.625, 0.675]),
    ("F2_BETA", [9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75]),
    # Formulacion 5 (distribucional + EASE): fuerza del re-ranking EASE, barrida
    # en un rango amplio (de casi-sin-regularizar a dominado por el ridge), y
    # dimension del embedding RFF con que se aproxima el MMD del jugador.
    ("F5_EASE_LAMBDA", [0.85, 0.875, 0.9]),
    ("F5_RFF_DIM", [1024, 1152, 1280]),
]

# El bloque de posicion del jugador (25 columnas `pos_*`) NO se barre: forma
# parte del modelo base, z-scoreado con el resto y dividido por sqrt(25) para que
# pese como una sola feature (`config.USE_POSITION_FEATURES` /
# `config.POSITION_SCALING = "zscore_sqrt"`). El resto de features sigue con sus
# dos normalizaciones, `por_liga` y `global`, que son un eje de la rejilla y no
# un hiperparametro. Para volver a contrastar el vector con y sin posicion hay
# que tocar esos dos atributos de `src/similitud/config.py` a mano.


def _valores_declarados() -> dict[str, list[object]]:
    return {attr: valores for attr, valores in HIPERPARAMETROS}


def _validar_hiperparametros() -> None:
    """Falla al arrancar si la lista de arriba tiene un typo o un eje vacio.

    Antes de gastar horas construyendo modelos: un nombre que no existe en
    `src.similitud.config` no sobrescribe nada y el barrido acabaria comparando
    combinaciones identicas sin decirlo.
    """
    vistos: set[str] = set()
    for attr, valores in HIPERPARAMETROS:
        if not hasattr(scfg, attr):
            raise SystemExit(
                f"hiperparametro desconocido en src.similitud.config: {attr!r} "
                "(revisa HIPERPARAMETROS en src/evaluacion/barrido.py)."
            )
        if not valores:
            raise SystemExit(f"el hiperparametro {attr!r} no declara ningun valor.")
        if attr in vistos:
            raise SystemExit(f"el hiperparametro {attr!r} esta declarado dos veces.")
        vistos.add(attr)


@contextmanager
def _config_temporal(valores: dict[str, object]):
    """Fija atributos de `src.similitud.config` y los restaura al salir."""
    previos: dict[str, object] = {}
    for clave, valor in valores.items():
        previos[clave] = getattr(scfg, clave)
        setattr(scfg, clave, valor)
    try:
        yield
    finally:
        for clave, valor in previos.items():
            setattr(scfg, clave, valor)


def _fmt_valores(valores: dict[str, object]) -> str:
    if not valores:
        return "(config por defecto)"
    return ", ".join(f"{k}={v}" for k, v in valores.items())


# --------------------------------------------------------------------------- #
# Producto cartesiano de los hiperparametros                                   #
# --------------------------------------------------------------------------- #

def _ejes_relevantes(
    formulaciones: tuple[str, ...], entidades: tuple[str, ...]
) -> list[str]:
    """Ejes cuyo valor cambia ALGUN modelo de la rejilla seleccionada.

    Se apoya en el alcance por celda de `huella`: con `--formulaciones 2` la
    lambda del EASE no toca ningun modelo construido, asi que barrerla solo
    duplicaria columnas identicas (y el tiempo de evaluacion). Los ejes
    descartados se quedan en el default del repo y se publican como tal.
    """
    alcance: set[str] = set()
    for form in formulaciones:
        for entidad in entidades:
            alcance |= set(huella.alcance(form, entidad))
    return [attr for attr, _ in HIPERPARAMETROS if attr in alcance]


def configuraciones(
    formulaciones: tuple[str, ...], entidades: tuple[str, ...]
) -> list[dict[str, object]]:
    """Producto cartesiano de los ejes utiles: una lista de {eje -> valor}.

    Sin nombres: nombrar es cosa de `registro.nombrar`, que asigna el `vNN` por la
    CONFIGURACION y no por la posicion en esta lista (que cambia en cuanto se
    edita `HIPERPARAMETROS`).
    """
    declarados = _valores_declarados()
    ejes = _ejes_relevantes(formulaciones, entidades)
    return [
        dict(zip(ejes, tupla))
        for tupla in itertools.product(*(declarados[eje] for eje in ejes))
    ]


def combinaciones(
    formulaciones: tuple[str, ...], entidades: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    """Nombres POSICIONALES `v01`, `v02`, ... para el producto cartesiano.

    Es un respaldo: los nombres reales de una carpeta viven en su
    `combinaciones.json` (`registro`), porque solo ahi son estables entre
    corridas. Esto solo lo usa `figuras3d` cuando dibuja una carpeta sin registro
    ni resumen, y solo vale si `HIPERPARAMETROS` no ha cambiado desde entonces.
    """
    combos = configuraciones(formulaciones, entidades)
    ancho = max(2, len(str(len(combos))))
    return {f"v{i:0{ancho}d}": c for i, c in enumerate(combos, start=1)}


def _ejes_publicados(extra: list[str] | None = None) -> list[str]:
    """Ejes que aparecen en la tabla de combinaciones: los declarados + `extra`.

    Incluye los que quedaron fuera del producto por no afectar a la rejilla
    seleccionada: su valor (el default) forma parte de la configuracion con la
    que se construyo todo y hay que poder leerlo.

    `extra` son los ejes que ya conoce el registro de la carpeta y que hoy no
    estan en `HIPERPARAMETROS`. Tienen que seguir publicandose (con su default)
    porque forman parte de la IDENTIDAD de las combinaciones ya registradas: si
    dejaran de contar, la misma configuracion recibiria un nombre nuevo. Se
    descartan los que ya no existan en `src.similitud.config`.
    """
    ejes = [attr for attr, _ in HIPERPARAMETROS]
    for eje in extra or ():
        if eje not in ejes and hasattr(scfg, eje):
            ejes.append(eje)
    return ejes


def _config_efectiva(valores: dict[str, object], ejes: list[str]) -> dict[str, object]:
    """Valor VIGENTE de cada eje en esa combinacion: default del repo + barrido.

    El resumen publica esto y no solo los ejes barridos: un eje ausente no quiere
    decir "sin posicion" ni "sin EASE", quiere decir lo que diga
    `src/similitud/config.py` el dia de la corrida. Sin el valor efectivo, dos
    barridos con defaults distintos producen tablas identicas que afirman cosas
    distintas.

    Debe llamarse FUERA de `_config_temporal` (con los defaults restaurados).
    """
    return {eje: valores.get(eje, getattr(scfg, eje)) for eje in ejes}


# --------------------------------------------------------------------------- #
# Construccion + evaluacion de una combinacion                                 #
# --------------------------------------------------------------------------- #

def _stem(formulacion: str, entidad: str, normalizacion: str) -> str:
    """Nombre del artefacto, igual que lo genera `ModeloSimilitud.guardar`."""
    return f"formulacion{formulacion}_{entidad}_{normalizacion}"


def _origen_reutilizable(
    out_dir: Path, model_dir: Path, stem: str, h: dict
) -> Path | None:
    """Busca en las DEMAS combinaciones un artefacto con exactamente esta huella.

    Es lo que hace que una combinacion que solo mueve `F2_L1` no reconstruya los
    modelos F5, y que mover la posicion no reconstruya los de equipo: su huella
    coincide porque el alcance de esos atributos no entra en esa celda (ver
    `huella.alcance`).
    """
    model_dir = Path(model_dir)
    for cand in sorted(Path(out_dir).glob(f"*/modelo/{stem}{huella.SUFIJO}")):
        origen = cand.parent
        if origen == model_dir:
            continue
        if huella.coincide(origen, stem, h):
            return origen
    return None


def _ruta_visible(p: Path) -> str:
    """Ruta para el log: relativa al cwd si cuelga de el, absoluta si no.

    Todos los comandos se lanzan desde la raiz del repo, asi que la relativa
    (`outputs/evaluacion/barrido/v03/modelo/...`) identifica el fichero sin
    llenar la linea con el prefijo de la instalacion.
    """
    p = Path(p).resolve()
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def _copiar_artefacto(origen: Path, destino: Path, stem: str) -> None:
    """Copia .npz + .json + huella. Copiar (~11-36 MB) cuesta segundos frente a
    los >10 min que tarda reconstruir un F2, y deja cada combinacion autocontenida.
    """
    destino.mkdir(parents=True, exist_ok=True)
    for ext in (".npz", ".json", huella.SUFIJO):
        shutil.copy2(origen / f"{stem}{ext}", destino / f"{stem}{ext}")


def _construir_grid(
    db_path: Path,
    model_dir: Path,
    out_dir: Path,
    rehacer: bool = False,
    adoptar: bool = False,
    formulaciones: tuple[str, ...] = ecfg.FORMULACIONES,
    entidades: tuple[str, ...] = ecfg.ENTIDADES,
) -> dict[str, int]:
    """Asegura la rejilla en `model_dir`, reconstruyendo solo lo necesario.

    Por cada celda, en orden: (1) si el artefacto de esta combinacion ya tiene la
    huella pedida —mismos hiperparametros, mismos datos, mismo codigo—, no se
    toca; (2) si otra combinacion tiene uno con la misma huella, se copia; (3) si
    `adoptar`, se intenta cargar un artefacto sin huella cuyo meta concuerde;
    (4) si no, se construye. Con `rehacer` se salta la caché entera. Devuelve el
    recuento por via, para el log.

    `formulaciones`/`entidades` acotan la rejilla (por defecto, entera). Las
    celdas fuera del subconjunto ni se construyen ni se borran: si quedaron de
    una corrida anterior siguen en disco con su huella, y otra combinacion puede
    reutilizarlas.
    """
    model_dir = Path(model_dir)
    conteo = {"en_cache": 0, "copiados": 0, "adoptados": 0, "construidos": 0}

    for entidad in entidades:
        for form in formulaciones:
            for norm in ecfg.NORMALIZACIONES:
                stem = _stem(form, entidad, norm)
                h = huella.calcular(form, entidad, norm, db_path)

                if not rehacer:
                    if huella.coincide(model_dir, stem, h):
                        print(f"    [cache]    {stem}")
                        print(f"               carga de disco: "
                              f"{_ruta_visible(model_dir / f'{stem}.npz')} "
                              f"(huella identica, no se reconstruye)")
                        conteo["en_cache"] += 1
                        continue
                    origen = _origen_reutilizable(out_dir, model_dir, stem, h)
                    if origen is not None:
                        print(f"    [copia]    {stem}")
                        print(f"               carga de disco: "
                              f"{_ruta_visible(origen / f'{stem}.npz')} "
                              f"(huella identica en la combinacion "
                              f"{origen.parent.name})")
                        _copiar_artefacto(origen, model_dir, stem)
                        conteo["copiados"] += 1
                        continue
                    if adoptar and huella.adoptar(model_dir, stem, entidad, h):
                        print(f"    [adopta]   {stem}")
                        print(f"               carga de disco: "
                              f"{_ruta_visible(model_dir / f'{stem}.npz')} "
                              f"(sin huella previa, meta concordante)")
                        conteo["adoptados"] += 1
                        continue

                # Ninguna via de la caché ha servido el artefacto: hay que
                # entrenarlo. Se dice por que, para que el log distinga "no habia
                # nada reutilizable" de "se pidio --rehacer".
                motivo = ("--rehacer: se ignora la cache" if rehacer
                          else "no hay ningun artefacto con esta huella")
                print(f"    [construye] {stem}")
                print(f"               NO se carga de disco, se construye en: "
                      f"{_ruta_visible(model_dir / f'{stem}.npz')} ({motivo})")

                # Borrar la huella ANTES: si la construccion peta, el artefacto a
                # medias queda sin huella y nadie lo reutilizara.
                huella.invalidar(model_dir, stem)
                sbuild._construir_uno(db_path, model_dir, form, entidad, norm)
                huella.escribir(model_dir, stem, h)
                conteo["construidos"] += 1

    return conteo


def evaluar_combinacion(
    nombre: str,
    valores: dict[str, object],
    db_path: Path,
    out_dir: Path,
    fases_sel: set[str],
    bootstrap: int,
    roles: dict,
    minutos: dict,
    con_figuras: bool = True,
    rehacer: bool = False,
    adoptar: bool = False,
    formulaciones: tuple[str, ...] = ecfg.FORMULACIONES,
    entidades: tuple[str, ...] = ecfg.ENTIDADES,
) -> dict:
    """Construye y evalua una combinacion completa; devuelve la tabla `salida`."""
    var_dir = out_dir / nombre
    model_dir = var_dir / "modelo"
    var_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"COMBINACION '{nombre}' -> {_fmt_valores(valores)}")
    print(f"{'=' * 70}")

    with _config_temporal(valores):
        print("[construir] rejilla de modelos ...")
        conteo = _construir_grid(db_path, model_dir, out_dir, rehacer, adoptar,
                                 formulaciones, entidades)
        print("    " + ", ".join(f"{v} {k}" for k, v in conteo.items() if v))

        modelos = evaluar.cargar_modelos(model_dir, formulaciones, entidades)
        if not modelos:
            raise SystemExit(f"No se construyo ningun modelo para {nombre!r}")

        n_ctx = len(entidades) * len(ecfg.NORMALIZACIONES)
        total = n_ctx + evaluar._pasos_totales(len(modelos), fases_sel, bootstrap)
        prog = evaluar.Progreso(total)
        print(f"[evaluar] {len(modelos)} modelos | fases {sorted(fases_sel)} | "
              f"bootstrap B={bootstrap} | {total} pasos.\n")

        ctxs = evaluar.crear_contextos(db_path, prog, entidades)
        salida = evaluar.ejecutar(
            modelos, ctxs, roles, minutos, fases_sel, bootstrap, prog
        )

    # La combinacion va como columna en los CSV: los ficheros de dos de ellas se
    # llaman igual y contienen los mismos nombres de modelo, asi que sin ella
    # solo los distingue la carpeta que los contiene.
    evaluar.escribir_csv(salida, var_dir, {"combinacion": nombre})
    if con_figuras:
        evaluar.escribir_figuras(salida, var_dir)
    return salida


# --------------------------------------------------------------------------- #
# Resumen comparativo (limpio: sin umbrales ni referencias de la documentacion) #
# --------------------------------------------------------------------------- #

# Los modelos se identifican por su TRIPLETA (formulacion, entidad,
# normalizacion), no por la etiqueta compacta `F2_jugador_global`: en las tablas
# comparativas esa etiqueta sola no dice de que eje viene cada palabra, y la
# misma etiqueta designa modelos distintos en combinaciones distintas.
def _id_modelo(r) -> tuple[str, str, str]:
    return (r["formulacion"], r["entidad"], r["normalizacion"])


def _extraer(tabla: str, campo: str, filtro=None):
    """Extractor {tripleta -> valor} de una metrica dentro de una tabla."""
    def extractor(s: dict) -> dict:
        return {_id_modelo(r): r[campo] for r in s[tabla]
                if filtro is None or filtro(r)}
    return extractor


def _solo_global(r) -> bool:
    """La Fase 1 reporta tres direcciones; la comparativa usa la media."""
    return r["direccion"] == "global"


# Que metrica sale de que tabla de `salida`, con el nombre que llevara en la tabla
# larga (`barrido_metricas.csv`) y en las comparativas.
ESPECS: list[tuple[str, str, object]] = [
    ("0", "pureza_top1", _extraer("f0", "pureza_top1")),
    ("0", "asimetria", _extraer("f0", "asimetria")),
    ("1", "top1", _extraer("f1", "top1", _solo_global)),
    ("1", "mrr", _extraer("f1", "mrr", _solo_global)),
    ("2", "rbo_medio", _extraer("f2", "rbo_medio")),
    ("5", "knn_accuracy", _extraer("f5", "knn_accuracy")),
    ("5", "coverage", _extraer("f5", "coverage")),
    ("5", "diversity", _extraer("f5", "diversity")),
]

# Secciones del resumen: (titulo, [(titulo de la tabla, metrica)]). La orientacion
# (mayor/menor es mejor) la da `puntuacion.orientacion`, para que las tablas y el
# score compuesto no puedan discrepar sobre el sentido de una metrica.
SECCIONES: list[tuple[str, list[tuple[str, str]]]] = [
    ("Sanity — pureza posicional y simetria de la S", [
        ("Pureza top-1 (mayor es mejor)", "pureza_top1"),
        ("Asimetria de la S (menor es mejor)", "asimetria"),
    ]),
    ("Auto-similitud (mayor es mejor)", [
        ("Top-1", "top1"),
        ("MRR", "mrr"),
    ]),
    ("Estabilidad ante remuestreo (mayor es mejor)", [
        ("RBO@10 medio", "rbo_medio"),
    ]),
    ("Downstream y diversidad (mayor es mejor)", [
        ("Clasificacion posicional k-NN (accuracy)", "knn_accuracy"),
        ("Cobertura", "coverage"),
        ("Diversidad intra-lista", "diversity"),
    ]),
]


def _es_finito(x) -> bool:
    return isinstance(x, (int, float)) and x == x and x not in (float("inf"), float("-inf"))


# Columnas que identifican el modelo al principio de cada tabla comparativa.
_COLUMNAS_ID = ("formulacion", "entidad", "normalizacion")


def _celdas_id(clave: tuple[str, str, str]) -> list[str]:
    form, entidad, norm = clave
    return [f"F{form}", entidad, norm]


def _orden_modelo(clave: tuple[str, str, str]) -> tuple[str, str, str]:
    """Agrupa por entidad (jugador y equipo no son comparables entre si)."""
    form, entidad, norm = clave
    return (entidad, form, norm)


def _valores_de(df: pd.DataFrame, metrica: str) -> dict:
    """{(combinacion, tripleta) -> valor finito} de una metrica de la tabla larga."""
    sub = df[df["metrica"] == metrica]
    valores: dict[tuple[str, tuple[str, str, str]], float] = {}
    for r in sub.itertuples(index=False):
        if _es_finito(r.valor):
            clave = (r.combinacion, (r.formulacion, r.entidad, r.normalizacion))
            valores[clave] = float(r.valor)
    return valores


def _tabla_comparativa(
    L: list[str], df: pd.DataFrame, metrica: str, titulo: str, combis: list[str]
) -> None:
    """Escribe una tabla modelo (filas) x combinacion (columnas) para una metrica.

    Cada fila lleva delante los tres ejes que identifican el modelo. Se resalta
    en **negrita** el mejor valor de la fila (segun la orientacion de la metrica)
    y se cierra con la combinacion de mejor media, para leer el efecto de los
    hiperparametros sin apoyarse en ningun umbral. Tanto el resaltado como esa
    linea se omiten cuando las combinaciones no discrepan: ahi no hay efecto que
    leer.

    Como la carpeta es acumulativa, una celda puede estar vacia (`n/a`) porque esa
    combinacion se evaluo con otras fases, no porque el modelo fallara. Las
    columnas SIN ningun valor de esta metrica se omiten enteras.
    """
    valores = _valores_de(df, metrica)
    if not valores:
        return
    mejor = puntuacion.orientacion(metrica)
    cols = [c for c in combis if any(k[0] == c for k in valores)]
    modelos = sorted({k[1] for k in valores}, key=_orden_modelo)

    L.append(f"### {titulo}\n")
    L.append("| " + " | ".join(_COLUMNAS_ID) + " | " + " | ".join(cols) + " |")
    L.append("|" + "---|" * (len(_COLUMNAS_ID) + len(cols)))
    for m in modelos:
        fila = [valores.get((c, m)) for c in cols]
        finitos = [x for x in fila if x is not None]
        # Solo hay "mejor" si las combinaciones discrepan. Una fila con el mismo
        # valor en todas es el MISMO modelo reutilizado (el equipo no tiene
        # posicion, p. ej.): resaltarlo haria creer que gana algo.
        objetivo = None
        if len(set(finitos)) > 1:
            objetivo = max(finitos) if mejor == "max" else min(finitos)
        celdas = []
        for x in fila:
            txt = _fmt(x)
            if objetivo is not None and x is not None and x == objetivo:
                txt = f"**{txt}**"
            celdas.append(txt)
        L.append("| " + " | ".join(_celdas_id(m)) + " | " + " | ".join(celdas) + " |")

    # Combinacion de mejor media (sobre los modelos con valor finito). Solo se
    # promedian columnas con los MISMOS modelos: con la carpeta acumulada, una
    # combinacion evaluada sobre media rejilla tendria una media que no compara
    # lo mismo que las demas.
    por_col: dict[str, list[float]] = {c: [] for c in cols}
    for (c, _m), x in valores.items():
        por_col[c].append(x)
    completas = {c: xs for c, xs in por_col.items() if len(xs) == len(modelos)}
    medias = {c: sum(xs) / len(xs) for c, xs in completas.items()}
    if len(set(medias.values())) > 1:
        elegir = max if mejor == "max" else min
        ganadora = elegir(medias, key=medias.get)
        L.append(f"\nMejor media entre combinaciones: **{ganadora}** "
                 f"({_fmt(medias[ganadora])}).\n")
    elif medias:
        # Todas dan lo mismo: proclamar una ganadora seria ruido.
        L.append("\nNinguna combinacion mueve esta metrica.\n")
    else:
        L.append("\n(Ninguna combinacion tiene esta metrica en todos los modelos: "
                 "sin media comparable.)\n")


def _tabla_larga_csv(resultados: dict) -> pd.DataFrame:
    """Formato largo para analisis externo, con el modelo ya descompuesto en sus
    tres ejes (ademas de la etiqueta compacta, para poder filtrar por cualquiera).
    """
    filas = []
    for combinacion, s in resultados.items():
        for fase, metrica, extractor in ESPECS:
            for clave, valor in extractor(s).items():
                form, entidad, norm = clave
                filas.append({"combinacion": combinacion,
                              "modelo": f"F{form}_{entidad}_{norm}",
                              "formulacion": form, "entidad": entidad,
                              "normalizacion": norm,
                              "fase": fase, "metrica": metrica, "valor": valor})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Secciones del resumen                                                        #
# --------------------------------------------------------------------------- #

def _seccion_combinaciones(L: list[str], reg: dict, ejes: list[str],
                           evaluadas: set[str]) -> None:
    """Indice nombre -> configuracion efectiva, con la fecha en que se evaluo."""
    configs = registro.hiperparametros(reg)
    proc = registro.procedencias(reg)

    L.append("## Combinaciones (columnas de las tablas)\n")
    L.append("Todas las combinaciones ACUMULADAS en esta carpeta, no solo las de la "
             "ultima corrida. El nombre `vNN` identifica una CONFIGURACION (se "
             "asigna en `combinaciones.json`), asi que significa lo mismo en todas "
             "las corridas y en todas las figuras de la carpeta.\n")
    if not ejes:
        L.append("No hay hiperparametros declarados: una sola combinacion, con la "
                 "configuracion por defecto de `src/similitud/config.py`.\n")
        return

    L.append("| combinacion | " + " | ".join(ejes) + " | evaluada |")
    L.append("|" + "---|" * (len(ejes) + 2))
    # Marcar "esta corrida" solo distingue algo si la carpeta trae ademas
    # combinaciones de corridas anteriores.
    marcar = bool(set(configs) - evaluadas)
    for nombre, config in configs.items():
        # Negrita = valor distinto del default del repo. Una combinacion puede
        # fijar un eje al valor que ya traia el default: eso no es un cambio. Un
        # eje que ya no existe en `config` no tiene default con el que contrastar:
        # se publica en redonda, porque la negrita afirmaria algo que no se sabe.
        celdas = []
        for eje in ejes:
            valor = config.get(eje)
            distinto = hasattr(scfg, eje) and valor != getattr(scfg, eje)
            celdas.append(f"**`{valor}`**" if distinto else f"`{valor}`")
        cuando = proc.get(nombre, {}).get("corrida", "?")
        marca = (f"{cuando} *(esta corrida)*"
                 if marcar and nombre in evaluadas else cuando)
        L.append(f"| {nombre} | " + " | ".join(celdas) + f" | {marca} |")

    L.append("\nValores **efectivos** de cada eje, no solo los que se barren: "
             "en negrita los que se apartan del default de "
             "`src/similitud/config.py`, en redonda los que lo heredan. Un eje "
             "que no afecta a la rejilla seleccionada (la lambda del EASE si "
             "solo se pide la F2, los `F2_*` si solo se pide la F5) no se "
             "barre: se queda en el default. El resto de hiperparametros es "
             "identico en todas.\n")

    if not registro.homogeneo(reg):
        L.append("> **Aviso**: las combinaciones acumuladas no se evaluaron todas "
                 "con la misma BD ni con el mismo codigo del nucleo numerico (ver "
                 "`combinaciones.json`). Las metricas de corridas distintas se "
                 "comparan aqui en la misma tabla, pero no son estrictamente "
                 "comparables: para homogeneizarlas hay que volver a evaluar las "
                 "combinaciones afectadas (deben estar en el producto cartesiano "
                 "de `HIPERPARAMETROS`) con `--rehacer`.\n")


def _seccion_mejores(L: list[str], df: pd.DataFrame, configs: dict,
                     ejes_var: dict) -> None:
    """Ranking por score compuesto: que configuracion va mejor, y con que numero."""
    scores = puntuacion.puntuar(df)
    if scores.empty:
        return

    L.append("## Mejores combinaciones (score compuesto)\n")
    L.append("Agregado de todas las metricas llevadas a z-score **dentro de la "
             "entidad** (`src/evaluacion/puntuacion.py`). Sirve para ordenar "
             "candidatos cuando cada metrica señala a una combinacion distinta; "
             "**no es un criterio validado**: el protocolo juzga por convergencia "
             "de señales, no por un numero, y ninguna conclusion de la memoria "
             "deberia apoyarse solo en el. Jugador y equipo nunca se mezclan.\n")

    cabecera = ["entidad", "modelo", "combinacion", "score"] + list(ejes_var)
    L.append("| " + " | ".join(cabecera) + " |")
    L.append("|" + "---|" * len(cabecera))
    # `puntuar` ya devuelve ordenado por (entidad, score desc).
    for entidad, bloque in scores.groupby("entidad"):
        for r in bloque.head(3).itertuples(index=False):
            conf = configs.get(r.combinacion, {})
            valores = [f"`{conf.get(eje)}`" for eje in ejes_var]
            L.append(f"| {entidad} | {r.modelo} | {r.combinacion} | "
                     f"{getattr(r, puntuacion.NOMBRE):+.3f} | "
                     + " | ".join(valores) + " |")
    L.append("\nSolo se listan los tres mejores por entidad; el score completo de "
             "todas las combinaciones sale en `figuras3d/score.csv` "
             "(`python -m src.evaluacion.figuras3d --barrido <esta carpeta>`), que "
             "ademas dibuja la SUPERFICIE del score sobre los ejes de "
             "hiperparametros — donde se ve si el optimo es una meseta o cae en el "
             "borde de la rejilla (y entonces el rango barrido se queda corto).\n")


def escribir_resumen_barrido(
    df: pd.DataFrame, reg: dict, out_dir: Path, meta: dict
) -> None:
    """Genera `resumen_barrido.md`: comparativa entre combinaciones, sin umbrales.

    Trabaja sobre la tabla larga ACUMULADA de la carpeta, no sobre los resultados
    en memoria de esta corrida: es lo que permite que el resumen incluya las
    combinaciones que se evaluaron en corridas anteriores.
    """
    configs = registro.hiperparametros(reg)
    ejes = meta["ejes"]
    evaluadas = set(meta.get("evaluadas", ()))

    L: list[str] = []
    L.append("# Barrido de hiperparametros — comparativa entre combinaciones\n")
    L.append(f"Generado el {meta['fecha']}. La carpeta acumula **{len(configs)} "
             f"combinaciones**; esta corrida evaluo {len(evaluadas)} de ellas "
             f"(fases {meta['fases']}, bootstrap B={meta['bootstrap']}) y las "
             "demas conservan las metricas de la corrida en que se evaluaron.\n")
    if meta.get("rejilla"):
        L.append(f"Rejilla construida y evaluada en esta corrida: "
                 f"{meta['rejilla']}.\n")

    _seccion_combinaciones(L, reg, ejes, evaluadas)

    L.append("## Modelos (filas de las tablas)\n")
    L.append("Cada fila es un modelo, identificado por sus tres ejes:\n")
    L.extend(evaluar.LEYENDA_EJES)
    L.append("\nUn modelo puede no verse afectado por los ejes que mueve una "
             "combinacion (el equipo no tiene posicion, la F2 no tiene EASE): "
             "entonces la fila trae el MISMO modelo, reutilizado desde la caché, "
             "en varias columnas. Por eso solo se resalta el mejor valor de una "
             "fila **si las combinaciones discrepan**: un valor repetido no es un "
             "empate entre modelos distintos.\n")

    combis = [c for c in configs if c in set(df["combinacion"])] if not df.empty else []
    for titulo, tablas in SECCIONES:
        if not any(_valores_de(df, m) for _t, m in tablas):
            continue
        L.append(f"## {titulo}\n")
        for titulo_tabla, metrica in tablas:
            _tabla_comparativa(L, df, metrica, titulo_tabla, combis)

    _seccion_mejores(L, df, configs, registro.ejes_variables(configs))

    L.append("Los detalles por combinacion (CSV y figuras) estan en las "
             "subcarpetas correspondientes.\n")

    (out_dir / "resumen_barrido.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _subconjunto(pedido: str, validos: tuple[str, ...], que: str) -> tuple[str, ...]:
    """Parsea una lista coma-separada validando contra los valores permitidos.

    Conserva el orden de `validos` para que la rejilla se recorra siempre igual
    independientemente de como se escriba el flag.
    """
    pedidos = [v.strip() for v in pedido.split(",") if v.strip()]
    desconocidos = [v for v in pedidos if v not in validos]
    if desconocidos:
        raise SystemExit(f"{que} desconocidas: {desconocidos}. "
                         f"Disponibles: {', '.join(validos)}.")
    if not pedidos:
        raise SystemExit(f"Hay que indicar al menos una de las {que}: "
                         f"{', '.join(validos)}.")
    return tuple(v for v in validos if v in pedidos)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Construye y evalua un modelo por cada combinacion de los "
                    "hiperparametros declarados en HIPERPARAMETROS."
    )
    p.add_argument("--db", type=Path, default=ecfg.DEFAULT_DB_PATH)
    p.add_argument("--out", type=Path, default=ecfg.DEFAULT_OUT_DIR / "barrido")
    p.add_argument("--formulaciones", type=str, default=",".join(ecfg.FORMULACIONES),
                   help="Formulaciones de la rejilla (coma-separado). "
                        f"Disponibles: {', '.join(ecfg.FORMULACIONES)}. Acotar "
                        "aqui evita construir y evaluar lo que no interesa.")
    p.add_argument("--entidades", type=str, default=",".join(ecfg.ENTIDADES),
                   help="Entidades de la rejilla (coma-separado). "
                        f"Disponibles: {', '.join(ecfg.ENTIDADES)}.")
    p.add_argument("--fases", type=str, default="0,1,5",
                   help="Fases a ejecutar por combinacion (coma-separado). El "
                        "barrido multiplica el coste, por eso omite por defecto "
                        "la estabilidad (2) y la triangulacion (3).")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Remuestreos de la Fase 2 (0 la omite; solo aplica si se "
                        "incluye la fase 2).")
    p.add_argument("--sin-figuras", action="store_true")
    p.add_argument("--rehacer", action="store_true",
                   help="Ignora la cache y reconstruye todos los modelos.")
    p.add_argument("--adoptar-existentes", action="store_true",
                   help="Da por buenos los modelos ya presentes en <out> que no "
                        "tienen huella (construidos antes de existir la cache), "
                        "si su metadata concuerda con los hiperparametros "
                        "vigentes. Afirma que ni el nucleo numerico ni la BD han "
                        "cambiado desde entonces: eso no es verificable a "
                        "posteriori. Los modelos con posicion nunca se adoptan.")
    args = p.parse_args(argv)

    _validar_hiperparametros()
    formulaciones = _subconjunto(args.formulaciones, ecfg.FORMULACIONES, "formulaciones")
    entidades = _subconjunto(args.entidades, ecfg.ENTIDADES, "entidades")
    fases_sel = {s.strip() for s in args.fases.split(",") if s.strip()}

    db_path = args.db
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # La carpeta es acumulativa: su registro decide como se llama cada
    # configuracion, para que los `vNN` de esta corrida sigan designando lo mismo
    # que los de las anteriores (ver `registro.py`).
    reg = registro.cargar(out_dir)
    ejes = _ejes_publicados(registro.ejes(reg))
    avisos = registro.completar_ejes(
        out_dir, reg, ejes, {eje: getattr(scfg, eje) for eje in ejes}
    )
    for aviso in avisos:
        print(f"  [aviso] eje sin registrar en una combinacion anterior: {aviso}")

    conocidas = set(reg["combinaciones"])
    efectivas = [_config_efectiva(v, ejes)
                 for v in configuraciones(formulaciones, entidades)]
    seleccion = registro.nombrar(reg, efectivas, ejes)
    nuevas = [n for n in seleccion if n not in conocidas]

    print(f"{len(seleccion)} combinaciones en esta corrida "
          f"({len(nuevas)} nuevas, {len(seleccion) - len(nuevas)} ya conocidas); "
          f"{len(reg['combinaciones'])} acumuladas en {out_dir}:")
    for nombre, valores in seleccion.items():
        marca = " [nueva]" if nombre in nuevas else ""
        print(f"  {nombre}: {_fmt_valores(valores)}{marca}")

    t0 = time.perf_counter()
    print("\nCargando datos auxiliares (roles y minutos)...")
    roles = datos.rol_por_jugador(db_path)
    minutos = datos.minutos_por_jugador(db_path)

    resultados: dict = {}
    for nombre, valores in seleccion.items():
        resultados[nombre] = evaluar_combinacion(
            nombre, valores, db_path, out_dir, fases_sel, args.bootstrap,
            roles, minutos, con_figuras=not args.sin_figuras,
            rehacer=args.rehacer, adoptar=args.adoptar_existentes,
            formulaciones=formulaciones, entidades=entidades,
        )

    fecha = time.strftime("%Y-%m-%d %H:%M")
    registro.anotar(reg, list(seleccion),
                    {"corrida": fecha, **huella.procedencia(db_path)})
    registro.guardar(out_dir, reg)

    # Las metricas de esta corrida sustituyen a las suyas anteriores; las de las
    # combinaciones que no se han vuelto a evaluar se conservan. Es lo que hace
    # que la superficie 3D se dibuje con TODOS los puntos de la carpeta.
    previas = registro.leer_metricas(out_dir)
    acumuladas = registro.acumular(previas, _tabla_larga_csv(resultados))
    registro.escribir_metricas(out_dir, acumuladas)

    escribir_resumen_barrido(acumuladas, reg, out_dir, {
        "fecha": fecha,
        "fases": sorted(fases_sel),
        "bootstrap": args.bootstrap if "2" in fases_sel else 0,
        "ejes": ejes,
        "evaluadas": list(seleccion),
        "rejilla": f"formulaciones {list(formulaciones)} x entidades "
                   f"{list(entidades)} x normalizaciones "
                   f"{list(ecfg.NORMALIZACIONES)}",
    })
    dt = time.perf_counter() - t0
    n_combis = len(set(acumuladas["combinacion"])) if not acumuladas.empty else 0
    print(f"\nListo en {dt:.0f}s. Comparativa de las {n_combis} combinaciones "
          f"acumuladas en {out_dir}/resumen_barrido.md")
    print(f"Superficies 3D con todos esos puntos: "
          f"python -m src.evaluacion.figuras3d --barrido {out_dir}")


if __name__ == "__main__":
    main()

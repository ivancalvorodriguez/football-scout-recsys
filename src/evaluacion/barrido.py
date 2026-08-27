"""Punto de entrada: construye Y evalua un modelo por cada COMBINACION de hiperparametros.

    python -m src.evaluacion.barrido [--db RUTA] [--out DIR]
                                     [--formulaciones 2,5] [--entidades jugador,equipo]
                                     [--normalizaciones por_liga,global]
                                     [--distancias euclidea,mahalanobis,coseno,manhattan]
                                     [--metodo-f5 mmd,sinkhorn]
                                     [--fases 0,1,5] [--bootstrap B] [--sin-figuras]
                                     [--holdout LIGA] [--trabajos N|auto]
                                     [--sin-reutilizar-metricas]
                                     [--rehacer] [--adoptar-existentes]

A diferencia de `src.evaluacion.evaluar` (que exige que los modelos ya esten
construidos en `outputs/modelo/`), este script CONSTRUYE los modelos sobre la
marcha antes de evaluarlos: por cada combinacion de hiperparametros reconstruye
la rejilla (formulacion 2/5 x jugador/equipo x por_liga/global) en su propia
carpeta y despues corre el protocolo de evaluacion. Asi se puede lanzar de cero,
sin `build` previo.

**`--distancias` es el CUARTO eje de la rejilla, no un hiperparametro.** La
distancia entre observaciones (`src.similitud.distancias`: euclidea, mahalanobis,
coseno, manhattan) decide donde caen los vecinos de la F2, como se mide el residuo
de SLIM, que kernel usa el MMD y que coste el Sinkhorn. Se comporta EXACTAMENTE
como la normalizacion y por eso se acota igual, con una flag y no con una entrada
de `HIPERPARAMETROS`: cada valor da un artefacto propio, una fila propia en las
tablas y —esto es lo que la separa de un hiperparametro— una FIGURA propia en las
superficies 3D, en vez de un eje dentro de la misma figura. Comparar geometrias no
es afinar un numero dentro de una.

Por defecto entra solo la `euclidea`, a diferencia de las normalizaciones (que
entran las dos): son cuatro valores, no dos, y uno de ellos —`manhattan` sobre la
F2 de jugador— no tiene la via rapida del producto de matrices y se va a horas. La
euclidea es ademas la geometria con la que se construyo y midio todo lo que hay
acumulado en `outputs/`, asi que un barrido sin la flag sigue significando lo
mismo que antes.

**Como se reparte el trabajo.** La ejecucion no es un bucle de combinaciones: es un
PLAN de celdas. Una celda es (combinacion, formulacion, entidad, normalizacion,
distancia) y es la unidad indivisible tanto de construccion (un artefacto) como de
evaluacion
(las fases de un modelo, `evaluar.evaluar_modelo`). El plan se calcula entero
antes de empezar, y eso habilita las dos cosas que hacen que el barrido termine:

1. **Deduplicacion por huella.** Dos celdas de combinaciones distintas cuya huella
   coincide son el MISMO modelo (es lo que ya sostenia la cache de artefactos: se
   copia el `.npz` de una a otra). Si el modelo es el mismo y las fases son
   deterministas —lo son: todas las semillas estan fijadas en
   `src.evaluacion.config`—, tambien lo son sus metricas. Se ajusta y se evalua
   una sola vez por huella y el resultado se reparte entre todas las celdas del
   grupo. Con la rejilla del repo (ejes F2 x ejes F5) eso divide el trabajo entre
   ~20: mover la lambda del EASE no cambia ni un numero de los modelos F2, y hasta
   ahora se recalculaban enteros en cada combinacion. `--sin-reutilizar-metricas`
   lo desactiva (evalua cada celda por su cuenta), que es como se comprueba que la
   igualdad se cumple de verdad.
2. **Paralelismo real** (`--trabajos N`). Las celdas son independientes, asi que
   se reparten entre procesos: distintos modelos —y, con ellos, distintas fases—
   avanzan a la vez. Tienen que ser PROCESOS y no hilos por dos razones: las fases
   son bucles de Python (el GIL las serializaria) y cada combinacion fija sus
   hiperparametros en `src.similitud.config`, que es estado global del interprete.
   Ver `src/evaluacion/paralelo.py`.

La Fase 3 (triangulacion) no cabe en una celda —compara las S de varios modelos
entre si—, asi que es una tarea aparte por combinacion.

**`--holdout` cambia lo que se construye, no solo lo que se mide.** Con el, los
artefactos del barrido se ajustan SIN esas ligas y la Fase 7 mide sobre ese mismo
artefacto (no se reajusta otro): el experimento es «entreno con estas ligas y veo
que hace con la que dejo fuera». Eso tiene tres consecuencias que conviene tener
presentes: los modelos cubren menos entidades (y son los comparables con los que
sirve la app, si su BD tampoco tiene esa liga), las metricas no son comparables
con las de un barrido sin hold-out —otro universo, otro pool— y las ligas
excluidas entran en la HUELLA del artefacto, de modo que la carpeta acumulativa
no mezcla unos con otros.

Los hiperparametros que se barren NO se piden por linea de comandos: estan
escritos en `HIPERPARAMETROS`, aqui abajo. El barrido recorre el PRODUCTO
CARTESIANO de esa lista — cada combinacion fija todos los ejes a la vez — y
construye/evalua cada una en `<out>/vNN/`. La linea de comandos no elige valores
de hiperparametro: solo decide QUE modelos entran en la rejilla
(`--formulaciones`, `--entidades`, `--normalizaciones`, `--metodo-f5`), COMO se
evaluan (`--fases`, `--bootstrap`, `--sin-figuras`) y que hace la caché
(`--rehacer`, `--adoptar-existentes`).

**`--metodo-f5` es la excepcion a lo anterior, y por eso es una flag y no una
entrada de `HIPERPARAMETROS`.** El metodo de la etapa 1 de la F5 (`mmd` o
`sinkhorn`) no es un numero que se afine: es QUE algoritmo se ejecuta, y hoy se
elige por ENTIDAD en `similitud.config` (jugador `mmd`, equipo `sinkhorn`) porque
sus costes no se parecen — el OT del Sinkhorn es cuadratico en observaciones
(~30 s en los 98 equipos, ~3,6 h en los 2.640 jugadores) y el MMD es lineal
(~4 s). Ponerlo en la lista lo dejaria encendido en todas las tandas; como flag,
forzarlo es una decision explicita de esa ejecucion.

Lo que si comparte con los hiperparametros es la mecanica: la flag inyecta
`F5_METODO_JUGADOR` y `F5_METODO_EQUIPO` como EJES de la rejilla, con lo que el
metodo entra en la identidad de la combinacion (`registro`) y en la huella del
artefacto (`huella._METODO_F5`). Dos consecuencias que son el motivo de hacerlo
asi: un barrido con `--metodo-f5` no pisa los `vNN` del que se lanzo sin ella, y
`--metodo-f5 mmd,sinkhorn` compara los dos metodos como una columna mas del
resumen. Con las dos entidades en la rejilla eso son cuatro combinaciones pero
solo cuatro artefactos (dos por entidad): cada celda depende del eje de su
entidad y la deduplicacion por huella se encarga del resto.

Como todo el pipeline lee los valores de `src.similitud.config` en tiempo de
llamada, basta con fijarlos antes de construir/evaluar y restaurarlos despues. El
objetivo es observar el EFECTO de cada hiperparametro comparando las
combinaciones entre si.

**La carpeta de salida es ACUMULATIVA** (ver `registro.py`): relanzar el barrido
sobre la misma `--out` despues de editar `HIPERPARAMETROS` no borra lo anterior.
Cada combinacion se identifica por su CONFIGURACION, no por su posicion en el
producto cartesiano, asi que conserva su nombre `vNN` y su carpeta entre
ejecuciones; las que ya se evaluaron y no vuelven a tocarse mantienen sus metricas
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
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.similitud import config as scfg
from src.similitud import warm

from . import config as ecfg
from . import (
    evaluar,
    generalizacion,
    huella,
    paralelo,
    puntuacion,
    registro,
    trabajo,
)
from .evaluar import _fmt
from .trabajo import Celda, config_temporal as _config_temporal


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
    ("F2_L1", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
    ("F2_BETA", [11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0]),
    # Formulacion 5 (distribucional + EASE): fuerza del re-ranking EASE, barrida
    # en un rango amplio (de casi-sin-regularizar a dominado por el ridge), y
    # dimension del embedding RFF con que se aproxima el MMD del jugador.
    # ("F5_EASE_LAMBDA", [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
    # ("F5_RFF_DIM", [1152, 1184, 1216, 1248, 1280]),

    # Arreglos de jugador.
    ("F5_EASE_LAMBDA", [0.0]),
    ("F5_RFF_DIM", [960, 992, 1024, 1056, 1088]),
    ]

# El bloque de posicion del jugador (25 columnas `pos_*`) NO se barre: forma
# parte del modelo base, z-scoreado con el resto y dividido por sqrt(25) para que
# pese como una sola feature (`config.USE_POSITION_FEATURES` /
# `config.POSITION_SCALING = "zscore_sqrt"`). El resto de features sigue con sus
# dos normalizaciones, `por_liga` y `global`, que son un eje de la REJILLA y no un
# hiperparametro: por eso se acotan con `--normalizaciones` (como las
# formulaciones y las entidades) y no se declaran aqui. Por defecto entran las
# dos, que es lo que permite compararlas. Para volver a contrastar el vector con y
# sin posicion hay que tocar esos dos atributos de `src/similitud/config.py` a mano.


Ejes = list[tuple[str, list[object]]]


def _ejes_declarados(extra: Ejes | None = None) -> Ejes:
    """`HIPERPARAMETROS` + los ejes que inyecta la linea de comandos.

    El unico eje inyectable hoy es el metodo de la etapa 1 de la F5
    (`--metodo-f5`, ver `_ejes_metodo_f5`). Se anade DESPUES de los declarados
    para que el orden de las columnas del resumen sea estable.
    """
    return [*HIPERPARAMETROS, *(extra or [])]


def _valores_declarados(extra: Ejes | None = None) -> dict[str, list[object]]:
    return {attr: valores for attr, valores in _ejes_declarados(extra)}


def _validar_hiperparametros(extra: Ejes | None = None) -> None:
    """Falla al arrancar si la lista de arriba tiene un typo o un eje vacio.

    Antes de gastar horas construyendo modelos: un nombre que no existe en
    `src.similitud.config` no sobrescribe nada y el barrido acabaria comparando
    combinaciones identicas sin decirlo.

    Los ejes inyectados por la linea de comandos pasan por lo mismo, y ademas el
    control de duplicados los cruza con los declarados: declarar `F5_METODO_*` en
    `HIPERPARAMETROS` y pasar `--metodo-f5` a la vez son dos fuentes para el mismo
    atributo, y una ganaria en silencio.
    """
    vistos: set[str] = set()
    for attr, valores in _ejes_declarados(extra):
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


def _fmt_valores(valores: dict[str, object]) -> str:
    if not valores:
        return "(config por defecto)"
    return ", ".join(f"{k}={v}" for k, v in valores.items())


# --------------------------------------------------------------------------- #
# Producto cartesiano de los hiperparametros                                   #
# --------------------------------------------------------------------------- #

def _ejes_metodo_f5(metodos: tuple[str, ...]) -> Ejes:
    """Ejes que inyecta `--metodo-f5`: uno por entidad, con los mismos valores.

    El metodo de la etapa 1 de la F5 no es UN atributo de `similitud.config` sino
    DOS (`F5_METODO_JUGADOR` y `F5_METODO_EQUIPO`, ver `formulacion5._metodo_por_entidad`),
    asi que la flag inyecta los dos. No hace falta filtrar por la rejilla pedida:
    `_ejes_relevantes` ya descarta el de la entidad que no se construye, porque
    `huella.alcance` los reparte por entidad.

    Que sean dos ejes independientes y no uno solo acoplado es lo que permite
    comparar los dos metodos en las dos entidades de una tanda: con
    `--metodo-f5 mmd,sinkhorn` salen cuatro combinaciones, pero solo cuatro
    modelos distintos (dos por entidad), porque cada celda depende unicamente del
    eje de SU entidad y la deduplicacion por huella reparte el resto.
    """
    valores: list[object] = list(metodos)
    return [("F5_METODO_JUGADOR", valores), ("F5_METODO_EQUIPO", valores)]


def _ejes_relevantes(
    formulaciones: tuple[str, ...], entidades: tuple[str, ...],
    extra: Ejes | None = None,
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
    return [attr for attr, _ in _ejes_declarados(extra) if attr in alcance]


def configuraciones(
    formulaciones: tuple[str, ...], entidades: tuple[str, ...],
    extra: Ejes | None = None,
) -> list[dict[str, object]]:
    """Producto cartesiano de los ejes utiles: una lista de {eje -> valor}.

    Sin nombres: nombrar es cosa de `registro.nombrar`, que asigna el `vNN` por la
    CONFIGURACION y no por la posicion en esta lista (que cambia en cuanto se
    edita `HIPERPARAMETROS`).
    """
    declarados = _valores_declarados(extra)
    ejes = _ejes_relevantes(formulaciones, entidades, extra)
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
    ejecuciones. Esto solo lo usa `figuras3d` cuando dibuja una carpeta sin registro
    ni resumen, y solo vale si `HIPERPARAMETROS` no ha cambiado desde entonces.
    """
    combos = configuraciones(formulaciones, entidades)
    ancho = max(2, len(str(len(combos))))
    return {f"v{i:0{ancho}d}": c for i, c in enumerate(combos, start=1)}


def _ejes_publicados(extra: list[str] | None = None,
                     inyectados: Ejes | None = None) -> list[str]:
    """Ejes que aparecen en la tabla de combinaciones: los declarados + `extra`.

    Incluye los que quedaron fuera del producto por no afectar a la rejilla
    seleccionada: su valor (el default) forma parte de la configuracion con la
    que se construyo todo y hay que poder leerlo.

    `extra` son los ejes que ya conoce el registro de la carpeta y que hoy no
    estan en `HIPERPARAMETROS`. Tienen que seguir publicandose (con su default)
    porque forman parte de la IDENTIDAD de las combinaciones ya registradas: si
    dejaran de contar, la misma configuracion recibiria un nombre nuevo. Se
    descartan los que ya no existan en `src.similitud.config`.

    `inyectados` son los ejes que anade la linea de comandos (`--metodo-f5`).
    Publicarlos es lo que hace que un barrido con la flag no pise al de sin ella:
    si el metodo no entrase en la identidad, dos modelos distintos compartirian
    `vNN` en la misma carpeta acumulativa y `barrido_metricas.csv` se sobrescribiria
    a si mismo. Las combinaciones registradas ANTES de que existiera la flag no
    declaran el eje, pero `registro.completar_ejes` lo recupera de la huella de sus
    artefactos —que si lo registra— antes de comparar identidades.
    """
    ejes = [attr for attr, _ in _ejes_declarados(inyectados)]
    for eje in extra or ():
        if eje not in ejes and hasattr(scfg, eje):
            ejes.append(eje)
    return ejes


def _config_efectiva(valores: dict[str, object], ejes: list[str]) -> dict[str, object]:
    """Valor VIGENTE de cada eje en esa combinacion: default del repo + barrido.

    El resumen publica esto y no solo los ejes barridos: un eje ausente no quiere
    decir "sin posicion" ni "sin EASE", quiere decir lo que diga
    `src/similitud/config.py` el dia de la ejecucion. Sin el valor efectivo, dos
    barridos con defaults distintos producen tablas identicas que afirman cosas
    distintas.

    Debe llamarse FUERA de `_config_temporal` (con los defaults restaurados).
    """
    return {eje: valores.get(eje, getattr(scfg, eje)) for eje in ejes}


# --------------------------------------------------------------------------- #
# Construccion + evaluacion de una combinacion                                 #
# --------------------------------------------------------------------------- #

def _stem(formulacion: str, entidad: str, normalizacion: str,
          distancia: str = ecfg.DISTANCIAS_POR_DEFECTO[0]) -> str:
    """Nombre del artefacto, igual que lo genera `ModeloSimilitud.guardar`."""
    return Celda(formulacion, entidad, normalizacion, distancia).stem


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
    """Copia .npz + .json + huella (+ estado warm, si lo hay).

    Copiar (~11-36 MB) cuesta segundos frente a los >10 min que tarda reconstruir
    un F2, y deja cada combinacion autocontenida.

    El estado warm va con ellos porque la Fase 7 lo necesita para proyectar sin
    reajustar (el ancho de kernel de la F5 vive ahi): sin copiarlo, una celda que
    sale de una copia pagaria un ajuste entero solo por no tener a mano un numero.
    Es opcional —los artefactos anteriores a `warm` no lo llevan— y por eso no
    esta en el bucle obligatorio.
    """
    destino.mkdir(parents=True, exist_ok=True)
    for ext in (".npz", ".json", huella.SUFIJO):
        shutil.copy2(origen / f"{stem}{ext}", destino / f"{stem}{ext}")
    estado = origen / f"{stem}{warm.SUFIJO_ESTADO}"
    if estado.is_file():
        shutil.copy2(estado, destino / estado.name)


# Nombre de la via en el recuento que se imprime.
_CONTEO = {"cache": "en_cache", "copia": "copiados",
           "adopta": "adoptados", "construye": "construidos"}


def _decidir_celda(
    db_path: Path, out_dir: Path, model_dir: Path, celda: Celda,
    rehacer: bool, adoptar: bool, holdout: tuple[str, ...] = (),
) -> tuple[str, Path | None, dict]:
    """De donde sale el artefacto de la celda: `(via, origen, huella)`.

    Por orden: (1) si el artefacto de esta combinacion ya tiene la huella pedida
    —mismos hiperparametros, mismos datos, mismo codigo—, no se toca (`cache`);
    (2) si otra combinacion tiene uno con la misma huella, se copia (`copia`);
    (3) si `adoptar`, se intenta dar por bueno un artefacto sin huella cuyo meta
    concuerde (`adopta`); (4) si no, hay que ajustarlo (`construye`). Con
    `rehacer` se salta la caché entera.

    Tiene que llamarse DENTRO de `config_temporal` de la combinacion: la huella se
    calcula con la configuracion vigente, y la adopcion la contrasta con ella. La
    via `adopta` escribe la huella del artefacto adoptado, que es lo que lo
    convierte en reutilizable; las demas no tocan el disco.
    """
    h = huella.calcular(celda.formulacion, celda.entidad, celda.normalizacion,
                        db_path, excluir_ligas=holdout,
                        distancia=celda.distancia)
    if not rehacer:
        if huella.coincide(model_dir, celda.stem, h):
            return "cache", model_dir, h
        origen = _origen_reutilizable(out_dir, model_dir, celda.stem, h)
        if origen is not None:
            return "copia", origen, h
        if adoptar and huella.adoptar(model_dir, celda.stem, celda.entidad, h):
            return "adopta", model_dir, h
    return "construye", None, h


def _log_via(
    via: str, celda: Celda, model_dir: Path, origen: Path | None, rehacer: bool
) -> None:
    """Dice de que via sale el artefacto y con que ruta.

    Es lo que separa en el log los minutos de un ajuste de los segundos de una
    copia. Las tres vias de cache dicen de donde se carga y por que valia; la
    construccion dice donde va a escribir y por que no habia nada que reutilizar.
    """
    destino = _ruta_visible(model_dir / f"{celda.stem}.npz")
    if via == "cache":
        print(f"    [cache]    {celda.stem}")
        print(f"               carga de disco: {destino} "
              f"(huella identica, no se reconstruye)")
    elif via == "copia":
        print(f"    [copia]    {celda.stem}")
        print(f"               carga de disco: "
              f"{_ruta_visible(origen / f'{celda.stem}.npz')} "
              f"(huella identica en la combinacion {origen.parent.name})")
    elif via == "adopta":
        print(f"    [adopta]   {celda.stem}")
        print(f"               carga de disco: {destino} "
              f"(sin huella previa, meta concordante)")
    else:
        motivo = ("--rehacer: se ignora la cache" if rehacer
                  else "no hay ningun artefacto con esta huella")
        print(f"    [construye] {celda.stem}")
        print(f"               NO se carga de disco, se construye en: "
              f"{destino} ({motivo})")


def _construir_grid(
    db_path: Path,
    model_dir: Path,
    out_dir: Path,
    rehacer: bool = False,
    adoptar: bool = False,
    formulaciones: tuple[str, ...] = ecfg.FORMULACIONES,
    entidades: tuple[str, ...] = ecfg.ENTIDADES,
    normalizaciones: tuple[str, ...] = ecfg.NORMALIZACIONES,
    holdout: tuple[str, ...] = (),
    distancias_sel: tuple[str, ...] = ecfg.DISTANCIAS_POR_DEFECTO,
) -> dict[str, int]:
    """Asegura la rejilla de UNA combinacion en `model_dir`, celda a celda.

    Camino secuencial y autocontenido (lo usa `evaluar_combinacion`): decide y
    aplica cada celda en el acto. `main` no pasa por aqui — planifica TODAS las
    combinaciones juntas (`planificar`) para poder ajustar una sola vez lo que
    varias comparten y repartirlo entre procesos—, pero las dos vias deciden con
    la misma funcion, `_decidir_celda`, para que no puedan divergir.

    Devuelve el recuento por via. `formulaciones`/`entidades`/`normalizaciones`
    acotan la rejilla (por defecto, entera): las celdas fuera del subconjunto ni se
    construyen ni se borran, y si quedaron de una ejecucion anterior siguen en
    disco con su huella para que otra combinacion las reutilice.

    Debe llamarse dentro del `config_temporal` de la combinacion.
    """
    model_dir = Path(model_dir)
    conteo = {"en_cache": 0, "copiados": 0, "adoptados": 0, "construidos": 0}

    for celda in trabajo.celdas(formulaciones, entidades, normalizaciones,
                                distancias_sel):
        via, origen, h = _decidir_celda(db_path, out_dir, model_dir, celda,
                                        rehacer, adoptar, holdout)
        _log_via(via, celda, model_dir, origen, rehacer)
        if via == "copia":
            _copiar_artefacto(origen, model_dir, celda.stem)
        elif via == "construye":
            # `valores` vacio: la configuracion de la combinacion ya la fijo el
            # llamador y este camino no cambia de proceso.
            trabajo.construir_celda(db_path, model_dir, celda, {}, h,
                                    excluir_ligas=holdout)
        conteo[_CONTEO[via]] += 1

    return conteo


# --------------------------------------------------------------------------- #
# Plan de la ejecucion: las celdas de TODAS las combinaciones a la vez            #
# --------------------------------------------------------------------------- #

@dataclass
class Paso:
    """Una celda de una combinacion, con lo que hay que hacer para tenerla.

    `via` sale de `_decidir_celda`; `origen` solo aplica a las copias. `huella` es
    la identidad del artefacto: dos pasos con la misma huella (y el mismo stem)
    son el mismo modelo, y de ahi salen las dos deduplicaciones de la ejecucion
    —ajustar una vez, evaluar una vez—.
    """

    combinacion: str
    valores: dict[str, object]
    celda: Celda
    model_dir: Path
    huella: dict = field(repr=False)
    via: str
    origen: Path | None = None

    @property
    def nombre(self) -> str:
        """Identificador legible y unico del paso (nombre de tarea en el log)."""
        return f"{self.combinacion} {self.celda.etiqueta}"

    @property
    def clave_artefacto(self) -> str:
        """Identidad del MODELO, sin la combinacion de la que salio."""
        return json.dumps([self.celda.stem, self.huella], sort_keys=True)


def planificar(
    seleccion: dict[str, dict],
    db_path: Path,
    out_dir: Path,
    formulaciones: tuple[str, ...] = ecfg.FORMULACIONES,
    entidades: tuple[str, ...] = ecfg.ENTIDADES,
    normalizaciones: tuple[str, ...] = ecfg.NORMALIZACIONES,
    rehacer: bool = False,
    adoptar: bool = False,
    holdout: tuple[str, ...] = (),
    distancias_sel: tuple[str, ...] = ecfg.DISTANCIAS_POR_DEFECTO,
) -> list[Paso]:
    """Decide, para TODAS las celdas de todas las combinaciones, de donde salen.

    Decidirlo entero antes de tocar nada es lo que permite ajustar una sola vez lo
    que varias combinaciones comparten: cuando dos celdas que hay que construir
    tienen la misma huella, una se queda como propietaria y la otra pasa a copiar
    de ella (la copia se hace luego, cuando la propietaria ya existe). Sin este
    paso previo, N trabajadores arrancando a la vez sobre combinaciones
    consecutivas ajustarian N veces el mismo modelo: ninguno veria la huella del
    otro hasta que terminara.
    """
    pasos: list[Paso] = []
    for nombre, valores in seleccion.items():
        model_dir = Path(out_dir) / nombre / "modelo"
        with _config_temporal(valores):
            for celda in trabajo.celdas(formulaciones, entidades, normalizaciones,
                                        distancias_sel):
                via, origen, h = _decidir_celda(db_path, out_dir, model_dir,
                                                celda, rehacer, adoptar, holdout)
                pasos.append(Paso(combinacion=nombre, valores=dict(valores),
                                  celda=celda, model_dir=model_dir, huella=h,
                                  via=via, origen=origen))
    return _repartir_construcciones(pasos)


def _repartir_construcciones(pasos: list[Paso]) -> list[Paso]:
    """Deja una sola propietaria por artefacto; las demas pasan a copiarlo."""
    propietaria: dict[str, Paso] = {}
    for paso in pasos:
        if paso.via != "construye":
            continue
        duena = propietaria.get(paso.clave_artefacto)
        if duena is None:
            propietaria[paso.clave_artefacto] = paso
        else:
            paso.via, paso.origen = "copia", duena.model_dir
    return pasos


def construir_plan(
    pasos: list[Paso], db_path: Path, trabajos: int = 1, rehacer: bool = False,
    holdout: tuple[str, ...] = (),
) -> dict[str, int]:
    """Ajusta los artefactos que faltan (en paralelo) y reparte las copias.

    Dos etapas, en este orden: primero se ajusta cada artefacto DISTINTO —son
    independientes entre si, asi que van al pool— y despues se copian a las demas
    combinaciones que lo necesitan. Las copias van en serie y en el proceso padre:
    son segundos de E/S frente a los minutos de un ajuste, y hacerlas al final
    garantiza que la propietaria ya ha terminado.
    """
    conteo = {"en_cache": 0, "copiados": 0, "adoptados": 0, "construidos": 0}
    for paso in pasos:
        conteo[_CONTEO[paso.via]] += 1

    ajustes = [p for p in pasos if p.via == "construye"]
    if ajustes:
        print(f"\n[construir] {len(ajustes)} artefactos que ajustar "
              f"(de {len(pasos)} celdas; el resto sale de la cache o de una copia):")
        for paso in ajustes:
            _log_via("construye", paso.celda, paso.model_dir, None, rehacer)
        paralelo.mapear(
            [paralelo.Tarea(p.nombre, trabajo.construir_celda,
                            (db_path, p.model_dir, p.celda, p.valores, p.huella,
                             holdout))
             for p in ajustes],
            trabajos, etiqueta="ajustes",
        )
    else:
        print(f"\n[construir] nada que ajustar: las {len(pasos)} celdas salen de "
              f"la cache o de una copia.")

    copias = [p for p in pasos if p.via == "copia"]
    for paso in copias:
        _copiar_artefacto(paso.origen, paso.model_dir, paso.celda.stem)
    if copias:
        print(f"[construir] {len(copias)} artefactos copiados entre combinaciones.")
    return conteo


# --------------------------------------------------------------------------- #
# Evaluacion del plan                                                          #
# --------------------------------------------------------------------------- #

def _orden_informe(
    formulaciones: tuple[str, ...], entidades: tuple[str, ...],
    normalizaciones: tuple[str, ...] = ecfg.NORMALIZACIONES,
    distancias_sel: tuple[str, ...] = ecfg.DISTANCIAS_POR_DEFECTO,
) -> list[Celda]:
    """Orden en que las celdas aparecen en los CSV y el informe.

    Deliberadamente distinto del de ejecucion (`trabajo.celdas`, que agrupa por
    contexto): este reproduce el de `evaluar.ejecutar`, para que los ficheros de
    una combinacion salgan igual que si se hubiera evaluado del tiron. La
    distancia va la ultima, que es como se recorre alli (una geometria completa
    detras de otra).
    """
    return [Celda(form, entidad, norm, dist)
            for dist in distancias_sel
            for form in formulaciones
            for entidad in entidades
            for norm in normalizaciones]


def _agrupar(pasos: list[Paso], reutilizar: bool) -> dict[str, list[Paso]]:
    """Agrupa las celdas que van a dar exactamente el mismo resultado.

    La clave es la identidad del artefacto (stem + huella): mismos
    hiperparametros, mismos datos, mismo codigo. Las fases son deterministas
    (semillas fijas en `src.evaluacion.config`) y se corren con las mismas
    `--fases`/`--bootstrap` para todas, asi que dos celdas con esa clave igual dan
    las mismas metricas. Con `reutilizar=False` cada celda es su propio grupo.

    El grupo se evalua con los hiperparametros de SU PRIMERA celda, aunque las
    demas combinaciones tengan otros: los que difieren estan fuera del alcance de
    esa celda (`huella.alcance`) —es justo lo que permite agruparlas— y no entran
    en ningun numero que produzcan sus fases.
    """
    grupos: dict[str, list[Paso]] = {}
    for paso in pasos:
        clave = paso.clave_artefacto if reutilizar else paso.nombre
        grupos.setdefault(clave, []).append(paso)
    return grupos


def evaluar_plan(
    pasos: list[Paso],
    db_path: Path,
    fases_sel: set[str],
    bootstrap: int,
    trabajos: int = 1,
    reutilizar: bool = True,
    formulaciones: tuple[str, ...] = ecfg.FORMULACIONES,
    entidades: tuple[str, ...] = ecfg.ENTIDADES,
    normalizaciones: tuple[str, ...] = ecfg.NORMALIZACIONES,
    holdout: tuple[str, ...] = (),
    distancias_sel: tuple[str, ...] = ecfg.DISTANCIAS_POR_DEFECTO,
) -> dict[str, dict]:
    """Evalua todas las celdas del plan y devuelve {combinacion -> tabla `salida`}.

    Una tarea por grupo de celdas equivalentes (ver `_agrupar`) mas, si se pide la
    Fase 3, una tarea por combinacion. El reparto de un resultado entre las celdas
    de su grupo es literal: son las mismas filas, y la combinacion no aparece en
    ellas (la añaden `escribir_csv` y `_tabla_larga_csv` a partir de la clave del
    diccionario).
    """
    grupos = _agrupar(pasos, reutilizar)
    detalle = trabajos <= 1          # solo hay consola que refrescar en serie

    tareas = []
    for grupo in grupos.values():
        lider = grupo[0]
        tareas.append(paralelo.Tarea(
            lider.nombre, trabajo.evaluar_celda,
            (db_path, lider.model_dir, lider.celda, lider.valores, fases_sel,
             bootstrap, detalle, holdout),
        ))
    ahorro = len(pasos) - len(tareas)
    print(f"\n[evaluar] {len(pasos)} celdas | fases {sorted(fases_sel)} | "
          f"bootstrap B={bootstrap}"
          + (f" | {ahorro} se reutilizan de otra combinacion (misma huella)"
             if ahorro else ""))
    valores = paralelo.mapear(tareas, trabajos, etiqueta="evaluaciones")

    por_celda: dict[tuple[str, Celda], dict] = {}
    for grupo in grupos.values():
        salida = valores[grupo[0].nombre]
        for paso in grupo:
            por_celda[(paso.combinacion, paso.celda)] = salida

    resultados: dict[str, dict] = {}
    for combinacion in dict.fromkeys(p.combinacion for p in pasos):
        salida = evaluar.tablas_vacias()
        for celda in _orden_informe(formulaciones, entidades, normalizaciones,
                                    distancias_sel):
            parcial = por_celda.get((combinacion, celda))
            if parcial is None:
                continue
            for tabla, filas in parcial.items():
                salida[tabla].extend(filas)
        resultados[combinacion] = salida

    if "3" in fases_sel:
        _evaluar_fase3(pasos, db_path, resultados, trabajos, reutilizar,
                       formulaciones, entidades, normalizaciones, detalle,
                       distancias_sel)
    return resultados


def _evaluar_fase3(
    pasos: list[Paso], db_path: Path, resultados: dict[str, dict],
    trabajos: int, reutilizar: bool,
    formulaciones: tuple[str, ...], entidades: tuple[str, ...],
    normalizaciones: tuple[str, ...], detalle: bool,
    distancias_sel: tuple[str, ...] = ecfg.DISTANCIAS_POR_DEFECTO,
) -> None:
    """Añade la Fase 3 (triangulacion) a cada combinacion, como tarea aparte.

    No cabe en una celda: compara entre si las S de todos los modelos de la
    combinacion. Su identidad, y por tanto su deduplicacion, es el conjunto de
    huellas de esos modelos.
    """
    por_combinacion: dict[str, list[Paso]] = {}
    for paso in pasos:
        por_combinacion.setdefault(paso.combinacion, []).append(paso)

    grupos: dict[str, list[str]] = {}
    for nombre, suyos in por_combinacion.items():
        clave = (json.dumps(sorted(p.clave_artefacto for p in suyos))
                 if reutilizar else nombre)
        grupos.setdefault(clave, []).append(nombre)

    tareas = []
    for combinaciones_iguales in grupos.values():
        lider = combinaciones_iguales[0]
        model_dir = por_combinacion[lider][0].model_dir
        valores = por_combinacion[lider][0].valores
        tareas.append(paralelo.Tarea(
            f"{lider} fase3", trabajo.evaluar_fase3,
            (db_path, model_dir, valores, formulaciones, entidades,
             normalizaciones, detalle, distancias_sel),
        ))
    print(f"\n[evaluar] Fase 3: {len(tareas)} triangulaciones para "
          f"{len(por_combinacion)} combinaciones.")
    filas = paralelo.mapear(tareas, trabajos, etiqueta="triangulaciones")

    for combinaciones_iguales in grupos.values():
        for nombre in combinaciones_iguales:
            resultados[nombre]["f3"] = filas[f"{combinaciones_iguales[0]} fase3"]


def escribir_combinacion(
    nombre: str, salida: dict, out_dir: Path, con_figuras: bool = True
) -> None:
    """Vuelca los CSV (y las figuras) de una combinacion en `<out>/<nombre>/`."""
    var_dir = Path(out_dir) / nombre
    var_dir.mkdir(parents=True, exist_ok=True)
    # La combinacion va como columna en los CSV: los ficheros de dos de ellas se
    # llaman igual y contienen los mismos nombres de modelo, asi que sin ella
    # solo los distingue la carpeta que los contiene.
    evaluar.escribir_csv(salida, var_dir, {"combinacion": nombre})
    if con_figuras:
        evaluar.escribir_figuras(salida, var_dir)


def evaluar_combinacion(
    nombre: str,
    valores: dict[str, object],
    db_path: Path,
    out_dir: Path,
    fases_sel: set[str],
    bootstrap: int,
    con_figuras: bool = True,
    rehacer: bool = False,
    adoptar: bool = False,
    formulaciones: tuple[str, ...] = ecfg.FORMULACIONES,
    entidades: tuple[str, ...] = ecfg.ENTIDADES,
    normalizaciones: tuple[str, ...] = ecfg.NORMALIZACIONES,
    holdout: tuple[str, ...] = (),
    distancias_sel: tuple[str, ...] = ecfg.DISTANCIAS_POR_DEFECTO,
) -> dict:
    """Construye y evalua UNA combinacion de principio a fin, en este proceso.

    Camino de una sola combinacion, sin pool ni deduplicacion entre combinaciones
    (no hay ninguna otra con la que compartir). `main` no pasa por aqui: planifica
    todas juntas para poder compartir el trabajo. Devuelve la tabla `salida`.
    """
    var_dir = Path(out_dir) / nombre
    model_dir = var_dir / "modelo"
    var_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"COMBINACION '{nombre}' -> {_fmt_valores(valores)}")
    print(f"{'=' * 70}")

    with _config_temporal(valores):
        print("[construir] rejilla de modelos ...")
        conteo = _construir_grid(db_path, model_dir, out_dir, rehacer, adoptar,
                                 formulaciones, entidades, normalizaciones,
                                 holdout=holdout, distancias_sel=distancias_sel)
        print("    " + ", ".join(f"{v} {k}" for k, v in conteo.items() if v))
        celdas = [c for c in trabajo.celdas(formulaciones, entidades,
                                            normalizaciones, distancias_sel)
                  if huella.artefacto_completo(model_dir, c.stem)]
        if not celdas:
            raise SystemExit(f"No se construyo ningun modelo para {nombre!r}")

    pasos = [Paso(combinacion=nombre, valores=dict(valores), celda=celda,
                  model_dir=model_dir, huella={}, via="cache")
             for celda in celdas]
    salida = evaluar_plan(pasos, db_path, fases_sel, bootstrap, trabajos=1,
                          reutilizar=False, formulaciones=formulaciones,
                          entidades=entidades,
                          normalizaciones=normalizaciones,
                          holdout=holdout, distancias_sel=distancias_sel)[nombre]
    escribir_combinacion(nombre, salida, out_dir, con_figuras)
    return salida


# --------------------------------------------------------------------------- #
# Resumen comparativo (limpio: sin umbrales ni referencias de la documentacion) #
# --------------------------------------------------------------------------- #

# Los modelos se identifican por su CUATERNA (formulacion, entidad,
# normalizacion, distancia), no por la etiqueta compacta `F2_jugador_global`: en
# las tablas comparativas esa etiqueta sola no dice de que eje viene cada palabra,
# y la misma etiqueta designa modelos distintos en combinaciones distintas.
def _id_modelo(r) -> tuple[str, str, str, str]:
    return (r["formulacion"], r["entidad"], r["normalizacion"],
            r.get("distancia") or ecfg.DISTANCIAS_POR_DEFECTO[0])


def _extraer(tabla: str, campo: str, filtro=None):
    """Extractor {tripleta -> valor} de una metrica dentro de una tabla."""
    def extractor(s: dict) -> dict:
        return {_id_modelo(r): r[campo] for r in s[tabla]
                if filtro is None or filtro(r)}
    return extractor


def _solo_global(r) -> bool:
    """La Fase 1 reporta tres direcciones; la comparativa usa la media."""
    return r["direccion"] == "global"


def _fuera_de_muestra(r) -> bool:
    """La Fase 7 reporta dos ambitos; la comparativa usa el de fuera."""
    return r["ambito"] == "fuera"




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
    # Fase 7: la MISMA metrica de arriba, medida sobre entidades que el ajuste no
    # vio. Es su valor absoluto fuera de muestra (`gen_<metrica>`), no una brecha:
    # un modelo malo dentro y fuera tendria brecha cero sin generalizar nada. Que
    # metricas tienen gemela lo decide `generalizacion.METRICAS`.
    *[("7", generalizacion.nombre_metrica(m),
       _extraer("f7", m, _fuera_de_muestra))
      for m in generalizacion.METRICAS],
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
    # Las mismas metricas de arriba, sobre la liga excluida del ajuste. Se listan
    # en el orden de `METRICAS` para que se lean en paralelo a sus gemelas.
    ("Generalizacion: las mismas metricas FUERA de muestra", [
        (f"{titulo} (fuera de muestra)", generalizacion.nombre_metrica(m))
        for m, titulo in (
            ("top1", "Auto-similitud top-1"),
            ("mrr", "Auto-similitud MRR"),
            ("pureza_top1", "Pureza top-1"),
            ("knn_accuracy", "Clasificacion posicional k-NN (accuracy)"),
            ("rbo_medio", "Estabilidad RBO@10 ante remuestreo"),
            ("coverage", "Cobertura"),
            ("diversity", "Diversidad intra-lista"),
        )
    ]),
]


def _es_finito(x) -> bool:
    return isinstance(x, (int, float)) and x == x and x not in (float("inf"), float("-inf"))


# Columnas que identifican el modelo al principio de cada tabla comparativa.
_COLUMNAS_ID = ("formulacion", "entidad", "normalizacion", "distancia")


def _celdas_id(clave: tuple[str, str, str, str]) -> list[str]:
    form, entidad, norm, dist = clave
    return [f"F{form}", entidad, norm, dist]


def _orden_modelo(clave: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    """Agrupa por entidad (jugador y equipo no son comparables entre si).

    Dentro de la entidad, la distancia va antes que la formulacion: las filas de
    una misma geometria salen juntas, que es como se leen (comparar F2 con F5
    tiene sentido dentro de una distancia; entre distancias lo que se compara es
    la geometria).
    """
    form, entidad, norm, dist = clave
    return (entidad, dist, form, norm)


def _valores_de(df: pd.DataFrame, metrica: str) -> dict:
    """{(combinacion, cuaterna) -> valor finito} de una metrica de la tabla larga."""
    sub = df[df["metrica"] == metrica]
    valores: dict[tuple[str, tuple[str, str, str, str]], float] = {}
    for r in sub.itertuples(index=False):
        if _es_finito(r.valor):
            clave = (r.combinacion,
                     (r.formulacion, r.entidad, r.normalizacion, r.distancia))
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
        # Solo las columnas publicadas: el CSV acumulado puede traer combinaciones
        # que el registro de la carpeta ya no conoce (un `combinaciones.json`
        # perdido), y esas no tienen columna sobre la que promediar.
        if c in por_col:
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
    cuatro ejes (ademas de la etiqueta compacta, para poder filtrar por cualquiera).

    La etiqueta lleva la distancia SIEMPRE, tambien `euclidea`. Es lo que hace que
    `figuras3d` dibuje una figura por geometria en vez de mezclarlas, y lo que
    permite que la carpeta acumulativa reciba modelos de varias distancias sin que
    `registro.acumular` pise unas filas con otras (`CLAVE_METRICA` incluye
    `modelo`).
    """
    filas = []
    for combinacion, s in resultados.items():
        for fase, metrica, extractor in ESPECS:
            for clave, valor in extractor(s).items():
                form, entidad, norm, dist = clave
                filas.append({"combinacion": combinacion,
                              "modelo": f"F{form}_{entidad}_{norm}_{dist}",
                              "formulacion": form, "entidad": entidad,
                              "normalizacion": norm, "distancia": dist,
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
             "ultima ejecucion. El nombre `vNN` identifica una CONFIGURACION (se "
             "asigna en `combinaciones.json`), asi que significa lo mismo en todas "
             "las ejecuciones y en todas las figuras de la carpeta.\n")
    if not ejes:
        L.append("No hay hiperparametros declarados: una sola combinacion, con la "
                 "configuracion por defecto de `src/similitud/config.py`.\n")
        return

    L.append("| combinacion | " + " | ".join(ejes) + " | evaluada |")
    L.append("|" + "---|" * (len(ejes) + 2))
    # Marcar "esta ejecucion" solo distingue algo si la carpeta trae ademas
    # combinaciones de ejecuciones anteriores.
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
        cuando = proc.get(nombre, {}).get(registro.CAMPO_FECHA, "?")
        marca = (f"{cuando} *(esta ejecucion)*"
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
                 "`combinaciones.json`). Las metricas de ejecuciones distintas se "
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
    en memoria de esta ejecucion: es lo que permite que el resumen incluya las
    combinaciones que se evaluaron en ejecuciones anteriores.

    Con `meta["regenerado"]` la cabecera dice que la pasada no evaluo nada: el
    resumen se reescribe desde el CSV ya guardado (lo hace
    `src.evaluacion.migrar_distancia --regenerar-resumen`), y anunciar unas fases
    y un bootstrap que no se han corrido seria falso.
    """
    configs = registro.hiperparametros(reg)
    ejes = meta["ejes"]
    evaluadas = set(meta.get("evaluadas", ()))

    L: list[str] = []
    L.append("# Barrido de hiperparametros — comparativa entre combinaciones\n")
    if meta.get("regenerado"):
        L.append(f"Regenerado el {meta['fecha']} desde `barrido_metricas.csv`, sin "
                 "construir ni evaluar nada: las tablas se reescriben con las "
                 f"metricas ya acumuladas en la carpeta (**{len(configs)} "
                 "combinaciones**), cada una de la ejecucion en que se midio.\n")
    else:
        L.append(f"Generado el {meta['fecha']}. La carpeta acumula **{len(configs)} "
                 f"combinaciones**; esta ejecucion evaluo {len(evaluadas)} de ellas "
                 f"(fases {meta['fases']}, bootstrap B={meta['bootstrap']}) y las "
                 "demas conservan las metricas de la ejecucion en que se "
                 "evaluaron.\n")
    if meta.get("rejilla"):
        L.append(f"Rejilla construida y evaluada en esta ejecucion: "
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


def _metodos_f5(pedido: str) -> tuple[str, ...]:
    """Valores de `--metodo-f5`. Vacio = el default POR ENTIDAD del repo.

    La tupla vacia no es "ningun metodo": es "no toques la eleccion por entidad de
    `similitud.config`" (jugador `mmd`, equipo `sinkhorn`), que es el
    comportamiento de siempre y no anade ningun eje a la rejilla.
    """
    if not pedido.strip():
        return ()
    return _subconjunto(pedido, ecfg.METODOS_F5, "metodos de la F5")


def _avisos_distancias(
    distancias_sel: tuple[str, ...], formulaciones: tuple[str, ...],
    entidades: tuple[str, ...],
) -> list[str]:
    """Avisos de `--distancias`: lo que cuesta y lo que deja de valer.

    Ninguno aborta —comparar geometrias es el sentido de la flag—, pero las tres
    situaciones sorprenden si no se dicen antes de empezar horas de trabajo.
    """
    avisos: list[str] = []
    otras = [d for d in distancias_sel if d != ecfg.DISTANCIAS_POR_DEFECTO[0]]
    if not otras:
        return avisos
    if "manhattan" in distancias_sel and "2" in formulaciones and "jugador" in entidades:
        avisos.append(
            "--distancias manhattan sobre la F2 de jugador es el caso caro: la L1 "
            "no tiene la identidad del producto escalar, asi que el kNN de "
            "candidatos pasa de un GEMM a O(M^2 d) fila a fila sobre ~50.000 "
            "observaciones (horas por artefacto). En la F5 no aplica: el ancho del "
            "kernel se estima sobre una submuestra y el RFF es lineal."
        )
    if "manhattan" in distancias_sel and "2" in formulaciones:
        avisos.append(
            "con manhattan el residuo de SLIM se resuelve por IRLS "
            "(`slim.elasticnet_residuo_l1`), no en forma cerrada: F2_BETA y F2_L1 "
            "quedan en otra escala efectiva y los valores afinados para la "
            "euclidea no son trasladables — hay que barrerlos de nuevo."
        )
    if "mahalanobis" in distancias_sel and "jugador" in entidades:
        avisos.append(
            "mahalanobis blanquea el espacio, asi que deshace la ponderacion del "
            "bloque de posicion (POSITION_SCALING='zscore_sqrt', las 25 columnas "
            "pos_* divididas por sqrt(25) para pesar como una sola feature): tras "
            "blanquear vuelven a pesar lo que diga la covarianza."
        )
    return avisos


def _avisos_metodo_f5(
    metodos: tuple[str, ...], formulaciones: tuple[str, ...],
    entidades: tuple[str, ...], fases: set[str],
) -> list[str]:
    """Avisos de `--metodo-f5`: cuando no hace nada y cuando cuesta caro.

    Ninguno aborta. Forzar el metodo es un experimento legitimo en las dos
    direcciones —y medir cuanto cuesta es parte de lo que se quiere ver—, pero las
    tres situaciones sorprenden si no se dicen antes de empezar.
    """
    if not metodos:
        return []
    avisos: list[str] = []
    if "5" not in formulaciones:
        avisos.append(
            "--metodo-f5 no tiene efecto con --formulaciones sin la 5: el metodo "
            "distribucional solo existe en la etapa 1 de la F5."
        )
        return avisos
    if "sinkhorn" in metodos and "jugador" in entidades:
        avisos.append(
            "--metodo-f5 sinkhorn sobre jugador es el caso caro: el coste OT es "
            "cuadratico en observaciones (P(P-1)/2 pares de nubes), unas 3,6 h por "
            "artefacto con la BD base frente a los ~4 s del mmd. Con equipo son "
            "~30 s."
        )
    if "sinkhorn" in metodos and "7" in fases:
        avisos.append(
            "la fase 7 no aplica a los modelos F5 con sinkhorn: no se saben "
            "proyectar fuera de muestra (`foldin.METODOS_PROYECTABLES`), asi que "
            "se saltaran con un motivo y se quedaran sin metricas `gen_*`."
        )
    return avisos


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
    p.add_argument("--normalizaciones", type=str,
                   default=",".join(ecfg.NORMALIZACIONES),
                   help="Normalizaciones de la rejilla (coma-separado). "
                        f"Disponibles: {', '.join(ecfg.NORMALIZACIONES)}. Por "
                        "defecto se construyen y evaluan LAS DOS, que es lo que "
                        "permite compararlas; acotarla a una divide por dos el "
                        "coste cuando la comparacion no es el objetivo de esa "
                        "tanda.")
    p.add_argument("--distancias", type=str,
                   default=",".join(ecfg.DISTANCIAS_POR_DEFECTO),
                   help="Distancias entre observaciones de la rejilla "
                        f"(coma-separado). Disponibles: {', '.join(ecfg.DISTANCIAS)}. "
                        "Es el CUARTO eje de la rejilla y se comporta como la "
                        "normalizacion: cada valor da un artefacto propio, una fila "
                        "propia en las tablas y una FIGURA propia en las superficies "
                        "3D (no un eje dentro de la misma figura). Por defecto solo "
                        "la euclidea, que es la geometria de todo lo acumulado en "
                        "outputs/: son cuatro valores y uno de ellos (manhattan sobre "
                        "la F2 de jugador) se va a horas.")
    p.add_argument("--metodo-f5", type=str, default="",
                   help="Metodo de la etapa 1 de la F5 (coma-separado). "
                        f"Disponibles: {', '.join(ecfg.METODOS_F5)}. Por defecto "
                        "se respeta la eleccion POR ENTIDAD de "
                        "src/similitud/config.py (jugador mmd, equipo sinkhorn). "
                        "Indicarlo lo fuerza en las dos entidades, y con dos "
                        "valores ('mmd,sinkhorn') el metodo pasa a ser un EJE mas "
                        "de la rejilla: entra en la identidad de la combinacion y "
                        "en la huella del artefacto, asi que no pisa lo ya "
                        "acumulado en la carpeta.")
    p.add_argument("--fases", type=str, default="0,1,5",
                   help="Fases a ejecutar por combinacion (coma-separado). El "
                        "barrido multiplica el coste, por eso omite por defecto "
                        "la estabilidad (2) y la triangulacion (3).")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Remuestreos de la Fase 2 (0 la omite; solo aplica si se "
                        "incluye la fase 2).")
    p.add_argument("--holdout", default="",
                   help="Ligas que quedan FUERA DEL AJUSTE (por clave `1238-108` "
                        "o por un trozo del nombre, `india`). Los modelos del "
                        "barrido se construyen sin ellas y la fase 7 las usa para "
                        "medir la generalizacion sobre ese mismo artefacto. Sin "
                        "esto no se puede correr la fase 7 y el score se queda sin "
                        "sus metricas de generalizacion. Solo tiene efecto si se "
                        "pide la fase 7: excluir una liga sin medir nada con ella "
                        "seria tirar datos.")
    p.add_argument("--sin-figuras", action="store_true")
    p.add_argument("--trabajos", type=str, default="1",
                   help=f"Procesos que construyen y evaluan a la vez ('{paralelo.AUTO}' "
                        "= uno por nucleo). Las celdas (combinacion x formulacion x "
                        "entidad x normalizacion) son independientes, asi que se "
                        "reparten entre ellos. Ojo con la memoria: cada trabajador "
                        "tiene su propia copia de la matriz de features y de la S que "
                        "este reconstruyendo. Con 1 (por defecto) no se crea ningun "
                        "proceso y el log sale en directo.")
    p.add_argument("--sin-reutilizar-metricas", action="store_true",
                   help="Evalua cada celda por separado aunque otra combinacion "
                        "tenga un modelo con la MISMA huella. Por defecto se evalua "
                        "una vez por huella y el resultado se reparte: el modelo es "
                        "el mismo artefacto y las fases son deterministas. Esta flag "
                        "es la forma de comprobarlo (y de rehacerlo todo si se duda).")
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

    # El metodo de la F5 se pide por la linea de comandos, pero se comporta como
    # un hiperparametro mas: se inyecta como eje y desde aqui abajo viaja con los
    # declarados en HIPERPARAMETROS.
    metodos_f5 = _metodos_f5(args.metodo_f5)
    ejes_metodo = _ejes_metodo_f5(metodos_f5) if metodos_f5 else []

    _validar_hiperparametros(ejes_metodo)
    trabajos = paralelo.resolver_trabajos(args.trabajos)
    formulaciones = _subconjunto(args.formulaciones, ecfg.FORMULACIONES, "formulaciones")
    entidades = _subconjunto(args.entidades, ecfg.ENTIDADES, "entidades")
    normalizaciones = _subconjunto(args.normalizaciones, ecfg.NORMALIZACIONES,
                                   "normalizaciones")
    distancias_sel = _subconjunto(args.distancias, ecfg.DISTANCIAS, "distancias")
    fases_sel = {s.strip() for s in args.fases.split(",") if s.strip()}
    for aviso in _avisos_metodo_f5(metodos_f5, formulaciones, entidades, fases_sel):
        print(f"  [aviso] {aviso}")
    for aviso in _avisos_distancias(distancias_sel, formulaciones, entidades):
        print(f"  [aviso] {aviso}")

    db_path = args.db
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    # Mismo criterio que en `evaluar`: la fase 7 sin ligas excluidas no mide nada,
    # asi que se para en vez de dejar el score sin su termino en silencio.
    holdout = evaluar._holdout_pedido(db_path, args.holdout, fases_sel)

    # La carpeta es acumulativa: su registro decide como se llama cada
    # configuracion, para que los `vNN` de esta ejecucion sigan designando lo mismo
    # que los de las anteriores (ver `registro.py`).
    reg = registro.cargar(out_dir)
    ejes = _ejes_publicados(registro.ejes(reg), ejes_metodo)
    avisos = registro.completar_ejes(
        out_dir, reg, ejes, {eje: getattr(scfg, eje) for eje in ejes}
    )
    for aviso in avisos:
        print(f"  [aviso] eje sin registrar en una combinacion anterior: {aviso}")

    conocidas = set(reg["combinaciones"])
    efectivas = [_config_efectiva(v, ejes)
                 for v in configuraciones(formulaciones, entidades, ejes_metodo)]
    seleccion = registro.nombrar(reg, efectivas, ejes)
    nuevas = [n for n in seleccion if n not in conocidas]

    print(f"{len(seleccion)} combinaciones en esta ejecucion "
          f"({len(nuevas)} nuevas, {len(seleccion) - len(nuevas)} ya conocidas); "
          f"{len(reg['combinaciones'])} acumuladas en {out_dir}:")
    for nombre, valores in seleccion.items():
        marca = " [nueva]" if nombre in nuevas else ""
        print(f"  {nombre}: {_fmt_valores(valores)}{marca}")

    t0 = time.perf_counter()

    # Todo el plan antes de tocar nada: es lo que permite ajustar y evaluar una
    # sola vez lo que varias combinaciones comparten, y repartir el resto.
    print("\nPlanificando la ejecucion (huella de cada celda)...")
    pasos = planificar(seleccion, db_path, out_dir, formulaciones, entidades,
                       normalizaciones, args.rehacer, args.adoptar_existentes,
                       holdout=holdout, distancias_sel=distancias_sel)

    conteo = construir_plan(pasos, db_path, trabajos, rehacer=args.rehacer,
                            holdout=holdout)
    print("[construir] " + ", ".join(f"{v} {k}" for k, v in conteo.items() if v))

    resultados = evaluar_plan(
        pasos, db_path, fases_sel, args.bootstrap, trabajos,
        reutilizar=not args.sin_reutilizar_metricas,
        formulaciones=formulaciones, entidades=entidades,
        normalizaciones=normalizaciones, holdout=holdout,
        distancias_sel=distancias_sel,
    )
    for nombre, salida in resultados.items():
        escribir_combinacion(nombre, salida, out_dir,
                             con_figuras=not args.sin_figuras)

    fecha = time.strftime("%Y-%m-%d %H:%M")
    registro.anotar(reg, list(seleccion),
                    {registro.CAMPO_FECHA: fecha, **huella.procedencia(db_path)})
    registro.guardar(out_dir, reg)

    # Las metricas de esta ejecucion sustituyen a las suyas anteriores; las de las
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
                   f"{list(normalizaciones)} x distancias "
                   f"{list(distancias_sel)}",
    })
    dt = time.perf_counter() - t0
    n_combis = len(set(acumuladas["combinacion"])) if not acumuladas.empty else 0
    print(f"\nListo en {dt:.0f}s. Comparativa de las {n_combis} combinaciones "
          f"acumuladas en {out_dir}/resumen_barrido.md")
    print(f"Superficies 3D con todos esos puntos: "
          f"python -m src.evaluacion.figuras3d --barrido {out_dir}")


if __name__ == "__main__":
    main()

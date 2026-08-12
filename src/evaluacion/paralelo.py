"""Ejecucion de tareas independientes en varios procesos.

Pieza generica (no sabe nada de fases ni de modelos) sobre la que el barrido
reparte su trabajo: recibe una lista de `Tarea` —cada una, una funcion de modulo
y sus argumentos— y las ejecuta con N trabajadores, devolviendo lo que cada una
produjo.

**Por que procesos y no hilos.** Las fases de la evaluacion son CPU-bound y buena
parte del coste esta en bucles de Python puro (`fase0_sanity` y `fase5_downstream`
recorren las P entidades una a una; `fase2_estabilidad` rankea B x P veces), no en
llamadas a BLAS que suelten el GIL. Con hilos, N tareas tardarian lo mismo que en
serie. Ademas cada combinacion del barrido FIJA los hiperparametros en
`src.similitud.config`, que es estado global del proceso: dos combinaciones no
pueden convivir en el mismo interprete sin pisarse los valores. Los procesos
resuelven las dos cosas a la vez.

**Por que `spawn` y por que las tareas viven en `trabajo.py`.** En Windows el
unico metodo disponible es `spawn`: el hijo arranca un interprete limpio, importa
el modulo de la funcion y desempaqueta los argumentos. Por eso la funcion de una
tarea tiene que ser de nivel de modulo (se serializa por NOMBRE, no por codigo) y
por eso no puede vivir en el modulo que se ejecuta con `python -m` (seria
`__main__` en el padre y otra cosa en el hijo). Los argumentos tienen que ser
datos sencillos: rutas, cadenas, diccionarios.

**Salida por bloques.** Con varios procesos escribiendo a la vez, la consola seria
ilegible. Cada tarea captura su propia salida y la devuelve con el resultado; el
padre la imprime entera cuando la tarea termina, precedida de su nombre. A cambio
se pierde el directo: una tarea de diez minutos no dice nada hasta acabar. Por eso
`trabajos=1` NO monta pool ninguno y ejecuta en este mismo proceso, dejando el log
tal cual sale (es ademas el modo en que se depura: un traceback en el proceso
padre es un traceback normal).

**Hilos de BLAS.** numpy reparte los productos matriciales entre todos los nucleos.
Con N procesos haciendo lo mismo, el sistema acaba con N x nucleos hilos
peleandose por la CPU y el conjunto va MAS lento que con menos trabajadores. Antes
de crear el pool se reparten los nucleos entre los trabajadores por las variables
de entorno que leen las librerias (las heredan los hijos al arrancar). Si el
usuario ya las tiene puestas no se tocan: es una decision suya y sabra por que.
"""

from __future__ import annotations

import io
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Callable, Sequence

from .evaluar import _dur

# Valor de `--trabajos` que significa "uno por nucleo".
AUTO = "auto"

# Variables que leen las librerias de algebra lineal al cargarse. Se reparten
# entre los trabajadores para no sobresuscribir la CPU (ver docstring).
_VARS_HILOS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# --------------------------------------------------------------------------- #
# Unidad de trabajo                                                            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Tarea:
    """Una unidad de trabajo independiente.

    `funcion` tiene que ser una funcion de nivel de modulo (se serializa por
    nombre) y `args` datos serializables. `nombre` identifica la tarea en el log y
    en el diccionario de resultados, asi que tiene que ser unico en la lista.
    """

    nombre: str
    funcion: Callable[..., Any]
    args: tuple = ()


@dataclass
class Resultado:
    """Lo que devolvio una tarea, mas su log y su tiempo."""

    nombre: str
    valor: Any = None
    log: str = ""
    segundos: float = 0.0
    error: str | None = None          # traceback formateado si la tarea fallo


# --------------------------------------------------------------------------- #
# Numero de trabajadores                                                       #
# --------------------------------------------------------------------------- #

def resolver_trabajos(pedido: str | int | None) -> int:
    """Traduce el valor de `--trabajos` a un numero de procesos.

    `auto` = un trabajador por nucleo. No se aplica ningun tope por memoria:
    cuanto ocupa una tarea depende del tamaño del pool de entidades, y estimarlo
    mal en un sentido malgasta la maquina y en el otro la tumba; es el usuario
    quien decide (ver `docs/evaluacion.md`). El tope por numero de tareas si lo
    pone `mapear`, que es quien las conoce.
    """
    if pedido is None:
        n = 1
    elif isinstance(pedido, str):
        texto = pedido.strip().lower()
        if texto == AUTO:
            n = os.cpu_count() or 1
        else:
            try:
                n = int(texto)
            except ValueError:
                raise SystemExit(
                    f"--trabajos: se esperaba un entero o {AUTO!r}, no {pedido!r}."
                )
    else:
        n = int(pedido)
    if n < 1:
        raise SystemExit(f"--trabajos tiene que ser >= 1 (se pidio {n}).")
    return n


@contextmanager
def _repartir_hilos(trabajos: int):
    """Reparte los nucleos entre los trabajadores mientras dure el pool.

    Los hijos heredan el entorno al arrancar, asi que basta con fijar las
    variables en el padre antes de crear el pool (y restaurarlas despues: el padre
    sigue vivo para el resto de la ejecucion). Solo se tocan las que no venian ya
    puestas.
    """
    por_proceso = max(1, (os.cpu_count() or trabajos) // max(1, trabajos))
    nuevas = [v for v in _VARS_HILOS if v not in os.environ]
    for var in nuevas:
        os.environ[var] = str(por_proceso)
    try:
        yield por_proceso if nuevas else None
    finally:
        for var in nuevas:
            os.environ.pop(var, None)


# --------------------------------------------------------------------------- #
# Ejecucion                                                                     #
# --------------------------------------------------------------------------- #

def _ejecutar_capturando(tarea: Tarea) -> Resultado:
    """Corre una tarea EN EL TRABAJADOR y devuelve su valor, su log y su tiempo.

    La excepcion no se propaga como excepcion: se devuelve ya formateada. Asi el
    padre puede imprimir el log de la tarea (donde esta el contexto de lo que
    estaba haciendo) ANTES de abortar; una excepcion cruda que viaja entre
    procesos llega sin nada de eso.
    """
    buffer = io.StringIO()
    t0 = time.perf_counter()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            valor = tarea.funcion(*tarea.args)
        error = None
    except BaseException:                      # noqa: BLE001 - se reenvia tal cual
        valor, error = None, traceback.format_exc()
    return Resultado(
        nombre=tarea.nombre,
        valor=valor,
        log=buffer.getvalue(),
        segundos=time.perf_counter() - t0,
        error=error,
    )


def _imprimir(res: Resultado, hecho: int, total: int, t0: float) -> None:
    """Linea de progreso de una tarea terminada + su log, indentado."""
    transcurrido = time.perf_counter() - t0
    restantes = total - hecho
    linea = (f"[{hecho}/{total}] {res.nombre} | {_dur(res.segundos)} | "
             f"transcurrido {_dur(transcurrido)}")
    if restantes > 0:
        linea += f" | quedan ~{_dur((transcurrido / hecho) * restantes)}"
    print(linea, flush=True)
    for ln in res.log.splitlines():
        print(f"    | {ln}", flush=True)


def _abortar(res: Resultado) -> None:
    raise RuntimeError(
        f"la tarea {res.nombre!r} fallo en un trabajador:\n{res.error}"
    )


def mapear(
    tareas: Sequence[Tarea],
    trabajos: int = 1,
    etiqueta: str = "tareas",
) -> dict[str, Any]:
    """Ejecuta las tareas y devuelve {nombre -> valor devuelto}.

    Con `trabajos == 1` no se crea ningun proceso: las tareas corren aqui, en
    orden, y su salida va directa a la consola. Con mas, se reparten entre
    procesos y cada una imprime su bloque al terminar (el orden de terminacion no
    es el de la lista; el diccionario de vuelta no depende de el).

    Si una tarea falla se imprime su log y se aborta la ejecucion entera: los
    resultados a medias del barrido irian a parar a `barrido_metricas.csv`, que es
    acumulativo, y quedarian ahi como si fuesen buenos.
    """
    tareas = list(tareas)
    if not tareas:
        return {}
    nombres = [t.nombre for t in tareas]
    if len(set(nombres)) != len(nombres):
        raise ValueError("las tareas tienen que tener nombres unicos")

    total = len(tareas)
    t0 = time.perf_counter()
    valores: dict[str, Any] = {}
    # Mas trabajadores que tareas son procesos que solo pagarian su arranque.
    trabajos = min(trabajos, total)

    if trabajos <= 1:
        print(f"{total} {etiqueta}, en serie (--trabajos 1).", flush=True)
        for i, tarea in enumerate(tareas, start=1):
            print(f"\n[{i}/{total}] {tarea.nombre}", flush=True)
            t = time.perf_counter()
            valores[tarea.nombre] = tarea.funcion(*tarea.args)
            print(f"      hecho en {_dur(time.perf_counter() - t)}", flush=True)
        return valores

    with _repartir_hilos(trabajos) as hilos:
        reparto = f", {hilos} hilo(s) de BLAS cada uno" if hilos else ""
        print(f"{total} {etiqueta} con {trabajos} trabajadores{reparto}. "
              f"Cada una imprime su bloque al terminar.", flush=True)
        contexto = get_context("spawn")
        with ProcessPoolExecutor(max_workers=trabajos, mp_context=contexto) as pool:
            futuros = {pool.submit(_ejecutar_capturando, t): t for t in tareas}
            hecho = 0
            for futuro in as_completed(futuros):
                res = futuro.result()
                hecho += 1
                _imprimir(res, hecho, total, t0)
                if res.error is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
                    _abortar(res)
                valores[res.nombre] = res.valor
    return valores

"""Unidades de trabajo del barrido, tal y como las ve un trabajador.

Aqui vive lo que se ejecuta —construir un artefacto, evaluar un modelo, correr la
triangulacion de una combinacion—; el reparto y la planificacion son de
`src.evaluacion.paralelo` y `src.evaluacion.barrido`. La separacion no es
cosmetica: con `spawn` (el unico metodo de arranque en Windows) el proceso hijo
importa el modulo donde vive la funcion, asi que estas funciones NO pueden estar
en `barrido.py`, que se ejecuta como `__main__`.

De ahi la forma de sus firmas: reciben datos sencillos (rutas, cadenas,
diccionarios de hiperparametros) y no objetos ya construidos. Un contexto o una
matriz S pesan cientos de MB; mandarlos por la tuberia costaria mas que
recalcularlos, y con `spawn` el hijo no hereda nada del padre.

**Que fija cada tarea.** Toda tarea empieza fijando en `src.similitud.config` los
hiperparametros de SU combinacion (`config_temporal`), porque todo el pipeline los
lee en tiempo de llamada. En un trabajador eso es ademas seguro por
construccion: cada proceso atiende una tarea a la vez.

**Caches por proceso.** Un trabajador atiende muchas tareas seguidas y varias
comparten trabajo previo: los roles y minutos son los mismos para toda la ejecucion
(salen de la BD), y la matriz de features de una celda solo depende de la entidad,
la normalizacion y los atributos de `huella.alcance_datos` — no de la formulacion
ni de los hiperparametros de ajuste. Reconstruirla en cada tarea seria releer la
BD y rehacer los z-scores decenas de veces. Se cachean, con un tope de contextos
vivos porque cada uno pesa lo que pesa la matriz X entera.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.similitud import build as sbuild
from src.similitud import config as scfg
from src.similitud import distancias, warm
from src.similitud.modelo import cargar_modelo, stem_artefacto

from . import config as ecfg
from . import construccion, datos, evaluar, fases, huella

# Contextos vivos a la vez en un trabajador. Cada uno guarda la matriz de
# features completa (y, si es de la F2, la W instancia-instancia), asi que el tope
# es lo que separa "reutilizar" de "quedarse sin memoria". Con las tareas
# ordenadas por (entidad, normalizacion) —que es como las emite el barrido— dos
# basta para encadenar las celdas de un mismo contexto sin acumular los cuatro.
MAX_CONTEXTOS = 2


# --------------------------------------------------------------------------- #
# Identificacion de una celda de la rejilla                                    #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, order=True)
class Celda:
    """Una celda de la rejilla: (formulacion, entidad, normalizacion, distancia).

    Es lo que identifica un artefacto y, por tanto, una unidad de construccion y
    una unidad de evaluacion.

    La `distancia` es el cuarto eje y se comporta como los otros tres: cambia el
    artefacto, cambia su nombre en disco y da una FILA propia en las tablas y una
    FIGURA propia en las superficies 3D. No es un hiperparametro (no es un valor
    que se afine dentro de un modelo), y por eso no entra en el producto
    cartesiano de `HIPERPARAMETROS` sino en `--distancias`, como
    `--normalizaciones`.
    """

    formulacion: str
    entidad: str
    normalizacion: str
    distancia: str = distancias.POR_DEFECTO

    @property
    def stem(self) -> str:
        """Nombre del artefacto, igual que lo genera `ModeloSimilitud.guardar`."""
        return stem_artefacto(self.formulacion, self.entidad,
                              self.normalizacion, self.distancia)

    @property
    def etiqueta(self) -> str:
        """Identificador del modelo en tablas, CSV y figuras.

        A diferencia del `stem`, aqui la distancia va SIEMPRE, tambien cuando es
        la euclidea: en disco callarla ahorra renombrar miles de ficheros, pero
        en un resultado no declarar con que geometria se midio es justo lo que
        haria incomparables dos filas de la misma tabla.
        """
        return (f"F{self.formulacion}_{self.entidad}_{self.normalizacion}"
                f"_{self.distancia}")


def celdas(
    formulaciones: tuple[str, ...] = ecfg.FORMULACIONES,
    entidades: tuple[str, ...] = ecfg.ENTIDADES,
    normalizaciones: tuple[str, ...] = ecfg.NORMALIZACIONES,
    distancias_sel: tuple[str, ...] = ecfg.DISTANCIAS_POR_DEFECTO,
) -> list[Celda]:
    """La rejilla en el orden en que la recorre el barrido.

    Ordenada por (entidad, normalizacion, distancia) para que las celdas que
    comparten contexto salgan seguidas y el cache del trabajador las encadene;
    dentro de un contexto, la F2 primero, que es la cara (empezar por lo largo
    acorta el tiempo total cuando hay varios trabajadores).

    La distancia va en el nivel del contexto porque lo cambia: el espacio en el
    que se miden los vecinos es parte de lo que el contexto cachea.
    """
    return [
        Celda(form, entidad, norm, dist)
        for entidad in entidades
        for norm in normalizaciones
        for dist in distancias_sel
        for form in formulaciones
    ]


# --------------------------------------------------------------------------- #
# Hiperparametros de la combinacion                                            #
# --------------------------------------------------------------------------- #

# `config_temporal` vive ahora en `evaluacion.config`, que es el modulo que
# importan tanto este como `evaluar` (que tambien lo necesita para medir cada
# artefacto con SUS hiperparametros). Se reexporta aqui porque este es el sitio
# donde el resto del proyecto lo busca y donde esta documentado su porque.
config_temporal = ecfg.config_temporal


def _valores_de(attrs: list[str]) -> str:
    """Forma canonica de un juego de atributos de config, para usar como clave."""
    return json.dumps({a: getattr(scfg, a) for a in attrs},
                      sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# Caches del proceso trabajador                                                #
# --------------------------------------------------------------------------- #

_DATOS: dict[str, tuple[dict, dict]] = {}
_CONTEXTOS: "OrderedDict[tuple, construccion.Contexto]" = OrderedDict()
_CLAVE_W: dict[tuple, str] = {}


def limpiar_caches() -> None:
    """Vacia las caches del proceso (las pruebas y el modo en serie lo usan)."""
    _DATOS.clear()
    _CONTEXTOS.clear()
    _CLAVE_W.clear()


def datos_auxiliares(db_path: Path) -> tuple[dict, dict]:
    """(roles, minutos) por jugador, una sola vez por proceso y BD."""
    clave = str(Path(db_path).resolve())
    if clave not in _DATOS:
        _DATOS[clave] = (datos.rol_por_jugador(db_path),
                         datos.minutos_por_jugador(db_path))
    return _DATOS[clave]


def contexto(db_path: Path, celda: Celda,
             excluir_ligas: tuple[str, ...] = ()) -> construccion.Contexto:
    """Contexto de la celda, reutilizando el del proceso si sigue valiendo.

    Se cachea por (BD, entidad, normalizacion, distancia, ligas excluidas,
    atributos de `alcance_datos`): dos celdas con esa clave igual tienen
    EXACTAMENTE la misma matriz de features Y la misma geometria, aunque ajusten
    formulaciones distintas o hiperparametros de ajuste distintos.

    La distancia entra en la clave porque el contexto no guarda solo X: guarda
    tambien el `Espacio` (el blanqueo de mahalanobis se estima de estos datos) y
    cachea la W de la F2, que se ajusta EN esa geometria. Reutilizar el contexto
    de otra distancia serviria una W medida con otra metrica.

    `excluir_ligas` tiene que ser el MISMO que se uso al construir el artefacto.
    Las fases reconstruyen la S desde este contexto y la comparan con la del
    modelo: con un contexto de 109 entidades y un artefacto de 98, no cuadraria
    nada — y es el tipo de fallo que sale como numeros raros, no como excepcion.

    La W instancia-instancia que el contexto cachea perezosamente NO entra en esa
    clave —depende de los `F2_*`—, asi que se descarta en cuanto la tarea pide la
    F2 con otros valores. Sin eso, reutilizar el contexto serviria una W ajustada
    con la lambda de otra combinacion: el fallo mas silencioso posible.

    Debe llamarse DENTRO de `config_temporal`.
    """
    etq = f"{celda.entidad}/{celda.normalizacion}/{celda.distancia}"
    clave = (str(Path(db_path).resolve()), celda.entidad, celda.normalizacion,
             celda.distancia, tuple(sorted(excluir_ligas)),
             _valores_de(huella.alcance_datos(celda.entidad)))
    ctx = _CONTEXTOS.get(clave)
    if ctx is None:
        print(f"[contexto] {etq}: se construye "
              f"(no esta en la cache del proceso) ...", flush=True)
        ctx = construccion.crear_contexto(
            db_path, celda.entidad, celda.normalizacion,
            excluir_ligas=tuple(excluir_ligas), distancia=celda.distancia)
        _CONTEXTOS[clave] = ctx
        while len(_CONTEXTOS) > MAX_CONTEXTOS:
            viejo, _ = _CONTEXTOS.popitem(last=False)
            _CLAVE_W.pop(viejo, None)
    else:
        _CONTEXTOS.move_to_end(clave)
        print(f"[contexto] {etq}: reutilizado del "
              f"proceso (misma matriz de features)", flush=True)

    if celda.formulacion == "2":
        clave_w = _valores_de(huella.alcance_ajuste("2", celda.entidad))
        if _CLAVE_W.get(clave) != clave_w:
            ctx._W = None
            _CLAVE_W[clave] = clave_w
    return ctx


# --------------------------------------------------------------------------- #
# Tareas                                                                       #
# --------------------------------------------------------------------------- #

def construir_celda(
    db_path: Path, model_dir: Path, celda: Celda,
    valores: dict[str, object], h: dict,
    excluir_ligas: tuple[str, ...] = (),
) -> str:
    """Ajusta el artefacto de la celda y le escribe su huella. Devuelve el stem.

    La huella se borra ANTES y se escribe DESPUES (mismo orden que el camino en
    serie): si el ajuste peta a mitad, el artefacto queda a medias pero sin huella
    y nadie lo reutilizara. `h` viene calculada por el planificador con esta misma
    configuracion; se pasa en vez de recalcularla para que el artefacto quede
    marcado con la huella que motivo construirlo.
    """
    model_dir = Path(model_dir)
    with config_temporal(valores):
        model_dir.mkdir(parents=True, exist_ok=True)
        huella.invalidar(model_dir, celda.stem)
        sbuild._construir_uno(db_path, model_dir, celda.formulacion,
                              celda.entidad, celda.normalizacion,
                              excluir_ligas=tuple(excluir_ligas),
                              distancia=celda.distancia)
        huella.escribir(model_dir, celda.stem, h)
    return celda.stem


def evaluar_celda(
    db_path: Path, model_dir: Path, celda: Celda, valores: dict[str, object],
    fases_sel: set[str], bootstrap: int, detalle: bool = False,
    holdout: tuple[str, ...] = (),
) -> dict[str, list]:
    """Corre las fases de un modelo ya construido; devuelve sus filas.

    `detalle` enciende el reporte del bucle interno (las B iteraciones del
    bootstrap, el ajuste SLIM...): solo tiene sentido cuando la salida va a una
    consola de verdad, es decir en el modo en serie. En un trabajador la salida se
    acumula en un buffer y esos refrescos, que en consola se sobrescriben, se
    convertirian en miles de lineas.

    `holdout` son las ligas del hold-out, y en el barrido significa que el
    artefacto se ha AJUSTADO sin ellas (ver `construir_celda`). Por eso el
    contexto se construye igual de recortado: las fases reconstruyen la S desde
    el y tiene que cuadrar con el modelo. La Fase 7 recibe ademas ese modelo y su
    estado warm, de modo que no vuelve a ajustar nada — ya es el modelo que no vio
    la liga.

    Va aqui —y no dentro de `valores`— porque no es un hiperparametro del modelo:
    es una eleccion del protocolo, la misma para todas las celdas de la ejecucion,
    igual que `fases_sel`.
    """
    with config_temporal(valores):
        modelo = cargar_modelo(model_dir, celda.formulacion, celda.entidad,
                               celda.normalizacion, celda.distancia)
        ctx = contexto(db_path, celda, excluir_ligas=holdout)
        roles, minutos = datos_auxiliares(db_path)
        prog = evaluar.Progreso(
            evaluar._pasos_totales(1, fases_sel, bootstrap, holdout), sub=detalle)
        return evaluar.evaluar_modelo(
            modelo, ctx, celda.formulacion, celda.entidad, celda.normalizacion,
            roles, minutos, fases_sel, bootstrap, prog,
            db_path=Path(db_path), holdout=holdout,
            ruta_warm=warm.ruta_estado(model_dir, celda.formulacion,
                                       celda.entidad, celda.normalizacion,
                                       celda.distancia),
            distancia=celda.distancia,
        )


def evaluar_fase3(
    db_path: Path, model_dir: Path, valores: dict[str, object],
    formulaciones: tuple[str, ...], entidades: tuple[str, ...],
    normalizaciones: tuple[str, ...] = ecfg.NORMALIZACIONES,
    detalle: bool = False,
    distancias_sel: tuple[str, ...] = ecfg.DISTANCIAS_POR_DEFECTO,
) -> list[dict]:
    """Triangulacion (Fase 3) de UNA combinacion; devuelve las filas de `f3`.

    Es la unica fase que no cabe en una celda: compara las S de modelos distintos
    entre si, asi que necesita a la vez todos los artefactos de la combinacion y
    los contextos de sus (entidad, normalizacion) —de ahi reconstruye la S
    distribucional cruda de la F5—. Por eso es su propia tarea, con el coste de
    memoria que eso implica.

    `normalizaciones` acota la rejilla igual que `formulaciones`/`entidades`: sin
    ello la triangulacion cargaria los artefactos de una normalizacion que esta
    ejecucion no ha pedido (y que pueden ser de una ejecucion anterior).

    Con la distancia el criterio es otro: **se triangula DENTRO de cada
    geometria**, una vez por distancia, y las filas se concatenan con su columna.
    Las comparaciones de esta fase son «F2 contra F5» y «por_liga contra global»,
    que solo significan algo si las dos matrices se midieron igual; cruzar dos
    distancias seria una pregunta distinta (cuanto se parecen dos geometrias), no
    esta.
    """
    with config_temporal(valores):
        filas: list[dict] = []
        prog = evaluar.Progreso(len(distancias_sel), sub=detalle)
        for dist in distancias_sel:
            modelos = evaluar.cargar_modelos(model_dir, formulaciones, entidades,
                                             normalizaciones, distancia=dist)
            if not modelos:
                continue
            ctxs = {
                (entidad, norm): contexto(db_path, Celda("5", entidad, norm, dist))
                for entidad in entidades
                for norm in normalizaciones
            }
            with evaluar._quizas_paso(
                    prog, f"[Fase 3] triangulacion (test de Mantel) {dist}"):
                filas.extend(
                    {**f, "distancia": dist}
                    for f in fases.fase3_triangulacion(
                        modelos, ctxs, evaluar._quizas_sub(prog, "comparacion"))
                )
        return filas

"""Fase 7 - Generalizacion: ¿y con una liga que el modelo no ha visto?

Todas las fases anteriores miden el modelo sobre las MISMAS entidades con las que
se ajusto. Eso responde a «¿ha aprendido algo?» pero no a la pregunta que se hace
quien lo va a usar: si entreno con las cinco grandes ligas y aparece un jugador de
la liga india, ¿le recomienda parecidos con el mismo criterio, o el modelo solo
sabe ordenar lo que ya vio?

El protocolo es un **hold-out por liga**:

1. Se parte de un modelo ajustado sin las observaciones de la liga excluida — ni
   sus filas ni su influencia en el z-score (ver «Sobre el z-score»). El barrido
   lo construye ya asi (`--holdout` decide lo que se AJUSTA, no solo lo que se
   mide) y entonces la fase lo usa tal cual: se mide la generalizacion del
   artefacto que se esta comparando, no la de un gemelo suyo. Si el modelo que
   llega SI vio la liga —el caso de `evaluar`, que carga artefactos ya hechos—,
   la fase se ajusta el suyo con el pipeline real
   (`formulacion2/5.construir_con_estado`).
2. Las entidades de la liga excluida se colocan en ese modelo con
   `src.similitud.foldin`, que es exactamente lo que hace la app cuando alguien
   pregunta por una entidad que esta en la base de datos y no en el modelo.
3. Se miden las MISMAS cosas dentro y fuera de muestra. Lo que sale de aqui como
   metrica es el valor de FUERA; la comparacion con el de dentro se publica en el
   informe para poder leer la caida.

## Que se mide

**Las mismas metricas de las fases anteriores**, una por una, sobre entidades que
el ajuste no vio (`METRICAS`). Cada una viaja despues como `gen_<metrica>` y es
una metrica mas del barrido y del score compuesto, con su **valor absoluto** fuera
de muestra. No se publica ninguna brecha: un modelo malo dentro y fuera tendria
brecha cero sin generalizar nada. La caida se lee igual, porque el informe pone
los dos ambitos uno al lado del otro.

- **`top1` y `mrr` — auto-similitud fuera de muestra**: cada entidad se desdobla
  en sus partidos pares y sus impares y se comprueba si una mitad recupera a la
  otra entre todas las mitades del mismo ambito. Es el pilar de la Fase 1
  aplicado a entidades que el modelo no vio, y **no es circular**: nadie le ha
  dicho al modelo que esas dos mitades son la misma persona. Cada formulacion
  compara las dos mitades con SU similitud —coseno de los embeddings en la F5,
  columnas de W agregadas en la F2 (`foldin.similitud_entre_nuevas`)—, que es lo
  mismo que hace la Fase 1 dentro de muestra; por eso `top1` y `gen_top1` son
  comparables dentro de un modelo. Se queda sin medir un F5 sin estado warm, de
  donde sale su ancho de kernel.
- **`pureza_top1` y `knn_accuracy` — rol** (solo jugador). La traduccion medible
  de «¿recomienda de manera optima?»: si a un extremo de la liga excluida le
  salen extremos, el criterio ha viajado. La etiqueta de rol no entra en el
  modelo, pero **si entra la posicion como feature**
  (`config.USE_POSITION_FEATURES`), asi que son parcialmente circulares — igual
  que en las Fases 0 y 5 y por el mismo motivo.
- **`rbo_medio` — estabilidad**, y solo fuera de muestra: se remuestrean con
  reemplazo los partidos de la entidad proyectada y se compara su top-k con el
  que da con todos. Dentro de muestra la gemela es la Fase 2, que remuestrea
  reconstruyendo la S entera; mezclarlas compararia dos protocolos distintos.
- **`coverage` y `diversity`**: cuantas entidades del modelo llegan a
  recomendarse y como de parecidos entre si son los candidatos de cada lista.
  Delatan el colapso —mandar a todo el mundo al mismo puñado de vecinos— antes
  que la pureza.

Los dos ambitos usan el **mismo numero de consultas** y el mismo pool
(`_pool_igualado`): el top-1 y la cobertura dependen de esos tamanos, asi que
comparar 284 contra 2.640 mediria el tamano y no la generalizacion.

`asimetria` (Fase 0) no tiene gemela: ver `SIN_GEMELA`.

## Que NO se puede concluir de la brecha

La brecha mezcla dos efectos que este diseño no separa: que la liga excluida sea
**distinta** (otro nivel, otro estilo) y que su respuesta se calcule
**proyectando** en vez de leyendo la S. El segundo esta acotado aparte en
`foldin` (sobre entidades que si estan en el modelo, la proyeccion conserva 9,1
de cada 10 del top-10 en la F5 y 7,7 en la F2), asi que una brecha muy por encima
de eso apunta a la liga y no al metodo. Separarlos del todo exigiria reajustar
excluyendo entidades sueltas de la MISMA liga, que es otro experimento.

## Sobre el z-score

- Con `global`, el ajuste calcula mu/sd **solo con las ligas de entrenamiento** y
  la liga excluida se estandariza con esas mu/sd congeladas. Es lo unico honesto:
  incluirla al calcular la media seria dejarle ver el conjunto de prueba.
- Con `por_liga`, cada competicion se estandariza contra si misma **por
  definicion**, asi que la liga excluida usa las suyas propias y no hay fuga
  posible. No se congela nada (congelarlo dejaria a la liga nueva sin mu/sd y
  todas sus entidades saldrian en el mismo punto).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.similitud import data as sdata
from src.similitud import features as sfeatures
from src.similitud import distancias, foldin, formulacion2, formulacion5
from src.similitud.modelo import ModeloSimilitud

from . import config, construccion, metricas
from .fases import top_k_indices

_FORMULACIONES = {"2": formulacion2, "5": formulacion5}


class SinHoldout(ValueError):
    """La particion pedida no deja un lado utilizable (o no existe la liga)."""


# --------------------------------------------------------------------------- #
# Que se mide, y como se llama fuera de muestra                                 #
# --------------------------------------------------------------------------- #
# La fase mide las MISMAS metricas que las fases 0/1/2/5, pero sobre entidades
# que el ajuste no vio. Cada una viaja despues con el prefijo `gen_` y es una
# metrica mas del barrido y del score compuesto (`puntuacion`), con su VALOR
# ABSOLUTO fuera de muestra —no una brecha—: un modelo malo dentro y fuera tiene
# brecha cero y no generaliza, generaliza mal. La brecha se sigue viendo en el
# informe, que publica los dos ambitos uno al lado del otro.
#
# `asimetria` (Fase 0) no tiene gemela y no la tendra: se mide sobre la S
# direccional ANTES de simetrizar, y fuera de muestra no hay S — la proyeccion da
# una fila, no una matriz. En la F5 esa fila es un coseno (simetrico por
# construccion, asimetria 0 siempre) y en la F2 es una sola direccion, cuya
# contraria exigiria reajustar la W de todas las entidades del modelo.
METRICAS: tuple[str, ...] = (
    "top1",            # auto-similitud: la mitad par recupera a la impar
    "mrr",             # idem, rango reciproco medio
    "pureza_top1",     # el top-1 comparte rol grueso
    "knn_accuracy",    # rol predicho por voto de los k vecinos
    "rbo_medio",       # estabilidad del top-k ante remuestreo de sus partidos
    "coverage",        # cuantas entidades del modelo llegan a recomendarse
    "diversity",       # distancia media intra-lista
)
SIN_GEMELA: tuple[str, ...] = ("asimetria",)
PREFIJO = "gen_"


def nombre_metrica(metrica: str) -> str:
    """Como se llama fuera de muestra la metrica `metrica`."""
    return f"{PREFIJO}{metrica}"


@dataclass(frozen=True)
class Ambito:
    """Las metricas de un lado de la particion (dentro o fuera de muestra).

    `metricas` lleva una entrada por cada nombre de `METRICAS` (NaN donde no se
    ha podido medir, que no es lo mismo que 0). El resto son cifras de contexto
    para leer el informe: cuantas entidades se han consultado, cuantas tenian rol
    conocido, cuantos top-1 distintos han salido y el pool con el que se midio la
    auto-similitud.
    """

    nombre: str                  # "dentro" | "fuera"
    n_entidades: int
    metricas: dict[str, float]
    n_rol: int
    distintos_top1: int
    score_top1: float
    pool_autosim: int

    def fila(self) -> dict:
        return {
            "ambito": self.nombre,
            "n_entidades": self.n_entidades,
            **{m: self.metricas.get(m, float("nan")) for m in METRICAS},
            "n_rol": self.n_rol,
            "distintos_top1": self.distintos_top1,
            "score_top1": self.score_top1,
            "pool_autosim": self.pool_autosim,
        }


# --------------------------------------------------------------------------- #
# Particion por liga                                                           #
# --------------------------------------------------------------------------- #

def resolver_holdout(ligas_disponibles: dict[str, str], pedidas) -> tuple[str, ...]:
    """Traduce lo que se escribio en `--holdout` a claves de liga.

    Se acepta la clave tal cual (`1238-108`) y tambien un trozo del nombre
    (`india`, `Indian Super`), porque la clave es un par de ids que nadie
    recuerda. Una peticion que no case con ninguna liga es un error y no un aviso:
    seguir adelante evaluaria en silencio un hold-out vacio, que da metricas de
    aspecto normal y no significan nada.
    """
    salida: list[str] = []
    for texto in pedidas:
        objetivo = str(texto).strip().casefold()
        if not objetivo:
            continue
        casan = [
            clave for clave, nombre in ligas_disponibles.items()
            if objetivo == clave.casefold() or objetivo in nombre.casefold()
        ]
        if not casan:
            raise SinHoldout(
                f"ninguna liga de la BD casa con {texto!r}; hay: "
                + ", ".join(f"{k} ({v})" for k, v in ligas_disponibles.items())
            )
        salida.extend(casan)
    return tuple(dict.fromkeys(salida))


def _vio_la_liga(db_path, entidad: str, modelo, holdout: tuple[str, ...]) -> bool:
    """¿Hay en el modelo alguna entidad que juegue en las ligas excluidas?

    Es la comprobacion que permite reutilizar un ajuste en vez de repetirlo: se
    contrastan sus `entity_ids` con los de las observaciones de esas ligas. Si
    alguna aparece, ese modelo SI la vio y no vale como hold-out —da igual lo que
    diga quien lo pasa—.
    """
    df = sdata.cargar(db_path, entidad)
    del_holdout = df[df[sdata.LEAGUE_KEY].astype(str).isin(set(holdout))]
    if del_holdout.empty:
        return False
    ids = {int(i) for i in del_holdout["entity_id"]}
    return bool(ids & {int(i) for i in modelo.entity_ids})


def features_congeladas(modelo):
    """mu/sd con las que se estandarizo el modelo, tal y como las guarda su meta.

    Es lo que pone las entidades nuevas en SU espacio (ver el docstring del
    modulo). Un artefacto sin ellas no sirve para proyectar, y se dice.
    """
    estadisticas = sfeatures.EstadisticasNorm.desde_dict(
        modelo.meta.get("estadisticas_normalizacion"))
    if estadisticas is None:
        raise SinHoldout(
            "el artefacto no guarda con que se estandarizo: no se puede colocar "
            "en el a las entidades de la liga excluida"
        )
    return estadisticas


def _mitades(match_id: np.ndarray, grupo: np.ndarray, n_grupos: int) -> np.ndarray:
    """Desdobla cada grupo en mitades (partidos pares / impares).

    Devuelve un vector con el indice de mitad de cada observacion: el grupo `g`
    ocupa las posiciones `2g` (pares) y `2g+1` (impares). Misma regla que
    `construccion.split_par_impar` —ordenar por `match_id` y alternar—, pero
    sobre un subconjunto de filas y sin necesitar un `Contexto`: aqui las
    entidades de fuera de muestra no tienen ninguno.
    """
    mitad = np.full(len(grupo), -1, dtype=np.int64)
    for g in range(n_grupos):
        filas = np.flatnonzero(grupo == g)
        if filas.size == 0:
            continue
        orden = filas[np.argsort(match_id[filas], kind="mergesort")]
        es_par = np.arange(len(orden)) % 2 == 0
        mitad[orden[es_par]] = 2 * g
        mitad[orden[~es_par]] = 2 * g + 1
    return mitad


def _pool_igualado(n: int, tope: int, seed: int) -> np.ndarray:
    """Submuestra fija de `tope` indices de `n` (o todos si ya son menos).

    El top-1 de un self-retrieval depende del tamano del pool (con 100 candidatos
    acertar es mas facil que con 2.600), asi que comparar dentro y fuera exige
    igualarlos. La submuestra es fija por seed para que la cifra sea reproducible.
    """
    if n <= tope:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=tope, replace=False))


# --------------------------------------------------------------------------- #
# Metricas de un ambito                                                        #
# --------------------------------------------------------------------------- #

def _voto(roles_vecinos: list[str | None]) -> str | None:
    """Rol mayoritario entre los vecinos, ignorando los desconocidos."""
    votos = [v for v in roles_vecinos if v and v != config.ROL_DESCONOCIDO]
    if not votos:
        return None
    valores, cuentas = np.unique(votos, return_counts=True)
    return str(valores[np.argmax(cuentas)])


def _acierto_por_rol(
    listas: dict[int, list[tuple[int, float]]],
    roles: dict[int, str],
) -> tuple[float, float, int]:
    """(pureza@1, acierto k-NN, n evaluadas) de un conjunto de rankings.

    `listas` es {entity_id consultado: [(entity_id recomendado, score), ...]}, ya
    ordenado. Se ignoran las entidades cuyo rol no se conoce: contarlas como
    fallo penalizaria al modelo por un hueco de la BD.
    """
    aciertos_1 = aciertos_k = evaluadas = 0
    for consultado, ranking in listas.items():
        rol = roles.get(int(consultado))
        if not rol or rol == config.ROL_DESCONOCIDO or not ranking:
            continue
        evaluadas += 1
        if roles.get(int(ranking[0][0])) == rol:
            aciertos_1 += 1
        votado = _voto([roles.get(int(j)) for j, _ in ranking[: config.KNN_K]])
        if votado == rol:
            aciertos_k += 1
    if not evaluadas:
        return float("nan"), float("nan"), 0
    return aciertos_1 / evaluadas, aciertos_k / evaluadas, evaluadas


def _colapso(listas: dict[int, list[tuple[int, float]]]) -> tuple[int, float]:
    """(cuantos top-1 distintos, similitud media del top-1)."""
    primeros = [ranking[0] for ranking in listas.values() if ranking]
    if not primeros:
        return 0, float("nan")
    return len({int(j) for j, _ in primeros}), float(np.mean([s for _, s in primeros]))


def _cobertura(listas: dict[int, list[tuple[int, float]]], n_modelo: int) -> float:
    """Fraccion de entidades del modelo que aparecen en algun top-k.

    Misma definicion que la Fase 5, sobre las listas de este ambito. Depende del
    NUMERO de consultas (con 10 consultas no se puede cubrir mas de 10·k), asi
    que los dos ambitos se comparan con el mismo numero: ver `_pool_igualado`.
    """
    if not n_modelo:
        return float("nan")
    aparecen = {int(j) for ranking in listas.values() for j, _ in ranking}
    return len(aparecen) / n_modelo


def _diversidad(listas: dict[int, list[tuple[int, float]]],
                fila_de: dict[int, int], display: np.ndarray) -> float:
    """Distancia euclidea media entre los perfiles de cada top-k (Fase 5).

    Se mide sobre los CANDIDATOS, que siempre son entidades del modelo y tienen
    su `feat_display`: es una propiedad de la lista recomendada, no de quien
    consulta, asi que vale igual para una consulta proyectada.
    """
    valores = []
    for ranking in listas.values():
        filas = [fila_de[int(j)] for j, _ in ranking if int(j) in fila_de]
        if len(filas) < 2:
            continue
        sub = display[filas]
        d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        iu = np.triu_indices(len(filas), k=1)
        valores.append(float(np.mean(d[iu])))
    return float(np.mean(valores)) if valores else float("nan")


def _estabilidad_proyectada(
    ref, mf, ids: list[int], base: dict[int, list[tuple[int, float]]], B: int,
) -> float:
    """RBO@10 medio del top-k de una entidad proyectada ante remuestreo.

    La gemela fuera de muestra de la Fase 2, con su misma logica: se remuestrean
    con reemplazo los partidos de la entidad y se compara el top-k resultante con
    el que da con todos. Aqui el remuestreo es **gratis** porque la proyeccion se
    descompone por observacion (`foldin.piezas`): cambiar las masas no obliga a
    rehacer ni el mapa RFF ni las columnas de W, igual que la Fase 2 reutiliza la
    W aprendida.

    Solo cuenta para las entidades con al menos dos partidos: con uno, todos los
    remuestreos son ese mismo partido y el RBO seria 1 por construccion — un
    numero perfecto que solo dice que no habia nada que remuestrear.
    """
    if B <= 0 or not ids:
        return float("nan")
    rng = np.random.default_rng(config.SEED)
    valores: list[float] = []
    for eid in ids:
        lista_base = [j for j, _ in base.get(eid, [])]
        if not lista_base:
            continue
        piezas = foldin.piezas(ref, mf, eid)
        n = piezas.n_observaciones
        if n < 2:
            continue
        for _ in range(B):
            mult = np.bincount(rng.integers(0, n, size=n), minlength=n)
            remuestreada = piezas.puntuar(piezas.pesos * mult)
            lista = [j for j, _ in remuestreada.top(config.K_LISTA)]
            valores.append(metricas.rbo(lista_base, lista, config.RBO_P))
    return float(np.mean(valores)) if valores else float("nan")


def _autosimilitud(puntuar, pares: list[tuple[int, int]],
                   pool: np.ndarray) -> tuple[float, float, int]:
    """Self-retrieval entre mitades: ¿recupera la mitad A a su mitad B?

    `puntuar(a)` devuelve la similitud de la mitad `a` con TODAS las mitades,
    `pares` son los indices (A, B) de cada entidad y `pool` las mitades B
    candidatas. Se devuelve (top-1, MRR, tamano del pool). Es la Fase 1 exacta
    —rango del objetivo dentro del pool—, pero sobre mitades calculadas fuera de
    muestra.

    La similitud entra como funcion y no como matriz porque cada formulacion la
    define a su manera (coseno de embeddings en la F5, agregacion de W en la F2) y
    lo que esta fase mide es lo mismo en las dos: el rango del objetivo. Es la
    misma division de trabajo que en la Fase 1, donde `reconstruir_S` y
    `reconstruir_S_f5` producen la S y el calculo del rango es comun.
    """
    if not pares or pool.size == 0:
        return float("nan"), float("nan"), int(pool.size)
    posicion = {int(v): p for p, v in enumerate(pool)}
    rangos = []
    for a, b in pares:
        if int(b) not in posicion:
            continue
        fila = puntuar(int(a))[pool]
        rangos.append(metricas.rango_del_objetivo(fila, posicion[int(b)]))
    if not rangos:
        return float("nan"), float("nan"), int(pool.size)
    arr = np.array(rangos, dtype=float)
    return metricas.recall_at_k(arr, 1), metricas.mrr(arr), int(pool.size)


# --------------------------------------------------------------------------- #
# Fase 7                                                                        #
# --------------------------------------------------------------------------- #

def fase7_generalizacion(
    db_path,
    entidad: str,
    normalizacion: str,
    formulacion: str,
    holdout: tuple[str, ...],
    roles: dict[int, str],
    reportar=None,
    reportar_ajuste=None,
    reportar_autosim=None,
    modelo=None,
    estado=None,
    distancia: str = distancias.POR_DEFECTO,
) -> list[dict]:
    """Ajusta sin las ligas de `holdout` y compara dentro contra fuera.

    Devuelve una fila por ambito (`dentro`, `fuera`) lista para el CSV. Es la
    fase mas cara de todas —incluye un ajuste completo del modelo— y por eso no
    entra en la seleccion por defecto: se pide con `--fases 7`.

    `modelo` y `estado` son un ajuste que YA excluye esas ligas (el del barrido,
    que construye con `--holdout`). Si se pasan, no se reajusta nada: es el mismo
    modelo que evaluan las demas fases, y asi la generalizacion se mide sobre el
    artefacto que de verdad se esta comparando —no sobre un gemelo suyo— y la fase
    deja de costar un ajuste entero. Se comprueba que sea cierto que no vio la
    liga; si la vio, se reajusta igualmente.

    Los tres reporteros van por separado porque son tres bucles muy distintos y en
    la F2 los caros son el primero y el tercero: `reportar_ajuste` sigue el ajuste
    (miles de columnas de W), `reportar` la proyeccion de las entidades excluidas
    (decenas) y `reportar_autosim` las columnas de las mitades en la
    auto-similitud de la F2 (miles otra vez, ver `_autosimilitud_por_ambito`). Con
    uno solo, la barra diria «proyeccion 1/3646» mientras lo que corre es el SLIM.
    """
    if not holdout:
        raise SinHoldout(
            "la fase de generalizacion necesita al menos una liga excluida "
            "(--holdout); sin ella no hay nada fuera de muestra que medir"
        )

    # 1. El modelo que NO vio la liga. Si nos lo dan hecho y se comprueba que es
    #    asi, se usa; si no, se ajusta aqui (es el caso de `evaluar`, que carga
    #    artefactos de `outputs/modelo/` construidos con todo).
    estadisticas_train = None
    # El estado warm solo hace falta en la F5 (de ahi sale su ancho de kernel);
    # la F2 se proyecta sin el, asi que un artefacto sin `.warm.npz` no obliga a
    # reajustar nada.
    if modelo is not None and (estado is not None or formulacion == "2"):
        if _vio_la_liga(db_path, entidad, modelo, holdout):
            modelo = estado = None
        else:
            estadisticas_train = features_congeladas(modelo)

    if modelo is None or (estado is None and formulacion != "2"):
        ctx_train = construccion.crear_contexto(
            db_path, entidad, normalizacion, excluir_ligas=holdout,
            distancia=distancia)
        if len(ctx_train.idx_real.ids) == 0:
            raise SinHoldout("el hold-out se ha llevado todas las entidades")
        modelo, estado = _FORMULACIONES[formulacion].construir_con_estado(
            ctx_train.mf, entidad, normalizacion, progreso=reportar_ajuste,
            distancia=distancia)
        estadisticas_train = ctx_train.mf.estadisticas

    # 2. Todas las observaciones en el espacio de ese ajuste (ver «Sobre el
    #    z-score» en el docstring del modulo). La GEOMETRIA tambien sale del
    #    ajuste y no de estos datos: reestimarla aqui (con la liga excluida
    #    dentro) pondria a las entidades de fuera en un espacio que el modelo no
    #    conoce, que es el mismo error que recalcular las mu/sd.
    espacio = foldin.espacio_del_modelo(modelo, estado=estado)
    ctx_todo = construccion.crear_contexto(
        db_path, entidad, normalizacion,
        estadisticas=(estadisticas_train if normalizacion == "global" else None),
        espacio=espacio,
    )
    mf = ctx_todo.mf
    ids_obs = np.asarray(mf.entity_id).astype(np.int64)
    del_modelo = {int(i) for i in modelo.entity_ids}
    liga = np.asarray(mf.league).astype(str)
    fuera_liga = np.isin(liga, np.asarray(holdout, dtype=str))
    ids_fuera = sorted({int(i) for i in ids_obs[fuera_liga]} - del_modelo)
    if not ids_fuera:
        raise SinHoldout(
            f"ninguna entidad exclusiva de {', '.join(holdout)}: esa liga no "
            "aporta nadie que el modelo no tenga ya por otra"
        )

    try:
        ref = foldin.preparar(
            modelo, mf, sigma=(estado.sigma if estado is not None else None),
            espacio=espacio)
    except foldin.EntidadNoProyectable as e:
        # Un modelo que no se sabe proyectar (la F5 con Sinkhorn) no tiene
        # generalizacion MEDIBLE por esta via, que no es lo mismo que generalizar
        # mal. Se traduce a `SinHoldout` —«esta fase no aplica a este modelo»—
        # para que el orquestador lo diga y siga con los demas en vez de abortar
        # la evaluacion entera por una combinacion de la rejilla.
        raise SinHoldout(f"no se puede proyectar sobre este modelo: {e}") from e

    # 3. Mismo NUMERO de consultas en los dos ambitos. No es cosmetico: la
    #    cobertura y el top-1 de la auto-similitud dependen de cuantas consultas
    #    y cuantos candidatos hay, asi que con tamaños distintos la comparacion
    #    mediria el tamaño.
    n_consultas = min(len(ids_fuera), len(modelo.entity_ids),
                      config.N_GENERALIZACION_MAX)
    ids_fuera = [ids_fuera[i] for i in
                 _pool_igualado(len(ids_fuera), n_consultas, config.SEED)]
    dentro_idx = _pool_igualado(
        len(modelo.entity_ids), n_consultas, config.SEED)

    fuera_listas: dict[int, list[tuple[int, float]]] = {}
    for n, eid in enumerate(ids_fuera, start=1):
        fuera_listas[eid] = foldin.proyectar_desde(ref, mf, eid).top(config.K_LISTA)
        if reportar is not None:
            reportar(n, len(ids_fuera))

    dentro_listas = {
        int(modelo.entity_ids[i]): [
            (int(modelo.entity_ids[j]), float(modelo.S[i, j]))
            for j in top_k_indices(modelo.S, int(i), config.K_LISTA)
        ]
        for i in dentro_idx
    }

    # 4. Auto-similitud fuera de muestra, con la similitud de cada formulacion.
    autos = _autosimilitud_por_ambito(
        modelo, ref, mf, ids_obs, ids_fuera, dentro_idx, estado,
        reportar=reportar_autosim)

    # 5. Estabilidad de las recomendaciones proyectadas ante remuestreo de los
    #    partidos de la entidad. Solo fuera de muestra: dentro, la gemela es la
    #    Fase 2 sobre el modelo entero, que remuestrea reconstruyendo la S y no
    #    es lo mismo ni se puede abaratar aqui.
    estabilidad = _estabilidad_proyectada(
        ref, mf, ids_fuera[:config.N_ESTABILIDAD_GENERALIZACION], fuera_listas,
        config.BOOTSTRAP_GENERALIZACION)
    fila_de = {int(e): i for i, e in enumerate(modelo.entity_ids)}

    # El rol es una etiqueta de JUGADOR. Los ids de equipo viven en otra
    # numeracion y algunos coinciden por casualidad con los de jugador: sin este
    # filtro, un equipo «acertaria» el rol de un futbolista con su mismo id.
    roles = roles if entidad == "jugador" else {}

    base = {
        "modelo": f"F{formulacion}_{entidad}_{normalizacion}_{distancia}",
        "formulacion": formulacion, "entidad": entidad,
        "normalizacion": normalizacion, "distancia": distancia,
        "holdout": ",".join(holdout),
        "fidelidad": ref.fidelidad,
        "n_train": len(modelo.entity_ids),
        "n_holdout": len(ids_fuera),
    }
    filas = []
    for nombre, listas in (("dentro", dentro_listas), ("fuera", fuera_listas)):
        pureza, knn, n_rol = _acierto_por_rol(listas, roles)
        distintos, score = _colapso(listas)
        top1, mrr, pool = autos[nombre]
        filas.append({**base, **Ambito(
            nombre=nombre,
            n_entidades=len(listas),
            metricas={
                "top1": top1,
                "mrr": mrr,
                "pureza_top1": pureza,
                "knn_accuracy": knn,
                # La estabilidad solo se mide fuera (ver el paso 5): dentro, su
                # gemela es la Fase 2 sobre el modelo entero.
                "rbo_medio": estabilidad if nombre == "fuera" else float("nan"),
                "coverage": _cobertura(listas, len(modelo.entity_ids)),
                "diversity": _diversidad(listas, fila_de, modelo.feat_display),
            },
            n_rol=n_rol,
            distintos_top1=distintos,
            score_top1=score,
            pool_autosim=pool,
        ).fila()})
    return filas


def _autosimilitud_por_ambito(
    modelo: ModeloSimilitud, ref: foldin.Referencia, mf, ids_obs: np.ndarray,
    ids_fuera: list[int], dentro_idx: np.ndarray, estado, reportar=None,
) -> dict[str, tuple[float, float, int]]:
    """Self-retrieval por mitades, dentro y fuera de muestra.

    Las dos mitades de una entidad de la liga excluida son dos entidades que el
    modelo no vio, asi que la pregunta —¿cuanto se parecen entre si?— la contesta
    cada formulacion con SU similitud:

    - **F5**: el coseno de sus `embeddings`, que es la etapa 1 tal cual. Las
      mitades de los dos ambitos se embeben JUNTAS y con el mismo mapa RFF.
    - **F2**: `foldin.similitud_entre_nuevas`, que resuelve las columnas de todas
      las mitades contra un diccionario formado por las observaciones del modelo
      (menos las de las entidades desdobladas) y las de las propias mitades, y
      agrega con la formula del ajuste. Entre dos entidades NUEVAS no falta
      ninguna direccion —las dos columnas se resuelven aqui—, que es lo que
      distingue este caso de proyectar contra el modelo (ver el docstring de
      `foldin`).

    En los dos casos las mitades se calculan de una vez y despues se rankean por
    separado con pools del mismo tamano: cualquier diferencia que salga es de las
    entidades, no del calculo. Y en los dos, la similitud es la de la Fase 1 de
    esa formulacion, asi que `top1` y `gen_top1` son comparables entre si dentro
    de un modelo — que es lo que la fase publica.

    Lo unico que se queda sin medir es la F5 sin estado warm: su ancho de kernel
    sale de ahi y estimarlo pondria a las mitades en otro espacio.
    """
    vacio = (float("nan"), float("nan"), 0)
    es_f5 = ref.mu is not None
    if es_f5 and (estado is None or estado.sigma is None):
        return {"dentro": vacio, "fuera": vacio}

    grupos = [int(modelo.entity_ids[i]) for i in dentro_idx] + list(ids_fuera)
    posicion = {eid: g for g, eid in enumerate(grupos)}
    pertenece = np.array([posicion.get(int(i), -1) for i in ids_obs])
    usadas = pertenece >= 0
    mitad = _mitades(
        np.asarray(mf.match_id)[usadas], pertenece[usadas], len(grupos))

    idx = np.full(len(ids_obs), -1, dtype=np.int64)
    idx[usadas] = mitad
    validas = idx >= 0
    if es_f5:
        mu = foldin.embeddings(
            _filtrar(mf, validas), idx[validas], 2 * len(grupos),
            float(estado.sigma), int(ref.rff_dim), espacio=ref.espacio,
        )

        def puntuar(a: int) -> np.ndarray:
            return mu @ mu[a]
    else:
        # La F2 no tiene embedding: hay que resolver las columnas de las mitades.
        # Es el bucle caro de esta fase (una regresion por observacion), y por eso
        # lleva su propio reportero.
        S_mitades = foldin.similitud_entre_nuevas(
            ref, mf, idx, 2 * len(grupos), progreso=reportar)

        def puntuar(a: int) -> np.ndarray:
            return S_mitades[a]

    # Una entidad con un solo partido no tiene mitad B: no es evaluable ni entra
    # en el pool (a diferencia de la Fase 1, donde su mitad A queda de
    # distractora: aqui los pools tienen que ser IGUALES entre ambitos, y para
    # eso hay que poder contarlos).
    tiene = np.zeros(2 * len(grupos), dtype=bool)
    tiene[np.unique(idx[validas])] = True
    n_dentro = len(dentro_idx)
    evaluables = {
        "dentro": [g for g in range(n_dentro) if tiene[2 * g] and tiene[2 * g + 1]],
        "fuera": [g for g in range(n_dentro, len(grupos))
                  if tiene[2 * g] and tiene[2 * g + 1]],
    }
    # Mismo tamano de pool en los dos lados: el top-1 de un self-retrieval baja
    # al crecer el pool, asi que con 368 candidatos dentro y 266 fuera la
    # comparacion mediria el pool y no la generalizacion.
    n = min(len(evaluables["dentro"]), len(evaluables["fuera"]))
    salida = {}
    for nombre, grupos_ok in evaluables.items():
        sel = _pool_igualado(len(grupos_ok), n, config.SEED)
        elegidos = [grupos_ok[i] for i in sel]
        pares = [(2 * g, 2 * g + 1) for g in elegidos]
        pool = np.array([2 * g + 1 for g in elegidos], dtype=np.int64)
        salida[nombre] = _autosimilitud(puntuar, pares, pool)
    return salida


def _filtrar(mf, mascara: np.ndarray):
    """Sub-matriz de features con las filas de `mascara` (como `foldin._recortar`)."""
    return sfeatures.MatrizFeatures(
        X=mf.X[mascara],
        feat_names=mf.feat_names,
        entity_id=np.asarray(mf.entity_id)[mascara],
        entity_name=np.asarray(mf.entity_name)[mascara],
        weight=np.asarray(mf.weight)[mascara],
        league=np.asarray(mf.league)[mascara],
        match_id=(None if mf.match_id is None
                  else np.asarray(mf.match_id)[mascara]),
        estadisticas=mf.estadisticas,
    )

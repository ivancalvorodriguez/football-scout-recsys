"""Proyeccion (*fold-in*) de una entidad que NO esta en el modelo.

Un artefacto servible es una foto: la S entidad-entidad de quienes estaban al
ajustar. Un jugador que solo aparece en la base de datos elegida no tiene fila en
esa S, asi que el buscador no puede recomendarle nada. Este modulo responde a esa
consulta **sin reajustar el modelo y sin tocarlo**: coloca a la entidad nueva en
la geometria que el modelo ya aprendio y puntua contra sus entidades.

Es una capa de CONSULTA, no de entrenamiento. No escribe artefactos, no cambia la
S servida y no altera las recomendaciones de nadie mas. La via completa para
incorporar entidades nuevas de verdad sigue siendo `src.incremental.reentrenar`.

## Que se puede proyectar y con que fidelidad (medido, no supuesto)

Lo decide si la etapa que produce la similitud esta definida FUERA DE MUESTRA.
Hay dos niveles, y `Proyeccion.fidelidad` dice siempre cual es el de cada
respuesta, porque no significan lo mismo:

- **Formulacion 5 con MMD -> `FIEL`**. Es la de los DOS modelos servidos desde
  que se promovieron los v3 (antes solo la del jugador). Su etapa 1
  es distribucional — el coseno entre *kernel mean embeddings*
  (`distributional.similitud_mmd`)—, y el embedding de una nube nueva se calcula
  igual que el de cualquier otra: media de phi(x) sobre sus observaciones, con el
  MISMO mapa RFF (misma semilla, mismo `rff_dim`) y el MISMO ancho de kernel que
  uso el ajuste. Es la definicion, no una aproximacion.
- **Formulacion 2 -> `APROXIMADA`**. Ya no la sirve ningun modelo de fabrica
  (el equipo era F2 hasta el v238), pero sigue viva para los artefactos F2 que
  queden y para `build --formulacion 2`. Su similitud
  es una agregacion de bloques de W por los DOS lados
  (`slim.agregar_W_a_entidades`) y luego simetrizada. De una entidad nueva se
  pueden resolver sus COLUMNAS (reconstruir cada observacion suya desde las del
  modelo), y eso da la direccion «cuanto aportan los demas a explicarla»; falta
  la contraria, que exigiria reajustar la W de todos. Es media magnitud, y por
  eso no se promete que sea el ranking del modelo. Medido sobre el modelo de
  equipo que entonces se servia (30 equipos sacados del indice y vueltos a
  proyectar, ver `SOLAPAMIENTO_F2_TOP10`): **7,7/10 de solapamiento en el top-10
  y 19/30 de top-1**. Se sirve —una entidad que solo esta en la BD no tiene otra
  respuesta que esta o ninguna— pero la interfaz **tiene que rotularla como
  aproximada**.

  Una version anterior de este modulo la rechazaba citando 1,6/10. Esa cifra era
  de una agregacion sin el denominador N_p*N_q de `agregar_W_a_entidades`: sin
  el, los equipos con mas partidos acumulan mas peso y el ranking lo ordena el
  volumen. Con la agregacion fiel a la del ajuste, la direccion unica conserva la
  mayor parte del top-10.
- **Formulacion 5 con Sinkhorn -> no se proyecta**. El coste de transporte optimo
  se calcula por PARES de nubes: colocar una entidad nueva exigiria recorrer las
  P nubes del modelo, que es justo el bucle caro de esa etapa. Se rechaza con
  `EntidadNoProyectable`.

## Dos entidades nuevas entre si (`similitud_entre_nuevas`)

Todo lo anterior es puntuar CONTRA EL MODELO. Hay una pregunta distinta —¿cuanto
se parecen entre si dos entidades de las que NINGUNA esta en el modelo?— y en ella
la F2 no pierde ninguna direccion. El motivo de que su proyeccion sea APROXIMADA
es que las columnas de las observaciones DEL MODELO ya estan resueltas y volver a
resolverlas seria reajustarlo; pero entre dos entidades nuevas las dos columnas
son nuestras: se resuelve la de cada observacion de las dos, y los bloques de W
que salen se agregan con `slim.agregar_W_a_entidades` —la misma funcion, el mismo
denominador N_p*N_q y la misma simetrizacion que el ajuste—. La S resultante tiene
la forma y la escala de la del modelo, y por eso el rotulo de esa via es FIEL
aunque la proyeccion contra el modelo siga siendo aproximada: son dos preguntas
distintas y solo una pierde media magnitud.

Las observaciones del modelo siguen haciendo falta, como **anclas**: entran en el
diccionario de candidatas de cada columna (es lo que hace la reconstruccion
selectiva y dispersa, igual que en el ajuste, donde cada observacion competia
contra todas las demas) pero no reciben nada, porque la S que se pide es solo la
de las entidades nuevas entre ellas.

Quien lo usa es la Fase 7 de la evaluacion, para medir la auto-similitud fuera de
muestra: desdobla cada entidad en sus partidos pares e impares y pregunta si una
mitad recupera a la otra. La F5 responde a eso con el coseno de sus embeddings
(`embeddings`), que ya esta definido entre nubes nuevas; esta funcion es lo mismo
para la F2.

## Por que NO se aplica el EASE de la etapa 2

Seria lo natural (la puntuacion servible es `S_ent @ B`) y **esta medido que no
funciona**: B se obtiene invirtiendo `S_ent^T S_ent`, cuyo numero de condicion en
el modelo de jugador servido es 7,4e12, y el modelo usa `lambda = 0`. Con eso B
reconstruye exactamente las filas que vio y extrapola de forma degenerada fuera de
muestra: 0,1/10 de solapamiento y 0/25 de top-1, colapsando consultas distintas al
mismo puñado de entidades. Regularizar mas (lambda = 0,1) sube a 3,6/10, pero eso
ya es otro modelo.

Lo que si generaliza es la etapa 1 sola: **9,1/10 de solapamiento con lo que el
modelo servido dice de esa misma entidad y 23/25 de top-1** (25 jugadores sacados
del indice y vueltos a proyectar). Esa es la puntuacion que se sirve aqui, y por
eso la interfaz debe rotular estas recomendaciones como calculadas fuera del
modelo: no son identicas a las de una entidad ajustada, son su geometria de
etapa 1.

## Como se remiden estas cifras

Sacando del indice una entidad que SI esta en el modelo, proyectandola y
comparando su top-10 proyectado con el que el modelo servido le da entre las
mismas candidatas. Es el unico contraste posible —de una entidad que solo esta en
la BD no hay ranking «verdadero» con el que comparar— y hay que rehacerlo si
cambia la agregacion o los hiperparametros servidos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, distancias, distributional, slim
from .distancias import Espacio
from .features import MatrizFeatures
from .modelo import ModeloSimilitud

# Fidelidad de una proyeccion respecto del modelo al que se proyecta.
FIEL = "fiel"                # es la misma magnitud que sirve el modelo
APROXIMADA = "aproximada"    # es media magnitud: una sola direccion (ver arriba)

# Metodos de la etapa 1 de la F5 que se saben proyectar. `sinkhorn` calcula el
# coste de transporte optimo por PARES de nubes: proyectar una entidad nueva
# exigiria recorrer las P nubes del modelo, que es el bucle caro de esa etapa.
METODOS_PROYECTABLES: tuple[str, ...] = ("mmd",)

# Cuanto se parece cada proyeccion al ranking que da el propio modelo para las
# entidades que SI estan en el (se las saca del indice, se las proyecta y se
# comparan los dos top-10; ver «Como se remiden estas cifras»). Viven aqui, y no
# en la interfaz, para que el rotulo cite la medida y no un adjetivo.
SOLAPAMIENTO_F5_TOP10 = 9.1    # jugador servido, 25 consultas
ACIERTOS_F5_TOP1 = 0.92
SOLAPAMIENTO_F2_TOP10 = 7.7    # equipo, F2 (30 consultas); ya no se sirve
ACIERTOS_F2_TOP1 = 0.63


class EntidadNoProyectable(LookupError):
    """No se puede puntuar esa entidad contra ese modelo sin reajustarlo."""


@dataclass(frozen=True)
class Proyeccion:
    """Puntuacion de una entidad nueva contra las entidades de un modelo.

    `puntuacion[j]` va alineada con `ids[j]`, que es el orden de `entity_ids` del
    artefacto: quien consuma esto puede indexar los nombres del modelo tal cual.

    `fidelidad` viaja con el resultado (`FIEL` / `APROXIMADA`) en vez de
    deducirse otra vez de la formulacion en cada consumidor: quien enseña estas
    recomendaciones tiene que poder decir de que clase son sin volver a razonar
    sobre el modelo.
    """

    ids: np.ndarray            # (P,) entity_id de cada entidad del modelo
    puntuacion: np.ndarray     # (P,) similitud con la entidad proyectada
    entidad_id: int            # a quien se ha proyectado
    n_observaciones: int       # con cuantas observaciones suyas se ha hecho
    fidelidad: str = FIEL

    def top(self, k: int) -> list[tuple[int, float]]:
        """Los `k` mejores como (entity_id, puntuacion), de mayor a menor.

        Las entidades marcadas como ausentes (-inf, ver `proyectar`) se DESCARTAN
        en vez de ir al final: no es que se parezcan poco, es que no hay con que
        compararlas, y rellenar el top-k con ellas seria inventarse una respuesta.
        Por eso puede devolver menos de `k` — igual que hace el servicio con la
        Formulacion 2 cuando la S es dispersa.
        """
        if k <= 0 or self.puntuacion.size == 0:
            return []
        finitas = np.where(np.isfinite(self.puntuacion))[0]
        if finitas.size == 0:
            return []
        orden = finitas[np.argsort(-self.puntuacion[finitas])][:k]
        return [(int(self.ids[j]), float(self.puntuacion[j])) for j in orden]


def fidelidad(modelo: ModeloSimilitud) -> str | None:
    """`FIEL`, `APROXIMADA` o None (no se puede) para ese modelo. Ver docstring."""
    if modelo.formulacion == "2":
        return APROXIMADA
    if modelo.formulacion == "5":
        metodo = modelo.meta.get("metodo_distribucional")
        return FIEL if metodo in METODOS_PROYECTABLES else None
    return None


def puede_proyectar(modelo: ModeloSimilitud) -> bool:
    """¿Sabe este modulo colocar una entidad nueva en ese modelo?"""
    return fidelidad(modelo) is not None


def _exigir_proyectable(modelo: ModeloSimilitud) -> str:
    clase = fidelidad(modelo)
    if clase is None:
        raise EntidadNoProyectable(
            f"el modelo F{modelo.formulacion} "
            f"({modelo.meta.get('metodo_distribucional') or 'sin metodo distribucional'}) "
            "no define la similitud fuera de muestra: hay que reentrenar para "
            "incluir entidades nuevas"
        )
    return clase


def espacio_del_modelo(
    modelo: ModeloSimilitud, estado=None, espacio: Espacio | None = None
) -> Espacio:
    """La geometria con la que se ajusto `modelo`, para proyectar en la misma.

    Por orden: la que se pase a mano, la del estado warm (que la guarda entera,
    blanqueo incluido) o —si el artefacto no declara ninguna o declara la
    euclidea— la euclidea, que es la unica que no tiene nada que recordar.

    Un modelo `mahalanobis` sin estado warm NO se puede proyectar: su blanqueo se
    estimo de los datos del ajuste y reestimarlo aqui pondria a la entidad nueva
    en otro espacio que las del modelo. Es el mismo caso que el sigma de la F5.
    """
    if espacio is not None:
        return espacio
    declarada = distancias.validar(modelo.meta.get("distancia"))
    if estado is not None and getattr(estado, "distancia", None) == declarada:
        return estado.espacio()
    if declarada == "mahalanobis":
        raise EntidadNoProyectable(
            "el modelo se ajusto con distancia mahalanobis y su blanqueo vive en "
            "el estado warm (<stem>.warm.npz), que no esta: sin el no se puede "
            "colocar a nadie en su espacio"
        )
    return Espacio(declarada)


def embeddings(
    mf: MatrizFeatures, ent_idx: np.ndarray, n_ent: int, sigma: float, rff_dim: int,
    espacio: Espacio | None = None,
) -> np.ndarray:
    """Kernel mean embeddings NORMALIZADOS, uno por entidad.

    Es la pieza intermedia de `distributional.similitud_mmd` (que hace exactamente
    esto y despues multiplica mu por mu.T). Se expone aparte porque proyectar
    necesita los mu de las entidades del modelo y el mu de la entidad nueva por
    separado, no la matriz P x P entera.

    `sigma`, `rff_dim` y `espacio` NO se estiman aqui: entran desde el modelo al
    que se proyecta. Estimarlos de los datos de la consulta pondria a la entidad
    nueva en un espacio distinto del de las entidades contra las que se la
    compara, que es justo el error que este modulo tiene que evitar.
    """
    esp = espacio if espacio is not None else Espacio(distancias.POR_DEFECTO)
    Z = distributional._rff(
        esp.transformar(mf.X), rff_dim, sigma, config.F5_RFF_SEED, espacio=esp)
    mu = distributional._embed_entidades(Z, ent_idx, mf.weight, n_ent)
    normas = np.linalg.norm(mu, axis=1)
    normas[normas == 0.0] = 1.0
    return mu / normas[:, None]


@dataclass(frozen=True)
class Referencia:
    """El lado del MODELO ya preparado, para proyectar muchas entidades sobre el.

    Todo lo que no depende de a quien se proyecte se calcula una vez: en la F5,
    los embeddings de las P entidades del modelo (que exigen mapear por RFF las
    ~50.000 observaciones de la BD, lo caro de la operacion); en la F2, que filas
    de `mf` son del modelo y con que masa entran.

    Existe porque hay dos consumidores que proyectan EN LOTE —la evaluacion de
    generalizacion, que proyecta una liga entera, y la app, que encadena
    consultas contra el mismo modelo— y repetir esa parte por consulta multiplica
    el coste por el numero de entidades sin cambiar el resultado.
    """

    ids: np.ndarray              # (P,) entity_id de cada entidad del modelo
    fidelidad: str
    fila_entidad: np.ndarray     # (M,) entidad del modelo de cada fila de mf (-1 si no)
    presentes: np.ndarray        # (P,) True si la entidad tiene observaciones aqui
    # Geometria del modelo: con la que se midio al ajustar y con la que hay que
    # medir aqui. Nunca se reestima (ver `espacio_del_modelo`).
    espacio: Espacio = field(default_factory=lambda: Espacio(distancias.POR_DEFECTO))
    # Formulacion 5.
    mu: np.ndarray | None = None       # (P, D) embeddings normalizados
    sigma: float | None = None
    rff_dim: int | None = None
    # Formulacion 2.
    hiper: dict | None = None          # n_neighbors/beta/l1 con los que se ajusto

    @property
    def n_entidades(self) -> int:
        return int(self.ids.shape[0])


def preparar(
    modelo: ModeloSimilitud,
    mf: MatrizFeatures,
    sigma: float | None = None,
    rff_dim: int | None = None,
    espacio: Espacio | None = None,
    estado=None,
) -> Referencia:
    """Prepara `modelo` para recibir proyecciones calculadas sobre `mf`.

    `mf` son las features de la BD elegida, construidas con las **mu/sd
    congeladas del modelo** (`meta["estadisticas_normalizacion"]`): sin eso, las
    entidades nuevas y las del modelo estarian en escalas distintas y ni el
    coseno (F5) ni la reconstruccion por minimos cuadrados (F2) significarian
    nada.

    `sigma` es el ancho de kernel del ajuste, que vive en el estado warm del
    modelo (`<stem>.warm.npz`). Lo exige la F5 y no se estima (ver `embeddings`);
    la F2 no lo usa.

    `espacio` (o, en su defecto, `estado`) es la DISTANCIA con la que se ajusto
    el modelo. Se resuelve en `espacio_del_modelo`: sin nada, la euclidea, que es
    lo que declara cualquier artefacto anterior a que la distancia fuera un
    parametro. Aplica a las DOS formulaciones, porque las dos miden: la F5 en su
    kernel y la F2 al elegir vecinas y al reconstruir la columna.
    """
    clase = _exigir_proyectable(modelo)
    esp = espacio_del_modelo(modelo, estado=estado, espacio=espacio)
    ids_modelo = np.asarray(modelo.entity_ids)
    posicion = {int(i): k for k, i in enumerate(ids_modelo)}
    ids_obs = np.asarray(mf.entity_id).astype(np.int64)

    fila_entidad = np.full(ids_obs.shape, -1, dtype=np.int64)
    for valor, k in posicion.items():
        fila_entidad[ids_obs == valor] = k
    del_modelo = fila_entidad >= 0
    presentes = np.zeros(len(ids_modelo), dtype=bool)
    presentes[np.unique(fila_entidad[del_modelo])] = True

    if modelo.formulacion == "2":
        return Referencia(
            ids=ids_modelo,
            fidelidad=clase,
            fila_entidad=fila_entidad,
            presentes=presentes,
            espacio=esp,
            hiper={
                "n_neighbors": int(
                    modelo.meta.get("n_neighbors") or config.F2_N_NEIGHBORS),
                "beta": float(modelo.meta.get("beta", config.F2_BETA)),
                "l1": float(modelo.meta.get("l1", config.F2_L1)),
            },
        )

    if sigma is None:
        raise EntidadNoProyectable(
            "la F5 necesita el ancho de kernel del ajuste (sigma) y no se "
            "estima: sale del estado warm del modelo"
        )
    if rff_dim is None:
        rff_dim = int(modelo.meta.get("rff_dim") or config.F5_RFF_DIM)
    mu = embeddings(
        _recortar(mf, del_modelo), fila_entidad[del_modelo],
        len(ids_modelo), float(sigma), int(rff_dim), espacio=esp,
    )
    return Referencia(
        ids=ids_modelo, fidelidad=clase, fila_entidad=fila_entidad,
        presentes=presentes, espacio=esp, mu=mu, sigma=float(sigma),
        rff_dim=int(rff_dim),
    )


def proyectar_desde(
    ref: Referencia, mf: MatrizFeatures, entidad_id: int
) -> Proyeccion:
    """Puntua a `entidad_id` contra un modelo ya preparado (ver `preparar`).

    `mf` tiene que ser LA MISMA matriz con la que se preparo la referencia: los
    indices de fila de `ref.fila_entidad` son los de esa matriz, y las
    observaciones de la entidad nueva se localizan dentro de ella.
    """
    entidad_id = int(entidad_id)
    ids_obs = np.asarray(mf.entity_id).astype(np.int64)
    if entidad_id in {int(i) for i in ref.ids}:
        raise EntidadNoProyectable(
            f"la entidad {entidad_id} ya esta en el modelo: se sirve su fila de S"
        )
    suya = ids_obs == entidad_id
    n_suyas = int(suya.sum())
    if n_suyas == 0:
        raise EntidadNoProyectable(
            f"la entidad {entidad_id} no tiene observaciones en estos datos"
        )

    if ref.mu is None:      # Formulacion 2
        puntuacion = _puntuar_f2(ref, mf, suya)
    else:                   # Formulacion 5
        propio = embeddings(
            _recortar(mf, suya), np.zeros(n_suyas, dtype=np.int64), 1,
            float(ref.sigma), int(ref.rff_dim), espacio=ref.espacio,
        )[0]
        puntuacion = ref.mu @ propio

    # Una entidad del modelo sin observaciones en ESTOS datos no ha podido
    # participar (su embedding es nulo en la F5, y en la F2 ninguna columna la
    # elige como vecina), asi que puntuaria 0 por casualidad: se marca como
    # ausente para que no se cuele en el top-k por delante de una similitud baja
    # pero real.
    puntuacion = np.where(ref.presentes, puntuacion, -np.inf)
    return Proyeccion(
        ids=ref.ids,
        puntuacion=puntuacion,
        entidad_id=entidad_id,
        n_observaciones=n_suyas,
        fidelidad=ref.fidelidad,
    )


@dataclass(frozen=True)
class Piezas:
    """La proyeccion de una entidad DESCOMPUESTA por observacion suya.

    `aporte[i]` es lo que la observacion i-esima de la entidad aporta a la
    puntuacion contra cada entidad del modelo, antes de promediar. Volver a
    ponderar esas filas da otra proyeccion **sin recalcular nada de lo caro**: ni
    el mapa RFF (F5) ni las columnas de W (F2), que no dependen de los pesos.

    Existe para el remuestreo: medir la estabilidad de una recomendacion
    proyectada exige rehacerla B veces cambiando la composicion de partidos de la
    entidad, y hacerlo por la via normal multiplicaria por B el coste del ajuste
    de cada columna. Es el mismo criterio que la Fase 2 de la evaluacion, que
    perturba las masas y reutiliza la estructura aprendida.
    """

    ids: np.ndarray            # (P,) entity_id de cada entidad del modelo
    aporte: np.ndarray         # (n_obs, P) o (n_obs, D) segun la formulacion
    pesos: np.ndarray          # (n_obs,) masa de cada observacion suya
    fidelidad: str
    entidad_id: int
    presentes: np.ndarray      # (P,) entidades del modelo con observaciones aqui
    mu_modelo: np.ndarray | None = None    # (P, D), solo F5
    masa_modelo: np.ndarray | None = None  # (P,), solo F2 (denominador)

    @property
    def n_observaciones(self) -> int:
        return int(self.aporte.shape[0])

    def puntuar(self, pesos: np.ndarray | None = None) -> Proyeccion:
        """Rehace la proyeccion con otras masas por observacion.

        `pesos` None reproduce exactamente la proyeccion original. Un vector de
        multiplicidades (el bootstrap) se pasa ya multiplicado por las masas.
        """
        w = self.pesos if pesos is None else np.asarray(pesos, dtype=float)
        masa = float(w.sum())
        if self.mu_modelo is not None:              # F5: media de phi(x) y coseno
            if masa <= 0.0:
                propio = np.zeros(self.aporte.shape[1])
            else:
                propio = (self.aporte * w[:, None]).sum(axis=0) / masa
            norma = float(np.linalg.norm(propio)) or 1.0
            puntuacion = self.mu_modelo @ (propio / norma)
        else:                                        # F2: agregacion de columnas
            num = (self.aporte * w[:, None]).sum(axis=0)
            den = self.masa_modelo * (masa if masa > 0.0 else 1.0)
            den = np.where(den == 0.0, 1.0, den)
            puntuacion = num / den
        return Proyeccion(
            ids=self.ids,
            puntuacion=np.where(self.presentes, puntuacion, -np.inf),
            entidad_id=self.entidad_id,
            n_observaciones=self.n_observaciones,
            fidelidad=self.fidelidad,
        )


def piezas(ref: Referencia, mf: MatrizFeatures, entidad_id: int) -> Piezas:
    """Descompone la proyeccion de `entidad_id` por observacion (ver `Piezas`).

    Mismo contrato que `proyectar_desde` —de hecho `piezas(...).puntuar()` da su
    mismo resultado, y hay una prueba que lo fija—; lo que cambia es que aqui lo
    caro queda hecho y se puede repesar tantas veces como haga falta.
    """
    entidad_id = int(entidad_id)
    ids_obs = np.asarray(mf.entity_id).astype(np.int64)
    if entidad_id in {int(i) for i in ref.ids}:
        raise EntidadNoProyectable(
            f"la entidad {entidad_id} ya esta en el modelo: se sirve su fila de S"
        )
    suya = ids_obs == entidad_id
    if not suya.any():
        raise EntidadNoProyectable(
            f"la entidad {entidad_id} no tiene observaciones en estos datos"
        )
    pesos = np.asarray(mf.weight, dtype=float)[suya]

    if ref.mu is not None:      # Formulacion 5
        Z = distributional._rff(
            ref.espacio.transformar(np.asarray(mf.X, dtype=float)[suya]),
            int(ref.rff_dim), float(ref.sigma), config.F5_RFF_SEED,
            espacio=ref.espacio)
        return Piezas(ids=ref.ids, aporte=Z, pesos=pesos, fidelidad=ref.fidelidad,
                      entidad_id=entidad_id, presentes=ref.presentes,
                      mu_modelo=ref.mu)

    # Formulacion 2: lo que aporta cada observacion nueva a cada entidad, sin el
    # denominador (que depende de los pesos y se aplica en `puntuar`).
    aporte, masa = _aportes_f2(ref, mf, suya)
    return Piezas(ids=ref.ids, aporte=aporte, pesos=pesos, fidelidad=ref.fidelidad,
                  entidad_id=entidad_id, presentes=ref.presentes, masa_modelo=masa)


def proyectar(
    modelo: ModeloSimilitud,
    mf: MatrizFeatures,
    entidad_id: int,
    sigma: float | None = None,
    rff_dim: int | None = None,
    espacio: Espacio | None = None,
    estado=None,
) -> Proyeccion:
    """Proyecta UNA entidad: `preparar` + `proyectar_desde` de una vez.

    Es la forma comoda para una consulta suelta. Para varias sobre el mismo
    modelo hay que preparar una vez y reutilizar la referencia: ahi esta todo el
    coste (ver `Referencia`).
    """
    ref = preparar(modelo, mf, sigma=sigma, rff_dim=rff_dim, espacio=espacio,
                   estado=estado)
    return proyectar_desde(ref, mf, entidad_id)


def _vecinas_en(
    base: np.ndarray, consulta: np.ndarray, k: int,
    propia: np.ndarray | None = None, paso: int = 512,
    espacio: Espacio | None = None,
) -> np.ndarray:
    """Indices de las `k` filas de `base` mas cercanas a cada fila de `consulta`.

    El equivalente de `slim.vecinos_mas_cercanos` cuando las candidatas y las
    consultas son matrices DISTINTAS; las dos comparten implementacion en
    `distancias.vecinos`, que es quien sabe medir en cada geometria. Las dos
    matrices vienen YA transformadas.

    `propia[i]` es la fila de `base` que ES la consulta `i` (o -1 si no esta
    ahi), y se excluye. Proyectar contra el modelo no la necesita —la entidad
    nueva no es candidata de si misma—, pero `similitud_entre_nuevas` si: alli
    las consultas estan DENTRO del diccionario, y sin excluirla cada observacion
    se reconstruiria consigo misma y la W seria la identidad. Es el mismo `w_ss =
    0` que impone `slim.slim_instancia`.

    Se recorre la consulta por bloques: la matriz de distancias completa es
    (n_consulta x n_base), y con las ~50.000 observaciones de la BD por las
    ~16.000 mitades de la Fase 7 no cabe en memoria.
    """
    esp = espacio if espacio is not None else Espacio(distancias.POR_DEFECTO)
    return distancias.vecinos(esp, base, consulta, k, propia=propia, paso=paso)


def similitud_entre_nuevas(
    ref: Referencia,
    mf: MatrizFeatures,
    grupo: np.ndarray,
    n_grupos: int,
    progreso=None,
) -> np.ndarray:
    """S (n_grupos x n_grupos) entre entidades que NO estan en el modelo.

    `grupo[i]` es el grupo al que pertenece la fila `i` de `mf`, o -1 si no es de
    ninguno. Un «grupo» es cualquier entidad fuera de muestra: en la Fase 7 son
    las mitades par/impar de las entidades evaluadas, pero la funcion no sabe eso.

    Se resuelve la columna de CADA observacion de esos grupos —el mismo problema
    que `slim.slim_instancia` resuelve para las del ajuste, con los
    hiperparametros del artefacto (ver `_puntuar_f2`)— contra un diccionario de
    candidatas formado por:

    - las observaciones del modelo que no pertenezcan a ningun grupo (**anclas**:
      compiten por explicar cada columna, pero no reciben similitud), y
    - las observaciones de los propios grupos, salvo la que se esta resolviendo.

    Que las anclas excluyan a los grupos no es un detalle: cuando el grupo es la
    mitad de una entidad que SI esta en el modelo (el ambito «dentro» de la Fase
    7), sus filas estarian dos veces —como ancla y como grupo— y la agregacion las
    contaria dos veces. Con la regla de arriba los dos ambitos se calculan igual:
    la entidad evaluada sale del diccionario y entra desdoblada.

    Devuelve la agregacion de esos bloques con `slim.agregar_W_a_entidades`: mismo
    denominador de masas, misma simetrizacion y diagonal a cero. La escala es la
    de la S del modelo.

    Solo la F2: en la F5 la similitud entre dos nubes nuevas es el coseno de sus
    `embeddings`, que ya esta definido y no cuesta ningun ajuste.
    """
    if ref.mu is not None:
        raise EntidadNoProyectable(
            "similitud_entre_nuevas es de la F2; en la F5 dos entidades nuevas se "
            "comparan por el coseno de sus `embeddings`"
        )
    hiper = ref.hiper or {}
    # Misma geometria que el ajuste: se transforma una vez y todo lo de abajo
    # —vecinas y minimos cuadrados— vive ya en ese espacio.
    X = ref.espacio.transformar(np.asarray(mf.X, dtype=float))
    resolver = (slim.elasticnet_no_negativo if ref.espacio.euclidea_transformada
                else slim.elasticnet_residuo_l1)
    peso = np.asarray(mf.weight, dtype=float)
    grupo = np.asarray(grupo, dtype=np.int64)
    filas = np.flatnonzero(grupo >= 0)
    if filas.size == 0:
        return np.zeros((n_grupos, n_grupos), dtype=float)

    anclas = np.flatnonzero((ref.fila_entidad >= 0) & (grupo < 0))
    base = np.concatenate([anclas, filas])
    # Posicion de cada fila dentro de `base` (para excluirse) y dentro de `filas`
    # (que es el indice con el que se agrega).
    en_filas = np.full(grupo.shape, -1, dtype=np.int64)
    en_filas[filas] = np.arange(filas.size)
    propia = anclas.size + np.arange(filas.size)

    vecinas = _vecinas_en(
        X[base], X[filas],
        int(hiper.get("n_neighbors", config.F2_N_NEIGHBORS)), propia=propia,
        espacio=ref.espacio)

    cols_idx: list[np.ndarray] = []
    cols_val: list[np.ndarray] = []
    for fila, s in enumerate(filas):
        cand = base[vecinas[fila]]
        w = resolver(
            X[cand].T, X[s],
            beta=float(hiper.get("beta", config.F2_BETA)),
            l1=float(hiper.get("l1", config.F2_L1)),
            max_iter=config.F2_MAX_ITER, tol=config.F2_TOL,
        )
        # Solo las candidatas de algun grupo aportan a esta S; las anclas han
        # competido por explicar la columna, que es para lo que estan.
        nz = (w > 0.0) & (grupo[cand] >= 0)
        cols_idx.append(en_filas[cand[nz]])
        cols_val.append(w[nz])
        if progreso is not None:
            progreso(fila + 1, filas.size)

    return slim.agregar_W_a_entidades(
        cols_idx, cols_val, grupo[filas], peso[filas], n_grupos)


def _aportes_f2(
    ref: Referencia, mf: MatrizFeatures, suya: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(aporte por observacion nueva a cada entidad, masa de cada entidad).

    Es `_puntuar_f2` partido en dos justo por donde entran los pesos de la
    entidad nueva: lo de arriba (resolver las columnas) no depende de ellos y lo
    de abajo (promediar) si. Ver `Piezas`.
    """
    hiper = ref.hiper or {}
    X = ref.espacio.transformar(np.asarray(mf.X, dtype=float))
    resolver = (slim.elasticnet_no_negativo if ref.espacio.euclidea_transformada
                else slim.elasticnet_residuo_l1)
    peso = np.asarray(mf.weight, dtype=float)
    idx = ref.fila_entidad
    del_modelo = np.flatnonzero((idx >= 0) & ~suya)
    nuevas = np.flatnonzero(suya)
    aporte = np.zeros((nuevas.size, ref.n_entidades), dtype=float)
    masa = np.zeros(ref.n_entidades, dtype=float)
    if del_modelo.size == 0:
        return aporte, masa
    np.add.at(masa, idx[del_modelo], peso[del_modelo])

    vecinas = _vecinas_en(
        X[del_modelo], X[nuevas], int(hiper.get("n_neighbors", config.F2_N_NEIGHBORS)),
        espacio=ref.espacio)
    for fila, s in enumerate(nuevas):
        cand = del_modelo[vecinas[fila]]
        w = resolver(
            X[cand].T, X[s],
            beta=float(hiper.get("beta", config.F2_BETA)),
            l1=float(hiper.get("l1", config.F2_L1)),
            max_iter=config.F2_MAX_ITER, tol=config.F2_TOL,
        )
        nz = w > 0.0
        if nz.any():
            np.add.at(aporte[fila], idx[cand[nz]], peso[cand[nz]] * w[nz])
    return aporte, masa


def _puntuar_f2(
    ref: Referencia, mf: MatrizFeatures, suya: np.ndarray
) -> np.ndarray:
    """Proyeccion APROXIMADA de la F2: solo las columnas de la entidad nueva.

    Se resuelve, para cada observacion suya `s`, el mismo problema que
    `slim.slim_instancia` resuelve para las del ajuste —reconstruirla como
    combinacion dispersa no negativa de sus vecinas— pero eligiendo vecinas SOLO
    entre las observaciones del modelo, que son las que ya tienen sitio en la S.
    Los pesos se agregan a entidad con la formula de `slim.agregar_W_a_entidades`
    (masas por minutos y division por N_p*N_q), de modo que la escala es la misma
    que la de la S servida.

    Lo que NO se hace es la otra direccion (W de las observaciones del modelo
    reconstruidas usando las nuevas): eso cambiaria la W de todos y ya seria un
    reajuste. Por eso el resultado es media magnitud y va marcado `APROXIMADA`.

    Los hiperparametros salen del `meta` del artefacto y no de `config` (los
    guarda `Referencia`): el modelo servido puede haberse ajustado con otros
    valores que los de ahora, y resolver la columna con una regularizacion
    distinta la sacaria del ajuste al que se la quiere pegar.
    """
    peso = np.asarray(mf.weight, dtype=float)[suya]
    aporte, masa = _aportes_f2(ref, mf, suya)
    num = (aporte * peso[:, None]).sum(axis=0)
    # Mismo denominador que la agregacion del ajuste: masa de la entidad receptora
    # por masa de la entidad nueva. Sin el, un equipo con mas partidos puntuaria
    # mas alto solo por tener mas observaciones que aportan peso.
    denominador = masa * peso.sum()
    denominador[denominador == 0.0] = 1.0
    return num / denominador


def _recortar(mf: MatrizFeatures, mascara: np.ndarray) -> MatrizFeatures:
    """Sub-matriz de features con las filas de `mascara`.

    `embeddings` solo mira `X` y `weight`, pero se recorta el `MatrizFeatures`
    entero para no depender de ese detalle desde aqui.
    """
    return MatrizFeatures(
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

"""Dominio de la app: buscar entidades, armar el top-k explicado y las fichas.

Deliberadamente SIN Flask y SIN base de datos: aquí solo hay funciones sobre un
`ModeloSimilitud` y su `MatrizFases`, de modo que la lógica se puede probar sin
cliente HTTP y las vistas quedan reducidas a leer parámetros y renderizar. Lo que
sí necesita la BD (equipo del jugador, minutos por posición, valores reales de
las métricas) vive en `contexto` y `crudos`, y entra aquí ya calculado.

El ranking sale tal cual de `src.similitud.modelo.top_k` (misma salida que el CLI
`probar`): la app no reordena ni filtra recomendaciones, solo las viste con el
contexto necesario para leerlas (liga, fases y features en las que coinciden).

Hay un segundo origen posible para ese ranking: una entidad que está en la base
de datos elegida pero no en el modelo, colocada en él con `src.similitud.foldin`
(ver `proyeccion`). Todo lo que rodea a la recomendación —radar, coincidencias,
puntos fuertes— se calcula aquí igual en los dos casos, sobre el vector agregado
de la entidad: el del artefacto (`feat_display`) si está dentro, y el que se
deriva de la BD si se ha proyectado. Lo único que cambia es de dónde salen las
puntuaciones, y eso lo dice `Recomendacion.proyectada`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.similitud.consulta import normalizar
from src.similitud.modelo import ModeloSimilitud, top_k

from . import config, fases, glosario
from .fases import MatrizFases, ValorFase


class EntidadDesconocida(LookupError):
    """El identificador pedido no existe en el modelo."""


class EntidadSinDatos(EntidadDesconocida):
    """Está en el modelo pero no en la base de datos con la que se consulta.

    Es un `EntidadDesconocida` a propósito: para quien consulta, una entidad de
    la que la BD elegida no sabe nada **no existe**, y las vistas que ya tratan
    el «no encontrado» no tienen que aprender un caso nuevo. Lo que cambia es el
    mensaje, porque el motivo sí es distinto y tiene arreglo (elegir otra base de
    datos), y por eso es una subclase y no el mismo error.
    """


# --- Presentación de features -------------------------------------------------


def etiqueta_feature(nombre: str) -> str:
    """Nombre de feature -> etiqueta legible para la interfaz.

    Las 25 columnas `pos_*` son la fracción de minutos por posición; mostrarlas
    con su nombre crudo ("pos_left_center_back") en una tabla de scouting no se
    entiende, así que se etiquetan aparte (ver `glosario`).
    """
    return glosario.etiqueta(nombre)


def clase_z(valor: float, invertida: bool = False) -> str:
    """Clase CSS con la que se pinta un z-score: verde = bueno, rojo = malo.

    El color va por CALIDAD, no por signo. En las métricas de
    `fases.FEATURES_INVERTIDAS` subir es empeorar, así que ahí el z positivo se
    pinta del color del negativo y viceversa: sin esto, un +1,5 en «veces que le
    regatean» se leería como una virtud. Es la misma orientación que ya aplican
    el radar (`fases`) y los puntos fuertes/débiles (`_orientadas`), llevada al
    número suelto.
    """
    bueno = valor <= 0 if invertida else valor >= 0
    return "z-alto" if bueno else "z-bajo"


@dataclass(frozen=True)
class Coincidencia:
    """Una feature en la que referencia y candidato se parecen.

    Lleva las dos lecturas de la misma métrica: el z-score, que es con lo que se
    ELIGE (ver `coincidencias`), y el valor real en sus unidades, que es lo que
    se MUESTRA. El valor real es `None` cuando no hay BD o la métrica no está
    definida para esa entidad.
    """

    feature: str
    referencia: float
    candidato: float
    referencia_cruda: float | None = None
    candidato_cruda: float | None = None
    # True si en esta métrica subir es empeorar (`fases.FEATURES_INVERTIDAS`).
    invertida: bool = False
    # Fase de juego a la que pertenece la métrica, cuando se conoce. Es lo que
    # convierte cuatro coincidencias sueltas en una lectura: «coinciden en tres
    # métricas de Progresión» dice algo que las métricas por separado no dicen,
    # y es además el puente con el radar, que agrupa por esas mismas fases.
    fase: str | None = None
    # Tipo de entidad: decide si un conteo es «por 90'» o «por partido».
    entidad: str = ""

    @property
    def etiqueta(self) -> str:
        return etiqueta_feature(self.feature)

    @property
    def unidad(self) -> str:
        return glosario.unidad(self.feature, self.entidad)

    @property
    def texto_referencia(self) -> str:
        return glosario.formatear(self.feature, self.referencia_cruda)

    @property
    def texto_candidato(self) -> str:
        return glosario.formatear(self.feature, self.candidato_cruda)

    @property
    def clase_referencia(self) -> str:
        return clase_z(self.referencia, self.invertida)

    @property
    def clase_candidato(self) -> str:
        return clase_z(self.candidato, self.invertida)


@dataclass(frozen=True)
class Rasgo:
    """Feature en la que una entidad se separa de la media.

    Mismo reparto que en `Coincidencia`: `valor` es el z-score (el criterio de
    selección) y `crudo` el valor real en sus unidades (lo que se enseña).
    """

    feature: str
    valor: float
    crudo: float | None = None
    # True si en esta métrica subir es empeorar (`fases.FEATURES_INVERTIDAS`):
    # la interfaz lo señala para que un valor alto no se lea como algo bueno.
    invertida: bool = False
    # Fase a la que pertenece la métrica, cuando se conoce.
    fase: str | None = None
    # Tipo de entidad: decide si un conteo es «por 90'» o «por partido».
    entidad: str = ""

    @property
    def etiqueta(self) -> str:
        return etiqueta_feature(self.feature)

    @property
    def unidad(self) -> str:
        return glosario.unidad(self.feature, self.entidad)

    @property
    def texto(self) -> str:
        return glosario.formatear(self.feature, self.crudo)

    @property
    def clase_z(self) -> str:
        return clase_z(self.valor, self.invertida)


# Índice de una entidad que NO tiene fila en S (está en la BD y no en el modelo).
# Se usa un centinela en vez de `None` porque el índice se compara y se ordena en
# varios sitios, y un None obligaría a comprobarlo en todos ellos.
FUERA_DEL_MODELO = -1


@dataclass(frozen=True)
class Entidad:
    """Entidad lista para mostrar.

    `indice` es la fila/columna en S (la identidad interna) e `id` el
    `entity_id` de la BD: se expone ese en las URL porque es estable entre
    reconstrucciones del modelo, mientras que el índice depende del universo.

    `indice` es `FUERA_DEL_MODELO` cuando la entidad solo está en la base de
    datos: entonces no hay fila de S que leer y su recomendación se calcula
    proyectándola (ver `recomendar_proyectada`).

    `en_datos` es la otra mitad, y son cosas independientes: False significa que
    el modelo sí la tiene pero la base de datos con la que se consulta no. Sale
    igual —el artefacto trae su nombre, su liga y su vector, y puntuarla es
    exactamente lo que el modelo sabe hacer— pero sin equipo, sin posiciones y
    sin valores reales, y **sin ficha**: no se le puede pedir nada a ella (ver
    `rutas._sin_datos`). La interfaz lo rotula y no la enlaza.
    """

    indice: int
    id: int
    nombre: str
    ligas: tuple[str, ...]
    en_datos: bool = True

    @property
    def en_modelo(self) -> bool:
        return self.indice != FUERA_DEL_MODELO


@dataclass(frozen=True)
class Candidato:
    """Entidad recomendada: entidad + puntuación + por qué se parece."""

    entidad: Entidad
    rango: int
    score: float
    coincidencias: tuple[Coincidencia, ...]
    perfil: tuple[ValorFase, ...] = ()


@dataclass(frozen=True)
class Recomendacion:
    """Resultado completo de una consulta: referencia + su top-k.

    `proyectada` es la fidelidad de la proyección cuando la referencia no estaba
    en el modelo (`foldin.FIEL` / `foldin.APROXIMADA`), y None cuando la
    respuesta sale de la matriz S como siempre. La interfaz TIENE que
    distinguirlas: no son la misma clase de respuesta.
    """

    referencia: Entidad
    rasgos: tuple[Rasgo, ...]
    candidatos: tuple[Candidato, ...]
    k_pedido: int
    perfil: tuple[ValorFase, ...] = ()
    proyectada: str | None = None

    @property
    def incompleto(self) -> bool:
        """True si el modelo devolvió menos candidatos de los pedidos.

        Pasa con la Formulación 2 (S muy dispersa): una entidad poco conectada no
        tiene k vecinas con relación aprendida. Se avisa en la interfaz en vez de
        rellenar el top-k con entidades arbitrarias.
        """
        return len(self.candidatos) < self.k_pedido


@dataclass(frozen=True)
class Ficha:
    """Detalle de UNA entidad: su perfil por fases y dónde destaca o flaquea.

    `proyectada` tiene el mismo significado que en `Recomendacion`: la entidad no
    está en el modelo y su vector se ha derivado de la base de datos.
    """

    entidad: Entidad
    perfil: tuple[ValorFase, ...]
    destacados: tuple[Rasgo, ...]
    flojos: tuple[Rasgo, ...]
    proyectada: str | None = None


# --- Acceso a las entidades del modelo ----------------------------------------

def _ligas(modelo: ModeloSimilitud, indice: int) -> tuple[str, ...]:
    """Ligas (competition-season) de la entidad, según `meta`.

    Se indexa por `entity_id` como str, tal y como lo serializa
    `modelo.ligas_por_entidad` (por id y no por nombre: dos jugadores pueden
    llamarse igual).
    """
    por_entidad = modelo.meta.get("ligas_por_entidad") or {}
    return tuple(por_entidad.get(str(modelo.entity_ids[indice]), ()))


def entidad(modelo: ModeloSimilitud, indice: int,
            sin_datos: frozenset[int] = frozenset()) -> Entidad:
    """Entidad de la fila `indice` de S.

    `sin_datos` son los índices que la base de datos de la consulta no tiene: no
    se ocultan, se marcan (`en_datos=False`). Ver `Entidad`.
    """
    if not 0 <= indice < len(modelo.entity_names):
        raise EntidadDesconocida(f"Índice de entidad fuera de rango: {indice}.")
    return Entidad(
        indice=indice,
        id=int(modelo.entity_ids[indice]),
        nombre=modelo.entity_names[indice],
        ligas=_ligas(modelo, indice),
        en_datos=indice not in sin_datos,
    )


def indice_por_id(modelo: ModeloSimilitud, entity_id: int,
                  sin_datos: frozenset[int] = frozenset()) -> int:
    """Fila de S de un `entity_id` de la BD.

    Con `sin_datos` (índices que la base de datos de la consulta no tiene) el
    resultado no vale aunque el modelo lo conozca: se levanta `EntidadSinDatos`.
    """
    coincide = np.flatnonzero(modelo.entity_ids == entity_id)
    if coincide.size == 0:
        raise EntidadDesconocida(f"No hay ninguna entidad con id {entity_id}.")
    indice = int(coincide[0])
    if indice in sin_datos:
        raise EntidadSinDatos(sin_datos_texto(modelo, indice))
    return indice


def sin_datos_texto(modelo: ModeloSimilitud, indice: int) -> str:
    """Por qué no se puede responder por una entidad que la BD no tiene.

    Se redacta aquí y no en la plantilla porque lo comparten la página, la API y
    las dos vistas (top-k y ficha), y porque el mensaje tiene que decir las dos
    mitades: que el modelo sí la conoce (si no, parecería un error de búsqueda) y
    que lo que falta son sus datos.
    """
    etiqueta = config.ETIQUETA_ENTIDAD.get(modelo.entidad, modelo.entidad).lower()
    nombre = modelo.entity_names[indice]
    return (
        f"«{nombre}» está en el modelo, pero el conjunto de datos con el que se "
        f"está consultando no tiene a este {etiqueta}. Sin sus estadísticas no "
        "se pueden calcular recomendaciones: elige el conjunto de datos con el "
        "que se entrenó el modelo, o uno que lo incluya."
    )


# --- Búsqueda por nombre ------------------------------------------------------

def buscar(
    modelo: ModeloSimilitud, texto: str, limite: int = config.MAX_SUGERENCIAS,
    sin_datos: frozenset[int] = frozenset(),
) -> list[Entidad]:
    """Entidades cuyo nombre contiene `texto` (sin acentos ni mayúsculas).

    Ordena por calidad de la coincidencia (exacta, luego por el principio del
    nombre, luego en cualquier posición) y alfabéticamente dentro de cada grupo.
    A diferencia de `consulta.resolver`, que falla ante la ambigüedad porque debe
    devolver UNA entidad, aquí la ambigüedad es el caso normal: el usuario está
    escribiendo y elegirá después.

    `sin_datos` son índices de S que la base de datos de la consulta no tiene y
    que por tanto no se ofrecen: sin sus observaciones no hay nada que enseñar de
    ellos ni forma de justificar una recomendación (ver `rutas._sin_datos`).
    """
    objetivo = normalizar(texto)
    if not objetivo:
        return []
    pares = (
        (i, n) for i, n in enumerate(modelo.entity_names) if i not in sin_datos
    )
    encontrados = _coincidencias_nombre(pares, objetivo)
    if limite > 0:
        encontrados = encontrados[:limite]
    return [entidad(modelo, i) for _, _, i in encontrados]


def _coincidencias_nombre(pares, objetivo: str) -> list[tuple[int, str, int]]:
    """(calidad, nombre normalizado, clave) de los nombres que contienen `objetivo`.

    Calidad 0 = exacta, 1 = por el principio, 2 = en cualquier posición. El orden
    resultante es el de la lista de sugerencias, y se comparte entre las entidades
    del modelo y las que solo están en la BD para que las dos se ordenen igual.
    """
    encontrados: list[tuple[int, str, int]] = []
    for clave, nombre in pares:
        norm = normalizar(str(nombre))
        if norm == objetivo:
            calidad = 0
        elif norm.startswith(objetivo):
            calidad = 1
        elif objetivo in norm:
            calidad = 2
        else:
            continue
        encontrados.append((calidad, norm, clave))
    encontrados.sort()
    return encontrados


def buscar_ausentes(
    modelo: ModeloSimilitud,
    nombres_bd: dict[int, str] | None,
    texto: str,
    limite: int = config.MAX_SUGERENCIAS,
) -> list[Entidad]:
    """Entidades de la BD que el modelo NO tiene y cuyo nombre contiene `texto`.

    Son las candidatas a proyectarse. Se devuelven con `indice`
    `FUERA_DEL_MODELO` y sin ligas: la liga sale del artefacto
    (`meta["ligas_por_entidad"]`) y estas no están en él, así que se rellenan al
    proyectar, que es cuando se leen sus observaciones.

    Sin BD (`nombres_bd is None`) no hay nada que añadir: la búsqueda se queda
    con las entidades del modelo, como antes.
    """
    objetivo = normalizar(texto)
    if not objetivo or not nombres_bd:
        return []
    del_modelo = {int(i) for i in modelo.entity_ids}
    fuera = ((i, n) for i, n in nombres_bd.items() if int(i) not in del_modelo)
    encontrados = _coincidencias_nombre(fuera, objetivo)
    if limite > 0:
        encontrados = encontrados[:limite]
    return [
        Entidad(indice=FUERA_DEL_MODELO, id=int(i), nombre=str(nombres_bd[i]), ligas=())
        for _, _, i in encontrados
    ]


# --- Explicación de las recomendaciones ---------------------------------------

def _columnas_metricas(modelo: ModeloSimilitud) -> list[int]:
    """Columnas de `feat_display` que son métricas de juego.

    Excluye el bloque `pos_*`. La posición describe QUÉ es la entidad, no cómo
    juega: dos extremos no «coinciden en» ser extremos —lo que se busca es que
    jueguen parecido—, y un jugador tampoco «destaca» por su propia posición.
    Colar esas 25 columnas aquí llenaría las dos listas de ruido posicional.

    La posición sigue dentro del vector con el que se ajusta el modelo (ver
    `similitud.config.USE_POSITION_FEATURES`); lo que se excluye es su aparición
    como explicación, y el campo de `contexto` ya la enseña donde corresponde.
    """
    return [
        i for i, nombre in enumerate(modelo.feat_names)
        if not nombre.startswith(glosario.PREFIJO_POSICION)
    ]


def _crudo(crudos: np.ndarray | None, i: int, col: int) -> float | None:
    """Valor real de la métrica `col` de la entidad `i`, si se conoce.

    Sin BD (`crudos is None`) o con la métrica indefinida para esa entidad
    (NaN) se devuelve None, y la interfaz escribe «—».
    """
    if crudos is None:
        return None
    valor = float(crudos[i, col])
    return None if not np.isfinite(valor) else valor


def _fila(crudos: np.ndarray | None, i: int) -> np.ndarray | None:
    """Los valores reales de la entidad `i`, o None si no hay tabla."""
    return None if crudos is None else np.asarray(crudos)[i]


def _valor(fila: np.ndarray | None, col: int) -> float | None:
    """Un valor real suelto de una fila ya extraída (mismo criterio que `_crudo`)."""
    if fila is None:
        return None
    valor = float(fila[col])
    return None if not np.isfinite(valor) else valor


def coincidencias(
    modelo: ModeloSimilitud,
    i: int,
    j: int,
    n: int = config.N_COINCIDENCIAS,
    crudos: np.ndarray | None = None,
) -> tuple[Coincidencia, ...]:
    """Features donde la referencia `i` y el candidato `j` más se parecen.

    Se puntúa cada feature con |z_ref| - |z_ref - z_cand|: alto cuando la
    referencia destaca en ella (es una señal, no ruido alrededor de la media) y
    además el candidato la acompaña. La ELECCIÓN es necesariamente en z-score:
    «destacar» es separarse de la media del resto, y en valores reales el
    ranking lo copiarían siempre las métricas de más volumen (todo el mundo
    «coincidiría en» pases intentados). Lo que se muestra, en cambio, es el valor
    real de `crudos`.

    La posición queda fuera (ver `_columnas_metricas`): coincidir en jugar de
    extremo no es coincidir en cómo se juega.

    Es un resumen a posteriori sobre `feat_display`, que NO participó en el
    ajuste del modelo: sirve para interpretar la recomendación, no la produce.
    """
    return coincidencias_de(
        modelo,
        modelo.feat_display[i], modelo.feat_display[j],
        _fila(crudos, i), _fila(crudos, j),
        n=n,
    )


def coincidencias_de(
    modelo: ModeloSimilitud,
    ref: np.ndarray,
    cand: np.ndarray,
    crudo_ref: np.ndarray | None = None,
    crudo_cand: np.ndarray | None = None,
    n: int = config.N_COINCIDENCIAS,
) -> tuple[Coincidencia, ...]:
    """`coincidencias` a partir de los VECTORES, no de dos filas del modelo.

    Misma regla y mismo resultado; lo que cambia es que la referencia puede no
    tener fila en el artefacto (entidad proyectada), y entonces su vector se
    deriva de la base de datos. Es la única diferencia entre explicar una
    recomendación del modelo y explicar una proyectada.
    """
    if n <= 0:
        return ()
    cols = _columnas_metricas(modelo)
    if not cols:
        return ()
    relevancia = np.abs(ref[cols]) - np.abs(ref[cols] - cand[cols])
    orden = [cols[p] for p in np.argsort(relevancia)[::-1][:n]]
    mapa = fases.fase_por_feature(modelo.entidad)
    return tuple(
        Coincidencia(
            feature=modelo.feat_names[f],
            referencia=float(ref[f]),
            candidato=float(cand[f]),
            referencia_cruda=_valor(crudo_ref, f),
            candidato_cruda=_valor(crudo_cand, f),
            invertida=modelo.feat_names[f] in fases.FEATURES_INVERTIDAS,
            fase=_etiqueta_fase(mapa, modelo.feat_names[f]),
            entidad=modelo.entidad,
        )
        for f in orden
    )


def _etiqueta_fase(mapa: dict[str, fases.Fase], feature: str) -> str | None:
    """Fase de una métrica según `mapa`, o None si no pertenece a ninguna.

    None y no un texto de relleno: una métrica del artefacto que no esté en la
    taxonomía (una añadida al pipeline y todavía no repartida) sale sin fase, y
    la interfaz lo escribe como «—» en vez de asignarle una que no le toca.
    """
    fase = mapa.get(feature)
    return fase.etiqueta if fase is not None else None


def rasgos(
    modelo: ModeloSimilitud,
    i: int,
    n: int = config.N_COINCIDENCIAS,
    crudos: np.ndarray | None = None,
) -> tuple[Rasgo, ...]:
    """Features en las que la entidad `i` más se aleja de la media.

    De QUÉ media depende de la normalización con la que se construyó el modelo
    (la de su liga-temporada en `por_liga`, la de todo el dataset en `global`):
    aquí solo se lee `feat_display`, así que la interfaz es la que debe rotularlo
    (`config.REFERENCIA_Z`).

    Perfil rápido de la referencia (mismos caveats que `coincidencias`: se ordena
    por z-score, se enseña el valor real y la posición queda fuera).
    """
    return rasgos_de(modelo, modelo.feat_display[i], _fila(crudos, i), n=n)


def rasgos_de(
    modelo: ModeloSimilitud,
    valores: np.ndarray,
    crudo: np.ndarray | None = None,
    n: int = config.N_COINCIDENCIAS,
) -> tuple[Rasgo, ...]:
    """`rasgos` a partir del VECTOR de la entidad (ver `coincidencias_de`)."""
    if n <= 0:
        return ()
    cols = _columnas_metricas(modelo)
    if not cols:
        return ()
    orden = [cols[p] for p in np.argsort(np.abs(valores[cols]))[::-1][:n]]
    return tuple(
        Rasgo(
            feature=modelo.feat_names[f],
            valor=float(valores[f]),
            crudo=_valor(crudo, f),
            invertida=modelo.feat_names[f] in fases.FEATURES_INVERTIDAS,
            entidad=modelo.entidad,
        )
        for f in orden
    )


# --- Puntos fuertes y débiles -------------------------------------------------

def _orientadas(modelo: ModeloSimilitud, valores: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """z-scores de una entidad con «más es mejor» en todas las columnas.

    Devuelve (valores orientados, columnas usadas). Las métricas de
    `fases.FEATURES_INVERTIDAS` cambian de signo: sin eso, «le regatean mucho»
    aparecería como punto fuerte por tener un z alto.
    """
    cols = _columnas_metricas(modelo)
    if not cols:
        return np.zeros(0), cols
    orientados = np.array([float(valores[c]) for c in cols])
    signos = np.array([
        -1.0 if modelo.feat_names[c] in fases.FEATURES_INVERTIDAS else 1.0
        for c in cols
    ])
    return orientados * signos, cols


def _rasgo(
    modelo: ModeloSimilitud,
    valores: np.ndarray,
    col: int,
    mapa_fases: dict,
    crudo: np.ndarray | None,
) -> Rasgo:
    nombre = modelo.feat_names[col]
    return Rasgo(
        feature=nombre,
        valor=float(valores[col]),
        crudo=_valor(crudo, col),
        invertida=nombre in fases.FEATURES_INVERTIDAS,
        fase=_etiqueta_fase(mapa_fases, nombre),
        entidad=modelo.entidad,
    )


def destacados_y_flojos(
    modelo: ModeloSimilitud,
    i: int,
    n: int = config.N_RASGOS_FICHA,
    crudos: np.ndarray | None = None,
) -> tuple[tuple[Rasgo, ...], tuple[Rasgo, ...]]:
    """Las `n` métricas donde la entidad va mejor y las `n` donde va peor.

    Se ordena por el z-score ya orientado (ver `_orientadas`), así que la lista
    de puntos fuertes contiene siempre cosas buenas y la de débiles, malas,
    aunque la métrica sea de las que empeoran al subir. Ordenar por el valor real
    no serviría: «destacar» es destacar RESPECTO AL RESTO (su liga-temporada o
    todo el dataset, según la normalización del modelo), y 1,2 despejes por 90
    solo es mucho o poco comparado con los demás. El valor que se enseña, eso sí,
    es el real.

    Las dos listas salen de un único orden y se recortan a `total // 2` cuando el
    modelo tiene menos de `2n` métricas: con pocas features, pedir 4 y 4 haría
    que la misma métrica apareciese a la vez como punto fuerte y como débil.
    """
    return destacados_y_flojos_de(
        modelo, modelo.feat_display[i], _fila(crudos, i), n=n)


def destacados_y_flojos_de(
    modelo: ModeloSimilitud,
    valores: np.ndarray,
    crudo: np.ndarray | None = None,
    n: int = config.N_RASGOS_FICHA,
) -> tuple[tuple[Rasgo, ...], tuple[Rasgo, ...]]:
    """`destacados_y_flojos` a partir del VECTOR (ver `coincidencias_de`)."""
    if n <= 0:
        return (), ()
    orientadas, cols = _orientadas(modelo, valores)
    if not cols:
        return (), ()
    n = min(n, len(cols) // 2)
    if n == 0:
        return (), ()
    mapa = fases.fase_por_feature(modelo.entidad)
    orden = np.argsort(orientadas)  # de peor a mejor
    peores = [cols[p] for p in orden[:n]]
    mejores = [cols[p] for p in orden[::-1][:n]]
    return (
        tuple(_rasgo(modelo, valores, c, mapa, crudo) for c in mejores),
        tuple(_rasgo(modelo, valores, c, mapa, crudo) for c in peores),
    )


# --- Perfil por fases ---------------------------------------------------------

def perfil(matriz: MatrizFases | None, indice: int) -> tuple[ValorFase, ...]:
    """Vértices del radar de fases de la entidad `indice`.

    Sin `matriz` no hay figura: se devuelve vacío y la interfaz omite el bloque.
    """
    if matriz is None:
        return ()
    return fases.perfil(matriz, indice)


# --- Consulta principal -------------------------------------------------------

def recomendar(
    modelo: ModeloSimilitud,
    indice: int,
    k: int = config.TOP_K_DEFECTO,
    n_coincidencias: int = config.N_COINCIDENCIAS,
    matriz: MatrizFases | None = None,
    crudos: np.ndarray | None = None,
    sin_datos: frozenset[int] = frozenset(),
) -> Recomendacion:
    """Top-k del modelo para la entidad `indice`, con su explicación.

    `sin_datos` **no filtra el ranking**: el top-k es el del modelo entero y las
    candidatas que la base de datos de la consulta no tiene salen marcadas
    (`Entidad.en_datos`). Ocultarlas escondería lo que el modelo sí sabe; lo que
    no se puede es preguntarles a ellas (eso lo corta `rutas._sin_datos`).
    """
    referencia = entidad(modelo, indice, sin_datos)
    candidatos = tuple(
        Candidato(
            entidad=entidad(modelo, j, sin_datos),
            rango=rango,
            score=score,
            coincidencias=coincidencias(
                modelo, indice, j, n=n_coincidencias, crudos=crudos
            ),
            perfil=perfil(matriz, j),
        )
        for rango, (j, score) in enumerate(top_k(modelo, indice, k), 1)
    )
    return Recomendacion(
        referencia=referencia,
        rasgos=rasgos(modelo, indice, n=n_coincidencias, crudos=crudos),
        candidatos=candidatos,
        k_pedido=k,
        perfil=perfil(matriz, indice),
    )


def ficha(
    modelo: ModeloSimilitud,
    indice: int,
    matriz: MatrizFases | None = None,
    n_rasgos: int = config.N_RASGOS_FICHA,
    crudos: np.ndarray | None = None,
) -> Ficha:
    """Detalle de una entidad: perfil por fases + puntos fuertes y débiles."""
    destacados, flojos = destacados_y_flojos(modelo, indice, n=n_rasgos, crudos=crudos)
    return Ficha(
        entidad=entidad(modelo, indice),
        perfil=perfil(matriz, indice),
        destacados=destacados,
        flojos=flojos,
    )


# --- Consulta de una entidad que no está en el modelo -------------------------

def matriz_ampliada(
    modelo: ModeloSimilitud, display: np.ndarray
) -> tuple[MatrizFases, int]:
    """Perfil por fases del modelo CON una entidad más al final.

    Devuelve (matriz, índice de la entidad añadida). El radar de una entidad es
    su promedio por fase convertido a **percentil dentro del universo del
    modelo** (`fases.construir_matriz`), así que para poder dibujar el de una
    entidad proyectada hay que meterla en ese universo: con su vector aparte no
    hay percentil que calcular.

    Añadirla mueve los percentiles de los demás en 1/(P+1), que a P = 2640 es
    ruido invisible; a cambio, la referencia y sus candidatos se leen en la misma
    escala, que es la condición para que el radar comparado signifique algo.
    """
    ampliado = np.vstack([modelo.feat_display, np.asarray(display)[None, :]])
    matriz = fases.construir_matriz(modelo.entidad, modelo.feat_names, ampliado)
    return matriz, len(modelo.entity_ids)


def recomendar_proyectada(
    modelo: ModeloSimilitud,
    proyectada,
    k: int = config.TOP_K_DEFECTO,
    n_coincidencias: int = config.N_COINCIDENCIAS,
    crudos: np.ndarray | None = None,
    crudo_referencia: np.ndarray | None = None,
    sin_datos: frozenset[int] = frozenset(),
) -> Recomendacion:
    """Top-k de una entidad que está en la BD y no en el modelo.

    `proyectada` es un `proyeccion.Proyectada`: trae la puntuación contra cada
    entidad del modelo y el vector agregado de la entidad. Se tipa por
    duck-typing y no importando el módulo porque ese vive en la capa de Flask/BD
    y este es el dominio; lo que se necesita de él son tres atributos.

    El resto —qué candidatos, en qué coinciden, qué radar— se calcula con las
    MISMAS funciones que una consulta normal: la diferencia empieza y acaba en de
    dónde salen las puntuaciones.

    `sin_datos` tiene el mismo papel que en `recomendar`: marca a las candidatas
    que la BD de la consulta no tiene, sin quitarlas del ranking.
    """
    matriz, indice_ref = matriz_ampliada(modelo, proyectada.display)
    referencia = Entidad(
        indice=FUERA_DEL_MODELO,
        id=int(proyectada.id),
        nombre=str(proyectada.nombre),
        ligas=tuple(proyectada.ligas),
    )
    por_id = {int(e): j for j, e in enumerate(modelo.entity_ids)}
    candidatos = tuple(
        Candidato(
            entidad=entidad(modelo, por_id[eid], sin_datos),
            rango=rango,
            score=score,
            coincidencias=coincidencias_de(
                modelo,
                proyectada.display, modelo.feat_display[por_id[eid]],
                crudo_referencia, _fila(crudos, por_id[eid]),
                n=n_coincidencias,
            ),
            perfil=perfil(matriz, por_id[eid]),
        )
        for rango, (eid, score) in enumerate(proyectada.proyeccion.top(k), 1)
        if eid in por_id
    )
    return Recomendacion(
        referencia=referencia,
        rasgos=rasgos_de(
            modelo, proyectada.display, crudo_referencia, n=n_coincidencias),
        candidatos=candidatos,
        k_pedido=k,
        perfil=perfil(matriz, indice_ref),
        proyectada=proyectada.fidelidad,
    )


def ficha_proyectada(
    modelo: ModeloSimilitud,
    proyectada,
    n_rasgos: int = config.N_RASGOS_FICHA,
    crudo_referencia: np.ndarray | None = None,
) -> Ficha:
    """Detalle de una entidad que está en la BD y no en el modelo.

    Su perfil y sus puntos fuertes son tan válidos como los de cualquier otra
    (salen del mismo vector de features estandarizado con las mismas mu/sd); lo
    único que no existe es su fila de S, y la ficha no la usa.
    """
    matriz, indice = matriz_ampliada(modelo, proyectada.display)
    destacados, flojos = destacados_y_flojos_de(
        modelo, proyectada.display, crudo_referencia, n=n_rasgos)
    return Ficha(
        entidad=Entidad(
            indice=FUERA_DEL_MODELO,
            id=int(proyectada.id),
            nombre=str(proyectada.nombre),
            ligas=tuple(proyectada.ligas),
        ),
        perfil=perfil(matriz, indice),
        destacados=destacados,
        flojos=flojos,
        proyectada=proyectada.fidelidad,
    )

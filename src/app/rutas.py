"""Vistas HTML y API JSON de la app.

Las vistas son finas a propósito: validan los parámetros de la petición, piden el
modelo al catálogo, delegan en `servicio` y renderizan. Toda la lógica de
dominio (búsqueda, top-k, explicación) está en `servicio`, y todo el ranking, en
`src.similitud`. Lo que viene de la base de datos (equipo del jugador, minutos
por posición) lo aporta `contexto`, y se inyecta aquí para que `servicio` siga
sin conocerla.

Rutas:

- ``GET /``                     formulario de búsqueda.
- ``GET /similares``            resultados (o lista de candidatos si el nombre es ambiguo).
- ``GET /jugador/<id>``         ficha detallada de un jugador.
- ``GET /equipo/<id>``          ficha detallada de un equipo.
- ``GET /glosario``             qué mide cada métrica, por entidad y fase.
- ``GET /api/sugerencias``      autocompletado (JSON).
- ``GET /api/similares``        mismo resultado que ``/similares`` en JSON.
- ``GET /api/ficha/<entidad>/<id>``  misma ficha en JSON.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from flask import Blueprint, abort, current_app, jsonify, render_template, request

from src.similitud import foldin
from src.similitud.consulta import (
    EntidadAmbigua,
    EntidadNoEncontrada,
    normalizar,
    resolver,
)

from . import auth, config, fuentes, glosario, servicio
from .catalogo import Catalogo, Cargado, ClaveModelo, ModeloNoDisponible
from .conjuntos import CatalogoDatos, Conjunto
from .contexto import CatalogoContexto, texto_trayectoria
from .crudos import CatalogoCrudos
from .fases import radar
from .glosario import UNIDAD_CONTEO

bp = Blueprint("web", __name__)


class PeticionInvalida(ValueError):
    """Parámetros de la petición mal formados (se traduce a 400)."""


# --- Helpers de petición ------------------------------------------------------

def _catalogo() -> Catalogo:
    return current_app.extensions["catalogo"]


def _usuario() -> str | None:
    """Cuenta de la petición en curso.

    TODA enumeración de modelos y de conjuntos pasa por aquí: las funciones de
    los catálogos exigen el usuario sin valor por defecto justo para que no se
    pueda listar el disco entero por descuido.
    """
    return auth.usuario_actual()


def _catalogo_datos() -> CatalogoDatos:
    return current_app.extensions["datos"]


def _fuentes() -> fuentes.Fuentes:
    return current_app.extensions["fuentes"]


def _contexto_bd() -> CatalogoContexto:
    return _fuentes().contexto()


def _crudos_bd() -> CatalogoCrudos:
    return _fuentes().crudos()


def _conjunto(clave: ClaveModelo) -> Conjunto:
    """Base de datos sobre la que responde esta petición, y la fija.

    A partir de aquí, todo lo que la interfaz saca de la BD —el equipo de cada
    jugador, sus minutos por posición, los valores reales, los nombres de liga y
    **qué entidades existen**— sale de ESA base de datos.

    Por defecto es la del modelo (`variante.datos`), que es con la que se
    entrenó: sus adornos son los que le corresponden. Pero se puede pedir otra
    con `?datos=<slug>`, y esa es la que manda. Consultar un modelo contra una BD
    más amplia que la suya es justamente lo que permite recomendar a quien no
    estaba cuando se ajustó (ver `_proyectar`): el modelo aporta la geometría
    aprendida y la BD, las entidades.
    """
    variante = _catalogo().variante(clave.variante, _usuario())
    pedido = _datos_pedidos()
    # `resolver` cae al base si ese conjunto ya no está: se pierde algún adorno,
    # pero el modelo se sigue sirviendo.
    conjunto = _catalogo_datos().resolver(pedido or variante.datos)
    fuentes.usar(conjunto.ruta)
    return conjunto


def _datos_pedidos() -> str | None:
    """Conjunto de datos (`?datos=<slug>`) de la petición, si viene.

    Se valida contra los que ESE USUARIO puede ver, y un slug que no le
    corresponde da **404** y no 403, por lo mismo que en `_modelo_pedido`: un 403
    confirmaría que ese conjunto existe.
    """
    valor = (request.args.get("datos") or "").strip()
    if not valor:
        return None
    if valor not in {c.slug for c in _catalogo_datos().conjuntos(_usuario())}:
        abort(404)
    return valor


def _crudos(cargado: Cargado):
    """Valores reales de las métricas alineados con el modelo, o None sin BD."""
    return _crudos_bd().matriz(
        cargado.modelo.entidad,
        cargado.modelo.entity_ids,
        cargado.modelo.feat_names,
    )


def _ajustes() -> config.Config:
    return current_app.config["APP_SIMILITUD"]


def _entidad_pedida() -> str:
    """Tipo de entidad de la petición, validado contra el universo conocido."""
    entidad = (request.args.get("entidad") or "").strip()
    if not entidad:
        # Sin entidad explícita, la primera que tenga modelo (normalmente jugador).
        disponibles = _catalogo().entidades_disponibles(_usuario())
        if not disponibles:
            raise ModeloNoDisponible(
                f"No hay ningún modelo en {_catalogo().model_dir}. "
                f"Constrúyelos con: python -m src.similitud.build"
            )
        return disponibles[0]
    if entidad not in config.ENTIDADES:
        raise PeticionInvalida(
            f"Entidad desconocida: {entidad!r}. Válidas: {', '.join(config.ENTIDADES)}."
        )
    return entidad


def _modelo_pedido() -> str | None:
    """Modelo (`?modelo=<slug>`) de la petición, si viene.

    Se valida contra los que ESE USUARIO puede ver —los suyos más los
    compartidos (el base)—, no contra todo lo que hay en disco: antes bastaba
    con acertar el slug de otra cuenta para que se lo sirvieran.

    Un slug que no le corresponde da **404**, no 403. Un 403 confirma que el
    modelo existe, que es la mitad de la información que se está protegiendo:
    con él se puede sondear qué han entrenado los demás probando nombres.
    """
    valor = (request.args.get("modelo") or "").strip()
    if not valor:
        return None
    if valor not in {v.slug for v in _catalogo().variantes(_usuario())}:
        abort(404)
    return valor


def _k_pedido() -> int:
    """`k` de la petición, acotado a [1, TOP_K_MAX].

    Se acota en vez de rechazar: un `k` enorme no es un error del usuario, es una
    petición cara. El tope evita que una URL manipulada haga renderizar miles de
    filas.
    """
    crudo = request.args.get("k")
    ajustes = _ajustes()
    if crudo is None or not crudo.strip():
        return ajustes.top_k_defecto
    try:
        k = int(crudo)
    except ValueError as e:
        raise PeticionInvalida(f"k debe ser un número entero, no {crudo!r}.") from e
    return max(1, min(k, ajustes.top_k_max))


def _clave_pedida(entidad: str) -> ClaveModelo:
    """Modelo a usar según los parámetros (el base si no se pide otro).

    Además **fija el conjunto de datos** de la petición, el del modelo elegido:
    es el único punto por el que pasan todas las vistas, así que es donde tiene
    que quedar decidido de qué BD salen los adornos (ver `_conjunto`).
    """
    clave = _catalogo().resolver(entidad, _usuario(), variante=_modelo_pedido())
    _conjunto(clave)
    return clave


def _id_pedido() -> int | None:
    """`id` de entidad de la petición, si viene.

    Es la forma preferida de identificar la referencia (la usa el formulario tras
    elegir una sugerencia): el `entity_id` es unívoco, el nombre no.
    """
    crudo = request.args.get("id")
    if crudo is None or not crudo.strip():
        return None
    try:
        return int(crudo)
    except ValueError as e:
        raise PeticionInvalida(f"id debe ser un número entero, no {crudo!r}.") from e


def _localizar(modelo, entity_id: int | None, nombre: str,
               sin_datos: frozenset[int] = frozenset()) -> int:
    """Fila de S de la referencia pedida, por id o por nombre.

    Por nombre se reutiliza `consulta.resolver` (misma semántica que el CLI:
    exacta primero, parcial única después, y fallo ante la ambigüedad).

    Las entidades que la base de datos de la consulta no tiene (`sin_datos`) no
    valen como referencia aunque el modelo las conozca (ver `_sin_datos`), y se
    resuelve **sin ellas**: una entidad invisible tampoco debe volver ambigua una
    búsqueda que sin ella tendría una sola respuesta.

    Si así no sale nada, se vuelve a resolver con TODAS antes de rendirse: cuando
    lo que falla es justo una de las escondidas, el mensaje útil es «está en el
    modelo pero no en estos datos» y no un «no encontrado» indistinguible de un
    nombre mal escrito.
    """
    if entity_id is not None:
        return servicio.indice_por_id(modelo, entity_id, sin_datos)
    visibles = [i for i in range(len(modelo.entity_names)) if i not in sin_datos]
    try:
        return visibles[resolver(nombre, [modelo.entity_names[i] for i in visibles])]
    except (EntidadAmbigua, EntidadNoEncontrada):
        oculta = _resolver_oculta(modelo, nombre, sin_datos)
        if oculta is None:
            raise      # el fallo no lo explica ninguna escondida: es el de siempre
        raise servicio.EntidadSinDatos(
            servicio.sin_datos_texto(modelo, oculta)) from None


def _resolver_oculta(modelo, nombre: str, sin_datos: frozenset[int]) -> int | None:
    """Índice de la entidad ESCONDIDA a la que se refería `nombre`, si es una.

    Se llama solo cuando la búsqueda entre las visibles ya ha fallado, y sirve
    para elegir el mensaje: si el nombre resuelve —con las mismas reglas— a una
    de las que la base de datos no tiene, se puede decir exactamente eso.
    """
    if not sin_datos:
        return None
    try:
        indice = resolver(nombre, modelo.entity_names)
    except (EntidadAmbigua, EntidadNoEncontrada):
        return None
    return indice if indice in sin_datos else None


def _sin_datos(modelo, entidad: str) -> frozenset[int]:
    """Índices de S cuyas entidades NO están en la base de datos de la petición.

    Son las `ajenas` de `cobertura`, vistas desde la consulta. **No se les puede
    pedir nada a ellas**: no salen en el buscador ni en el autocompletado, no
    valen como referencia y no tienen ficha. Sin sus estadísticas no hay equipo,
    ni posiciones, ni valores reales, ni forma de derivar su perfil.

    Lo que sí hacen es **salir como candidatas** de otra: ahí el artefacto ya trae
    todo lo que hace falta (nombre, liga, puntuación) y esconderlas tiraría
    justamente lo que el modelo sabe. Salen marcadas y sin enlace
    (`servicio.Entidad.en_datos`).

    Vacío cuando la BD no se puede leer (`universo().ids` devuelve None): sin
    universo con el que comparar no se afirma nada, igual que en `cobertura`, y
    lo seguro es no esconder a nadie.
    """
    ids_bd = _fuentes().universo().ids(entidad)
    if ids_bd is None:
        return frozenset()
    return frozenset(
        i for i, eid in enumerate(modelo.entity_ids) if int(eid) not in ids_bd
    )


# --- Entidades que están en la base de datos y no en el modelo ----------------

def _nombres_bd(entidad: str) -> dict[int, str] | None:
    """id -> nombre de las entidades que tiene la BD de esta petición."""
    return _fuentes().universo().nombres(entidad)


def _ausentes(modelo, entidad: str, texto: str,
              limite: int) -> list[servicio.Entidad]:
    """Las que coinciden con `texto` y no están en el modelo (proyectables)."""
    return servicio.buscar_ausentes(modelo, _nombres_bd(entidad), texto, limite)


def _id_ausente(modelo, entidad: str, entity_id: int | None,
                nombre: str) -> int | None:
    """`entity_id` de una entidad de la BD que el modelo no tiene, o None.

    Es el segundo intento de la búsqueda: solo se llega aquí cuando el modelo no
    ha sabido resolver la consulta. Por id basta con que la BD lo conozca; por
    nombre se exige una ÚNICA candidata, con el mismo criterio que
    `consulta.resolver` usa dentro del modelo (exacta primero, parcial única
    después). Con varias no se elige por el usuario: se le enseña la lista.
    """
    nombres = _nombres_bd(entidad)
    if not nombres:
        return None
    del_modelo = {int(i) for i in modelo.entity_ids}
    if entity_id is not None:
        return (int(entity_id)
                if int(entity_id) in nombres and int(entity_id) not in del_modelo
                else None)
    candidatas = servicio.buscar_ausentes(modelo, nombres, nombre, limite=0)
    if not candidatas:
        return None
    # `normalizar` y no una comparación de cadenas: es lo que usa `resolver`
    # dentro del modelo, y con otra regla «Söyüncü» sería exacto en un lado y
    # parcial en el otro.
    objetivo = normalizar(nombre)
    exactas = [c for c in candidatas if normalizar(c.nombre) == objetivo]
    if len(exactas) == 1:
        return exactas[0].id
    return candidatas[0].id if len(candidatas) == 1 else None


def _proyectar(clave: ClaveModelo, modelo, entity_id: int):
    """Coloca en el modelo una entidad que solo está en la BD elegida.

    Devuelve un `proyeccion.Proyectada`. Traduce los dos «no se puede» a
    excepciones que las vistas ya saben pintar, porque para el usuario son
    mensajes distintos: que esta BD no sirva para proyectar sobre este modelo
    (falta el estado warm, otro espacio de features) no es lo mismo que el modelo
    no admita proyecciones.
    """
    proyectada = _fuentes().proyeccion().proyectar(
        modelo, _catalogo().ruta_artefacto(clave), entity_id)
    if proyectada is None:
        raise foldin.EntidadNoProyectable(
            "No se puede colocar a esta entidad en el modelo con estos datos: "
            "el modelo no guarda con qué se estandarizó o su ajuste no está "
            "completo. Entrena un modelo con este conjunto de datos desde "
            "«Datos y modelos» y volverá a aparecer."
        )
    return proyectada


def _crudos_de(modelo, entity_id: int):
    """Valores reales de UNA entidad de la BD, alineados con `feat_names`.

    La tabla de `crudos` se deriva de la BD entera, así que cubre también a las
    entidades que no están en el modelo: una entidad proyectada puede enseñar sus
    métricas en unidades reales igual que cualquier otra. None sin BD.
    """
    matriz = _crudos_bd().matriz(
        modelo.entidad, np.array([int(entity_id)]), modelo.feat_names)
    return None if matriz is None else matriz[0]


# --- Contexto común de las plantillas -----------------------------------------

def _referencia_z(clave: ClaveModelo | None) -> str:
    """Contra qué se mide el z-score en el modelo que se está sirviendo.

    Lo decide la normalización, no la plantilla: con `global` la media es la de
    todo el dataset y llamarla «la de su liga» sería falso.
    """
    if clave is None:
        return config.REFERENCIA_Z_DEFECTO
    return config.REFERENCIA_Z.get(clave.normalizacion, config.REFERENCIA_Z_DEFECTO)


def _datos_del_modelo(clave: ClaveModelo | None) -> Conjunto | None:
    """Conjunto con el que se entrenó el modelo servido, o None si no hay modelo."""
    if clave is None:
        return None
    try:
        variante = _catalogo().variante(clave.variante, _usuario())
    except ModeloNoDisponible:
        return None
    return _catalogo_datos().resolver(variante.datos)


def _contexto_base(entidad: str, clave: ClaveModelo | None = None) -> dict[str, Any]:
    catalogo = _catalogo()
    # Los modelos del desplegable son los que cubren ESTA entidad: uno recién
    # reentrenado puede tener todavía el artefacto de jugador y no el de equipo.
    variantes = catalogo.variantes_de(entidad, _usuario())
    conjunto = _conjunto(clave) if clave is not None else None
    return {
        # Bases de datos entre las que se puede elegir: las que esa cuenta ve y
        # tienen BD escrita (una recién creada puede estar todavía copiándose).
        "conjuntos": _catalogo_datos().disponibles(_usuario()),
        # Con cuál se entrenó el modelo que se está sirviendo. Se pasa aparte del
        # conjunto elegido porque cuando NO coinciden hay que decirlo: es la
        # situación en la que aparecen entidades proyectadas.
        "datos_del_modelo": _datos_del_modelo(clave),
        "entidad": entidad,
        "entidades": catalogo.entidades_disponibles(_usuario()),
        "variantes": variantes,
        # Con qué se construye esta entidad (misma receta en todos los modelos):
        # la interfaz lo rotula aunque el usuario no pueda elegirlo.
        "receta": ClaveModelo(entidad),
        "variante": next(
            (v for v in variantes if clave is not None and v.slug == clave.variante),
            None,
        ),
        # Sobre qué datos responde el modelo: la interfaz lo rotula junto a él,
        # porque es la otra mitad de la respuesta (mismo modelo sobre otra BD da
        # otras recomendaciones).
        "conjunto": conjunto,
        "nombres_conjunto": {
            c.slug: c.nombre for c in _catalogo_datos().conjuntos(_usuario())
        },
        "clave": clave,
        "k": _k_pedido(),
        # Qué se le dice al usuario cuando la referencia se ha proyectado. El
        # texto (y las cifras que cita) no son de la plantilla: ver `config`.
        "texto_proyeccion": config.TEXTO_PROYECCION,
        "etiqueta_entidad": config.ETIQUETA_ENTIDAD,
        "etiqueta_plural": config.ETIQUETA_PLURAL,
        "etiqueta_formulacion": config.ETIQUETA_FORMULACION,
        "etiqueta_normalizacion": config.ETIQUETA_NORMALIZACION,
        "referencia_z": _referencia_z(clave),
    }


def _json_entidad(e: servicio.Entidad, entidad: str) -> dict[str, Any]:
    """Entidad en JSON.

    Se exponen las dos formas de la liga: `ligas` son las claves
    `competition-season` (identidad estable, la que usa el pipeline) y
    `ligas_nombre` su traducción legible, que depende de que la BD esté presente.
    En el jugador se añade su trayectoria (equipos por liga, en orden
    cronológico), que también sale de la BD y va vacía sin ella.
    """
    catalogo_ligas = _fuentes().ligas()
    salida = {
        "id": e.id,
        "indice": e.indice,
        "nombre": e.nombre,
        "ligas": list(e.ligas),
        "ligas_nombre": catalogo_ligas.nombres(e.ligas),
        # False = está en la BD consultada pero no en el modelo, así que su
        # respuesta sale de una proyección y no de la matriz S.
        "en_modelo": e.en_modelo,
        # El caso simétrico: False = el modelo la tiene y la BD consultada no.
        # Puede salir como candidata (el modelo sabe cuánto se parece) pero no
        # tiene ficha ni se le pueden pedir similares a ella.
        "en_datos": e.en_datos,
    }
    if entidad == "jugador":
        jugador = _contexto_bd().jugador(e.id)
        actual = jugador.equipo_actual
        salida["equipo"] = actual.nombre if actual else None
        salida["contexto"] = texto_trayectoria(
            jugador.trayectoria, catalogo_ligas.nombre
        )
        # Agrupada por liga y en orden cronológico, igual que en la interfaz.
        salida["trayectoria"] = [
            {
                "liga": p.liga,
                "liga_nombre": catalogo_ligas.nombre(p.liga),
                "equipos": [
                    {"id": m.equipo_id, "nombre": m.nombre,
                     "minutos": round(m.minutos, 1), "partidos": m.partidos,
                     "desde": m.desde}
                    for m in p.equipos
                ],
            }
            for p in jugador.trayectoria
        ]
    return salida


def _json_perfil(valores) -> list[dict[str, Any]]:
    return [
        {
            "fase": v.fase.clave,
            "etiqueta": v.fase.etiqueta,
            "percentil": v.percentil,
            "z": v.z,
        }
        for v in valores
    ]


def _json_rasgo(r: servicio.Rasgo) -> dict[str, Any]:
    """Rasgo en JSON.

    `valor` sigue siendo el z-score (es el criterio con el que se ha elegido esa
    métrica) y `crudo` el valor real en sus unidades, que es lo que se enseña.
    `crudo` es `null` sin BD o si la métrica no está definida para la entidad.
    """
    return {
        "feature": r.feature,
        "etiqueta": r.etiqueta,
        "valor": r.valor,
        "crudo": r.crudo,
        "texto": r.texto,
        "invertida": r.invertida,
        "fase": r.fase,
    }


def _json_modelo(clave: ClaveModelo) -> dict[str, Any]:
    """Con qué se ha respondido: el modelo elegido y de qué está hecho.

    `variante` es lo que el usuario elige (y lo que se pasa como `?modelo=`);
    `formulacion` y `normalizacion` van también porque son la receta con la que
    se construyó, y quien consuma la API tiene derecho a saberla sin abrir el
    artefacto.
    """
    catalogo = _catalogo()
    try:
        variante = catalogo.variante(clave.variante, _usuario())
        nombre, datos = variante.nombre, variante.datos
    except ModeloNoDisponible:
        nombre, datos = clave.variante, config.VARIANTE_BASE
    return {
        "entidad": clave.entidad,
        "variante": clave.variante,
        "nombre": nombre,
        # Con qué datos se ENTRENÓ y sobre cuáles se ha CONSULTADO. Suelen ser
        # los mismos, pero se pueden separar (`?datos=`), y entonces la
        # diferencia es justo lo que explica que haya entidades proyectadas.
        "datos": datos,
        "datos_consultados": _datos_pedidos() or datos,
        "formulacion": clave.formulacion,
        "normalizacion": clave.normalizacion,
    }


def _json_recomendacion(rec: servicio.Recomendacion, clave: ClaveModelo) -> dict[str, Any]:
    entidad = clave.entidad
    return {
        "modelo": _json_modelo(clave),
        "referencia": _json_entidad(rec.referencia, entidad),
        "perfil": _json_perfil(rec.perfil),
        "rasgos": [_json_rasgo(r) for r in rec.rasgos],
        "k": rec.k_pedido,
        "incompleto": rec.incompleto,
        # None si la referencia estaba en el modelo; `fiel`/`aproximada` si se ha
        # proyectado (ver `src.similitud.foldin`). Quien consuma la API tiene que
        # poder distinguirlo sin mirar la interfaz.
        "proyectada": rec.proyectada,
        "candidatos": [
            {
                **_json_entidad(c.entidad, entidad),
                "rango": c.rango,
                "score": c.score,
                "perfil": _json_perfil(c.perfil),
                "coincidencias": [
                    {
                        "feature": co.feature,
                        "etiqueta": co.etiqueta,
                        # z-score (con lo que se elige) y valor real (lo que se
                        # muestra); el segundo es null si no hay BD.
                        "referencia": co.referencia,
                        "candidato": co.candidato,
                        "referencia_cruda": co.referencia_cruda,
                        "candidato_cruda": co.candidato_cruda,
                        "texto_referencia": co.texto_referencia,
                        "texto_candidato": co.texto_candidato,
                        "invertida": co.invertida,
                        "fase": co.fase,
                    }
                    for co in c.coincidencias
                ],
            }
            for c in rec.candidatos
        ],
    }


# --- Vistas HTML --------------------------------------------------------------

@bp.get("/")
def inicio():
    """Formulario de búsqueda."""
    catalogo = _catalogo()
    entidades = catalogo.entidades_disponibles(_usuario())
    if not entidades:
        return render_template(
            "sin_modelos.html", model_dir=catalogo.model_dir
        ), 503
    entidad = _entidad_pedida()
    # Se resuelve el modelo aunque aquí no se consulte: fija el conjunto de datos
    # de la petición (ver `_clave_pedida`), que es lo que rotula el buscador y lo
    # que decide qué ofrece el autocompletado. Resolver no carga la matriz S.
    clave = _clave_pedida(entidad)
    return render_template("inicio.html", **_contexto_base(entidad, clave))


@bp.get("/similares")
def similares():
    """Top-k de la entidad pedida (por `id` o por `nombre`)."""
    entidad = _entidad_pedida()
    clave = _clave_pedida(entidad)
    cargado = _catalogo().cargado(clave)
    modelo = cargado.modelo
    nombre = (request.args.get("nombre") or "").strip()
    entity_id = _id_pedido()
    contexto = _contexto_base(entidad, clave)

    if entity_id is None and not nombre:
        raise PeticionInvalida("Indica un nombre a buscar.")

    crudos = _crudos(cargado)
    sin_datos = _sin_datos(modelo, entidad)
    try:
        indice = _localizar(modelo, entity_id, nombre, sin_datos)
    except (EntidadAmbigua, EntidadNoEncontrada, servicio.EntidadDesconocida) as e:
        # El modelo no la tiene. Antes de rendirse: puede que la BD elegida sí,
        # y entonces se la puede colocar en el modelo (ver `_proyectar`). Esto es
        # lo que hace que ampliar los datos —o elegir otra base— sirva de algo
        # sin tener que reentrenar primero.
        ausente = _id_ausente(modelo, entidad, entity_id, nombre)
        if ausente is not None:
            try:
                rec = servicio.recomendar_proyectada(
                    modelo,
                    _proyectar(clave, modelo, ausente),
                    k=contexto["k"],
                    n_coincidencias=_ajustes().n_coincidencias,
                    crudos=crudos,
                    crudo_referencia=_crudos_de(modelo, ausente),
                    sin_datos=sin_datos,
                )
                return _pintar_similares(entidad, rec, crudos, contexto)
            except foldin.EntidadNoProyectable as fallo:
                return render_template(
                    "error.html", mensaje=str(fallo), **contexto), 409
        if isinstance(e, servicio.EntidadDesconocida):
            return render_template("error.html", mensaje=str(e), **contexto), 404
        # Varias coincidencias parciales (o ninguna): en vez de fallar como el
        # CLI, se listan para que el usuario elija —incluyendo las que solo están
        # en la BD, que también se pueden consultar—. Con la lista vacía es un
        # 404 con la misma plantilla.
        candidatos = servicio.buscar(
            modelo, nombre, limite=config.MAX_CANDIDATOS_AMBIGUOS,
            sin_datos=sin_datos,
        ) + _ausentes(modelo, entidad, nombre, config.MAX_CANDIDATOS_AMBIGUOS)
        return render_template(
            "candidatos.html", consulta=nombre, candidatos=candidatos, **contexto
        ), (200 if candidatos else 404)

    rec = servicio.recomendar(
        modelo,
        indice,
        k=contexto["k"],
        n_coincidencias=_ajustes().n_coincidencias,
        matriz=cargado.fases,
        crudos=crudos,
        sin_datos=sin_datos,
    )
    return _pintar_similares(entidad, rec, crudos, contexto)


def _pintar_similares(entidad: str, rec: servicio.Recomendacion,
                      crudos, contexto: dict[str, Any]):
    """Página de resultados, venga el top-k de la S o de una proyección."""
    # Un radar por candidato, con la referencia debajo para leer el parecido de
    # un vistazo; y el reparto de minutos por posición cuando hay BD.
    comparativas = {
        c.entidad.indice: _comparativa(entidad, rec, c) for c in rec.candidatos
    }
    return render_template(
        "resultados.html",
        rec=rec,
        radar_referencia=radar([("Referencia", "referencia", rec.perfil)]),
        comparativas=comparativas,
        unidad=UNIDAD_CONTEO.get(entidad, ""),
        # Sin BD no hay valores reales: la interfaz vuelve a enseñar solo el
        # z-score en vez de una columna entera de guiones.
        hay_crudos=crudos is not None,
        **contexto,
    )


def _comparativa(entidad: str, rec: servicio.Recomendacion,
                 candidato: servicio.Candidato) -> dict[str, Any]:
    """Figuras que explican por qué se parecen la referencia y un candidato."""
    marcas = ()
    if entidad == "jugador":
        marcas = _contexto_bd().marcas(rec.referencia.id, candidato.entidad.id)
    return {
        "radar": radar([
            (rec.referencia.nombre, "referencia", rec.perfil),
            (candidato.entidad.nombre, "candidato", candidato.perfil),
        ]),
        "marcas": marcas,
    }


def _ficha(entidad: str, entity_id: int):
    """Página de detalle de una entidad (compartida por jugador y equipo)."""
    clave = _clave_pedida(entidad)
    cargado: Cargado = _catalogo().cargado(clave)
    contexto = _contexto_base(entidad, clave)
    crudos = _crudos(cargado)
    sin_datos = _sin_datos(cargado.modelo, entidad)
    try:
        indice = servicio.indice_por_id(cargado.modelo, entity_id, sin_datos)
    except servicio.EntidadDesconocida as e:
        # Igual que en `/similares`: si la BD la conoce, su ficha se puede armar
        # entera (perfil, puntos fuertes y flojos) aunque no tenga fila en S.
        if _id_ausente(cargado.modelo, entidad, entity_id, "") is None:
            return render_template("error.html", mensaje=str(e), **contexto), 404
        try:
            detalle = servicio.ficha_proyectada(
                cargado.modelo,
                _proyectar(clave, cargado.modelo, entity_id),
                n_rasgos=_ajustes().n_rasgos_ficha,
                crudo_referencia=_crudos_de(cargado.modelo, entity_id),
            )
        except foldin.EntidadNoProyectable as fallo:
            return render_template("error.html", mensaje=str(fallo), **contexto), 409
        return _pintar_ficha(entidad, entity_id, detalle, crudos, contexto)

    detalle = servicio.ficha(
        cargado.modelo,
        indice,
        matriz=cargado.fases,
        n_rasgos=_ajustes().n_rasgos_ficha,
        crudos=crudos,
    )
    return _pintar_ficha(entidad, entity_id, detalle, crudos, contexto)


def _pintar_ficha(entidad: str, entity_id: int, detalle: servicio.Ficha,
                  crudos, contexto: dict[str, Any]):
    """Página de detalle, esté la entidad en el modelo o proyectada."""
    datos_bd = _contexto_bd().jugador(entity_id) if entidad == "jugador" else None
    return render_template(
        "ficha.html",
        ficha=detalle,
        radar=radar([(detalle.entidad.nombre, "referencia", detalle.perfil)]),
        datos_bd=datos_bd,
        marcas=_contexto_bd().marcas(entity_id) if entidad == "jugador" else (),
        n_posiciones=config.N_POSICIONES_FICHA,
        unidad=UNIDAD_CONTEO.get(entidad, ""),
        hay_crudos=crudos is not None,
        **contexto,
    )


@bp.get("/jugador/<int:entity_id>")
def ficha_jugador(entity_id: int):
    """Ficha de un jugador."""
    return _ficha("jugador", entity_id)


@bp.get("/equipo/<int:entity_id>")
def ficha_equipo(entity_id: int):
    """Ficha de un equipo."""
    return _ficha("equipo", entity_id)


@bp.get("/glosario")
def glosario_metricas():
    """Qué mide cada métrica, agrupada por entidad y fase de juego.

    Deliberadamente NO usa `_contexto_base`: el glosario describe el catálogo de
    métricas del pipeline, que no depende de qué modelos haya en disco ni de qué
    conjunto de datos se esté sirviendo. Así se puede consultar también cuando la
    app está recién instalada y todavía no hay nada construido, que es justo
    cuando más falta hace saber qué significa cada cosa.
    """
    return render_template(
        "glosario.html",
        secciones=[
            {
                "entidad": entidad,
                "etiqueta": config.ETIQUETA_PLURAL[entidad],
                "unidad_conteo": UNIDAD_CONTEO.get(entidad, ""),
                "bloques": glosario.glosario_de(entidad),
            }
            for entidad in config.ENTIDADES
        ],
        definicion_posicion=glosario.DEFINICION_POSICION,
    )


# --- API JSON -----------------------------------------------------------------

@bp.get("/api/sugerencias")
def api_sugerencias():
    """Autocompletado: entidades cuyo nombre contiene `q`.

    Ofrece las del modelo y, detrás, las que solo están en la base de datos
    elegida (marcadas con `en_modelo: false`). Van detrás y no mezcladas porque
    su respuesta es de otra clase —proyectada— y lo normal es querer primero las
    que el modelo tiene ajustadas.
    """
    entidad = _entidad_pedida()
    clave = _clave_pedida(entidad)
    modelo = _catalogo().obtener(clave)
    texto = request.args.get("q") or ""
    tope = _ajustes().max_sugerencias
    sugerencias = servicio.buscar(
        modelo, texto, limite=tope, sin_datos=_sin_datos(modelo, entidad))
    faltan = max(0, tope - len(sugerencias))
    if faltan:
        sugerencias += _ausentes(modelo, entidad, texto, faltan)
    return jsonify({
        "entidad": entidad,
        "consulta": texto,
        "sugerencias": [_json_entidad(e, entidad) for e in sugerencias],
    })


@bp.get("/api/similares")
def api_similares():
    """Mismo resultado que `/similares`, en JSON."""
    entidad = _entidad_pedida()
    clave = _clave_pedida(entidad)
    cargado = _catalogo().cargado(clave)
    modelo = cargado.modelo
    nombre = (request.args.get("nombre") or "").strip()
    entity_id = _id_pedido()
    if entity_id is None and not nombre:
        raise PeticionInvalida("Falta 'id' o 'nombre'.")
    crudos = _crudos(cargado)
    sin_datos = _sin_datos(modelo, entidad)
    try:
        indice = _localizar(modelo, entity_id, nombre, sin_datos)
    except (EntidadAmbigua, EntidadNoEncontrada, servicio.EntidadDesconocida) as e:
        # Mismo orden que en la vista: primero se intenta proyectar desde la BD.
        ausente = _id_ausente(modelo, entidad, entity_id, nombre)
        if ausente is not None:
            try:
                rec = servicio.recomendar_proyectada(
                    modelo,
                    _proyectar(clave, modelo, ausente),
                    k=_k_pedido(),
                    n_coincidencias=_ajustes().n_coincidencias,
                    crudos=crudos,
                    crudo_referencia=_crudos_de(modelo, ausente),
                    sin_datos=sin_datos,
                )
                return jsonify(_json_recomendacion(rec, clave))
            except foldin.EntidadNoProyectable as fallo:
                return jsonify({"error": str(fallo)}), 409
        if isinstance(e, servicio.EntidadDesconocida):
            return jsonify({"error": str(e)}), 404
        # La API sí distingue los dos casos: 300 (elige uno) vs 404 (no existe).
        candidatos = servicio.buscar(
            modelo, nombre, limite=config.MAX_CANDIDATOS_AMBIGUOS,
            sin_datos=sin_datos,
        ) + _ausentes(modelo, entidad, nombre, config.MAX_CANDIDATOS_AMBIGUOS)
        estado = 300 if isinstance(e, EntidadAmbigua) else 404
        return jsonify({
            "error": str(e),
            "candidatos": [_json_entidad(c, entidad) for c in candidatos],
        }), estado

    rec = servicio.recomendar(
        modelo,
        indice,
        k=_k_pedido(),
        n_coincidencias=_ajustes().n_coincidencias,
        matriz=cargado.fases,
        crudos=crudos,
        sin_datos=sin_datos,
    )
    return jsonify(_json_recomendacion(rec, clave))


@bp.get("/api/ficha/<entidad>/<int:entity_id>")
def api_ficha(entidad: str, entity_id: int):
    """Ficha de una entidad en JSON (mismo contenido que la página)."""
    if entidad not in config.ENTIDADES:
        raise PeticionInvalida(
            f"Entidad desconocida: {entidad!r}. Válidas: {', '.join(config.ENTIDADES)}."
        )
    clave = _clave_pedida(entidad)
    cargado = _catalogo().cargado(clave)
    try:
        indice = servicio.indice_por_id(
            cargado.modelo, entity_id, _sin_datos(cargado.modelo, entidad))
        detalle = servicio.ficha(
            cargado.modelo,
            indice,
            matriz=cargado.fases,
            n_rasgos=_ajustes().n_rasgos_ficha,
            crudos=_crudos(cargado),
        )
    except servicio.EntidadDesconocida as e:
        if _id_ausente(cargado.modelo, entidad, entity_id, "") is None:
            return jsonify({"error": str(e)}), 404
        try:
            detalle = servicio.ficha_proyectada(
                cargado.modelo,
                _proyectar(clave, cargado.modelo, entity_id),
                n_rasgos=_ajustes().n_rasgos_ficha,
                crudo_referencia=_crudos_de(cargado.modelo, entity_id),
            )
        except foldin.EntidadNoProyectable as fallo:
            return jsonify({"error": str(fallo)}), 409
    salida = {
        "modelo": _json_modelo(clave),
        "entidad": _json_entidad(detalle.entidad, entidad),
        "perfil": _json_perfil(detalle.perfil),
        "destacados": [_json_rasgo(r) for r in detalle.destacados],
        "flojos": [_json_rasgo(r) for r in detalle.flojos],
        "proyectada": detalle.proyectada,
    }
    if entidad == "jugador":
        datos = _contexto_bd().jugador(entity_id)
        salida["posiciones"] = [
            {
                "slug": u.posicion.slug,
                "etiqueta": u.posicion.etiqueta,
                "fraccion": u.fraccion,
                "minutos": round(u.minutos, 1),
            }
            for u in datos.posiciones
        ]
    return jsonify(salida)


# --- Errores ------------------------------------------------------------------

def _es_api() -> bool:
    return request.path.startswith("/api/")


@bp.app_errorhandler(PeticionInvalida)
def _peticion_invalida(e: PeticionInvalida):
    if _es_api():
        return jsonify({"error": str(e)}), 400
    return render_template("error.html", mensaje=str(e), **_contexto_minimo()), 400


@bp.app_errorhandler(ModeloNoDisponible)
def _modelo_no_disponible(e: ModeloNoDisponible):
    if _es_api():
        return jsonify({"error": str(e)}), 503
    # Si hay OTROS modelos, lo que falla es el pedido (p. ej. un reentrenamiento
    # que solo llegó a dejar el artefacto de jugador): decirle al usuario que no
    # hay nada sería falso, así que se muestra el error concreto.
    if _catalogo().disponibles(_usuario()):
        return render_template("error.html", mensaje=str(e), **_contexto_minimo()), 503
    return render_template(
        "sin_modelos.html", model_dir=_catalogo().model_dir, mensaje=str(e)
    ), 503


@bp.app_errorhandler(404)
def _no_encontrado(e):
    if _es_api():
        return jsonify({"error": "Ruta no encontrada."}), 404
    return render_template(
        "error.html", mensaje="La página que buscas no existe.", **_contexto_minimo()
    ), 404


def _contexto_minimo() -> dict[str, Any]:
    """Contexto suficiente para pintar la cabecera en las páginas de error.

    No usa `_contexto_base` porque este puede volver a fallar (la petición que ha
    fallado es justamente la que trae parámetros inválidos).
    """
    catalogo = _catalogo()
    entidades = catalogo.entidades_disponibles(_usuario())
    return {
        "entidad": entidades[0] if entidades else config.ENTIDADES[0],
        "entidades": entidades,
        "variantes": [],
        "variante": None,
        "conjunto": None,
        "conjuntos": [],
        "datos_del_modelo": None,
        "nombres_conjunto": {},
        "receta": ClaveModelo(entidades[0] if entidades else config.ENTIDADES[0]),
        "clave": None,
        "k": config.TOP_K_DEFECTO,
        "etiqueta_entidad": config.ETIQUETA_ENTIDAD,
        "etiqueta_plural": config.ETIQUETA_PLURAL,
        "etiqueta_formulacion": config.ETIQUETA_FORMULACION,
        "etiqueta_normalizacion": config.ETIQUETA_NORMALIZACION,
        "referencia_z": config.REFERENCIA_Z_DEFECTO,
    }

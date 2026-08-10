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

from flask import Blueprint, current_app, jsonify, render_template, request

from src.similitud.consulta import EntidadAmbigua, EntidadNoEncontrada, resolver

from . import config, fuentes, glosario, servicio
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


def _catalogo_datos() -> CatalogoDatos:
    return current_app.extensions["datos"]


def _fuentes() -> fuentes.Fuentes:
    return current_app.extensions["fuentes"]


def _contexto_bd() -> CatalogoContexto:
    return _fuentes().contexto()


def _crudos_bd() -> CatalogoCrudos:
    return _fuentes().crudos()


def _conjunto(clave: ClaveModelo) -> Conjunto:
    """Conjunto de datos del modelo `clave`, y lo fija para esta petición.

    A partir de aquí, todo lo que la interfaz saca de la BD —el equipo de cada
    jugador, sus minutos por posición, los valores reales y los nombres de
    liga— sale de ESA base de datos. Sin esto, un modelo entrenado sobre datos
    ampliados se serviría con los adornos de la BD vieja y las entidades nuevas
    saldrían sin nada.
    """
    variante = _catalogo().variante(clave.variante)
    # `resolver` cae al base si ese conjunto ya no está: se pierde algún adorno,
    # pero el modelo se sigue sirviendo.
    conjunto = _catalogo_datos().resolver(variante.datos)
    fuentes.usar(conjunto.ruta)
    return conjunto


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
        disponibles = _catalogo().entidades_disponibles()
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

    Se valida contra los que existen en disco y no contra una lista fija: los
    modelos con nombre los crea el usuario reentrenando, así que el universo
    cambia mientras la app corre.
    """
    valor = (request.args.get("modelo") or "").strip()
    if not valor:
        return None
    disponibles = [v.slug for v in _catalogo().variantes()]
    if valor not in disponibles:
        raise PeticionInvalida(
            f"Modelo desconocido: {valor!r}. Disponibles: {', '.join(disponibles)}."
        )
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
    clave = _catalogo().resolver(entidad, variante=_modelo_pedido())
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


def _localizar(modelo, entity_id: int | None, nombre: str) -> int:
    """Fila de S de la referencia pedida, por id o por nombre.

    Por nombre se reutiliza `consulta.resolver` (misma semántica que el CLI:
    exacta primero, parcial única después, y fallo ante la ambigüedad).
    """
    if entity_id is not None:
        return servicio.indice_por_id(modelo, entity_id)
    return resolver(nombre, modelo.entity_names)


# --- Contexto común de las plantillas -----------------------------------------

def _referencia_z(clave: ClaveModelo | None) -> str:
    """Contra qué se mide el z-score en el modelo que se está sirviendo.

    Lo decide la normalización, no la plantilla: con `global` la media es la de
    todo el dataset y llamarla «la de su liga» sería falso.
    """
    if clave is None:
        return config.REFERENCIA_Z_DEFECTO
    return config.REFERENCIA_Z.get(clave.normalizacion, config.REFERENCIA_Z_DEFECTO)


def _contexto_base(entidad: str, clave: ClaveModelo | None = None) -> dict[str, Any]:
    catalogo = _catalogo()
    # Los modelos del desplegable son los que cubren ESTA entidad: uno recién
    # reentrenado puede tener todavía el artefacto de jugador y no el de equipo.
    variantes = catalogo.variantes_de(entidad)
    return {
        "entidad": entidad,
        "entidades": catalogo.entidades_disponibles(),
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
        "conjunto": _conjunto(clave) if clave is not None else None,
        "nombres_conjunto": {
            c.slug: c.nombre for c in _catalogo_datos().conjuntos()
        },
        "clave": clave,
        "k": _k_pedido(),
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
        variante = catalogo.variante(clave.variante)
        nombre, datos = variante.nombre, variante.datos
    except ModeloNoDisponible:
        nombre, datos = clave.variante, config.VARIANTE_BASE
    return {
        "entidad": clave.entidad,
        "variante": clave.variante,
        "nombre": nombre,
        "datos": datos,
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
    entidades = catalogo.entidades_disponibles()
    if not entidades:
        return render_template(
            "sin_modelos.html", model_dir=catalogo.model_dir
        ), 503
    entidad = _entidad_pedida()
    return render_template("inicio.html", **_contexto_base(entidad))


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

    try:
        indice = _localizar(modelo, entity_id, nombre)
    except EntidadAmbigua:
        # Varias coincidencias parciales: en vez de fallar como el CLI, se listan
        # para que el usuario elija (es la respuesta útil en una interfaz).
        candidatos = servicio.buscar(
            modelo, nombre, limite=config.MAX_CANDIDATOS_AMBIGUOS
        )
        return render_template(
            "candidatos.html", consulta=nombre, candidatos=candidatos, **contexto
        )
    except EntidadNoEncontrada:
        return render_template(
            "candidatos.html", consulta=nombre, candidatos=[], **contexto
        ), 404
    except servicio.EntidadDesconocida as e:
        return render_template("error.html", mensaje=str(e), **contexto), 404

    crudos = _crudos(cargado)
    rec = servicio.recomendar(
        modelo,
        indice,
        k=contexto["k"],
        n_coincidencias=_ajustes().n_coincidencias,
        matriz=cargado.fases,
        crudos=crudos,
    )
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
    try:
        indice = servicio.indice_por_id(cargado.modelo, entity_id)
    except servicio.EntidadDesconocida as e:
        return render_template("error.html", mensaje=str(e), **contexto), 404

    crudos = _crudos(cargado)
    detalle = servicio.ficha(
        cargado.modelo,
        indice,
        matriz=cargado.fases,
        n_rasgos=_ajustes().n_rasgos_ficha,
        crudos=crudos,
    )
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
    """Autocompletado: entidades cuyo nombre contiene `q`."""
    entidad = _entidad_pedida()
    clave = _clave_pedida(entidad)
    modelo = _catalogo().obtener(clave)
    texto = request.args.get("q") or ""
    sugerencias = servicio.buscar(modelo, texto, limite=_ajustes().max_sugerencias)
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
    try:
        indice = _localizar(modelo, entity_id, nombre)
    except (EntidadAmbigua, EntidadNoEncontrada) as e:
        # La API sí distingue los dos casos: 300 (elige uno) vs 404 (no existe).
        candidatos = servicio.buscar(
            modelo, nombre, limite=config.MAX_CANDIDATOS_AMBIGUOS
        )
        estado = 300 if isinstance(e, EntidadAmbigua) else 404
        return jsonify({
            "error": str(e),
            "candidatos": [_json_entidad(c, entidad) for c in candidatos],
        }), estado
    except servicio.EntidadDesconocida as e:
        return jsonify({"error": str(e)}), 404

    rec = servicio.recomendar(
        modelo,
        indice,
        k=_k_pedido(),
        n_coincidencias=_ajustes().n_coincidencias,
        matriz=cargado.fases,
        crudos=_crudos(cargado),
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
        indice = servicio.indice_por_id(cargado.modelo, entity_id)
    except servicio.EntidadDesconocida as e:
        return jsonify({"error": str(e)}), 404

    detalle = servicio.ficha(
        cargado.modelo,
        indice,
        matriz=cargado.fases,
        n_rasgos=_ajustes().n_rasgos_ficha,
        crudos=_crudos(cargado),
    )
    salida = {
        "modelo": _json_modelo(clave),
        "entidad": _json_entidad(detalle.entidad, entidad),
        "perfil": _json_perfil(detalle.perfil),
        "destacados": [_json_rasgo(r) for r in detalle.destacados],
        "flojos": [_json_rasgo(r) for r in detalle.flojos],
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
    if _catalogo().disponibles():
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
    entidades = catalogo.entidades_disponibles()
    return {
        "entidad": entidades[0] if entidades else config.ENTIDADES[0],
        "entidades": entidades,
        "variantes": [],
        "variante": None,
        "conjunto": None,
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

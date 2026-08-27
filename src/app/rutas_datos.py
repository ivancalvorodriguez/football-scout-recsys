"""Vistas de la sección «Datos»: añadir partidos y reentrenar desde la interfaz.

Es la cara web de `src/incremental/`. No reimplementa nada: valida el formulario,
compone los argumentos y le pide a `tareas.GestorTareas` que ejecute el mismo
comando que documenta `docs/incremental.md`. Lo que la página enseña de la tarea
es su estado —barra, paso, tiempo y, si falla, la razón—, no su consola: el
comando y el log completo se quedan en el servidor (ver `tareas.Tarea.como_dict`).

Rutas:

- ``GET  /datos``             estado de la BD y de los modelos, y los formularios.
- ``POST /datos/validar``     comprueba un paquete sin escribir nada.
- ``POST /datos/ingerir``     incorpora los partidos a la BD.
- ``POST /datos/reentrenar``  entrena un modelo nuevo con nombre (warm start).
- ``POST /datos/cancelar``    mata la tarea en curso (solo su dueño).
- ``POST /datos/borrar-modelo``    borra un modelo propio.
- ``POST /datos/borrar-conjunto``  borra un conjunto de datos propio.
- ``GET  /datos/tarea``       estado de la tarea en curso (JSON, para la página).
- ``GET  /datos/carpetas``    listado de un directorio (JSON, para el explorador).

Las acciones lanzan trabajo en segundo plano y redirigen; la página va
consultando `/datos/tarea`. Todos los POST exigen token CSRF (`src.app.csrf`).

**Cada cuenta ve y usa solo lo suyo**, más el modelo y el conjunto de fábrica,
que son compartidos y nadie puede borrar. Eso vale para los desplegables, para el
POST que los recibe (una opción que la plantilla no ofrece tampoco se acepta al
enviarla a mano) y para el estado de la tarea.

**Sobre el explorador de carpetas** (`/datos/carpetas`): sigue sirviendo el árbol
de directorios de la máquina, y eso sigue siendo exponer la forma del sistema de
ficheros del servidor. Lo que ha cambiado es que ahora hay que estar autenticado
para llegar: la exposición **está mitigada por la autenticación, no ha
desaparecido**. En un servidor compartido —o si algún día se permite el
auto-registro— vuelve a ser un problema y habría que acotarlo a una raíz
permitida. Se deja constancia aquí a propósito, para que no se lea como resuelto.
"""

from __future__ import annotations

import os
import sqlite3
import string
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from . import auth, cobertura, config, fuentes
from .catalogo import Catalogo, CuotaSuperada, ModeloNoDisponible
from .conjuntos import CatalogoDatos, ConjuntoNoDisponible
from .conjuntos import CuotaSuperada as CuotaDatosSuperada
from .nombres import NombreInvalido
from .tareas import GestorTareas, TareaAjena, TareaEnCurso

bp_datos = Blueprint("datos", __name__, url_prefix="/datos")

# Cuántas carpetas devuelve el explorador de una vez. Un directorio con miles de
# entradas (p. ej. `events/`) no se navega en una lista: se acota y se avisa.
MAX_CARPETAS = 500


def _gestor() -> GestorTareas:
    return current_app.extensions["tareas"]


def _catalogo() -> Catalogo:
    return current_app.extensions["catalogo"]


def _datos() -> CatalogoDatos:
    return current_app.extensions["datos"]


def _fuentes() -> fuentes.Fuentes:
    return current_app.extensions["fuentes"]


def _ajustes() -> config.Config:
    return current_app.config["APP_SIMILITUD"]


def _usuario() -> str:
    """Cuenta de la petición. Aquí nunca es None: la sección exige sesión."""
    return auth.exigir_usuario()


def _visible(ruta: Path) -> str:
    """Cómo se escribe una ruta del servidor en la interfaz.

    En local, entera: el usuario está delante de esa máquina y la ruta le sirve
    para abrir la carpeta. En producción, relativa a la raíz del repositorio (y
    si cae fuera, solo el nombre del fichero): la ruta absoluta dice dónde está
    instalada la app y bajo qué cuenta corre, que es reconocimiento gratis para
    quien se haya colado.
    """
    ruta = Path(ruta)
    if not _ajustes().produccion:
        return str(ruta)
    try:
        return str(ruta.resolve().relative_to(config.RAIZ_REPO))
    except (ValueError, OSError):
        return ruta.name


# --- Estado que se muestra ----------------------------------------------------
def resumen_bd(db_path: Path) -> dict[str, Any] | None:
    """Cuántos partidos, jugadores y equipos hay, y de qué ligas. None sin BD."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return None
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            cuenta = lambda t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: E731
            ligas = conn.execute(
                "SELECT c.competition_name, s.season_name, COUNT(*) "
                "FROM matches m "
                "JOIN competitions c ON c.competition_id = m.competition_id "
                "JOIN seasons s ON s.season_id = m.season_id "
                "GROUP BY 1, 2 ORDER BY 1, 2"
            ).fetchall()
            return {
                "ruta": db_path,
                "partidos": cuenta("matches"),
                "jugadores": cuenta("players"),
                "equipos": cuenta("teams"),
                "ligas": [
                    {"competicion": c, "temporada": t, "partidos": n} for c, t, n in ligas
                ],
            }
    except sqlite3.Error:
        return None


def resumen_datos(catalogo: CatalogoDatos, usuario: str | None) -> list[dict[str, Any]]:
    """Los conjuntos de datos de esa cuenta (y los compartidos), con su recuento.

    Los que todavía no tienen fichero salen igual (`bd` a None): puede ser una
    incorporación en curso, o el conjunto base antes de la primera extracción.
    """
    return [
        {"conjunto": conjunto, "bd": resumen_bd(conjunto.ruta)}
        for conjunto in catalogo.conjuntos(usuario)
    ]


def resumen_modelos(
    catalogo: Catalogo,
    catalogo_datos: CatalogoDatos,
    fuentes_bd: fuentes.Fuentes,
    usuario: str | None,
) -> list[dict[str, Any]]:
    """Los modelos, con sus artefactos, la fecha y la cobertura de cada uno.

    Uno por nombre (el base y los reentrenados), cada uno con la lista de sus dos
    artefactos —jugador y equipo— y cuándo se escribió cada uno: comparar esa
    fecha con la última incorporación es lo que dice si el modelo ya conoce a los
    jugadores nuevos. Los modelos a medio construir salen igual, con `entidades`
    incompleto, para que un reentrenamiento fallido no desaparezca sin más.

    La **cobertura** se mide contra la BD del conjunto de datos de CADA modelo
    (el que apunta su `variante.json`), no contra la BD base: es la única
    comparación que significa algo, porque es esa la que se usará para servirlo.
    Sin ella esta página anunciaba «2640 jugadores» al lado de un modelo que solo
    conocía a 2176, sin relacionar nunca las dos cifras.
    """
    salida = []
    for variante in catalogo.variantes(usuario):
        # El base sin artefactos no es un modelo a medias: es que no se ha
        # construido todavía. Omitirlo deja que la tarjeta diga eso mismo.
        if variante.es_base and not variante.entidades:
            continue
        conjunto = catalogo_datos.resolver(variante.datos)
        universo = fuentes_bd.universo(conjunto.ruta)
        artefactos = []
        for clave in catalogo.opciones_de_variante(variante.slug, usuario):
            npz = catalogo.ruta_artefacto(clave)
            artefactos.append({
                "clave": clave,
                "fecha": (datetime.fromtimestamp(npz.stat().st_mtime)
                          if npz.exists() else None),
                # Solo los ids, no la S: la tarjeta de cada modelo no puede
                # costar 51 MB de matriz (ver `cobertura.ids_de_artefacto`).
                "cobertura": cobertura.medir(
                    clave.entidad,
                    cobertura.ids_de_artefacto(npz),
                    universo.ids(clave.entidad),
                ),
            })
        salida.append({"variante": variante, "artefactos": artefactos})
    return salida


# Campos del formulario de «Añadir partidos» que se conservan entre envíos. Tras
# lanzar una tarea se redirige (patrón POST-Redirect-GET, para que recargar no
# vuelva a lanzarla), y sin esto el formulario se quedaría vacío justo cuando el
# flujo natural es validar y, con lo mismo, incorporar.
CAMPOS_PAQUETE = ("paquete", "competicion", "temporada", "partidos")


def _contexto_pagina(**extra: Any) -> dict[str, Any]:
    ajustes = _ajustes()
    catalogo = _catalogo()
    catalogo_datos = _datos()
    usuario = _usuario()
    enviado = {c: (request.values.get(c) or "").strip() for c in CAMPOS_PAQUETE}
    # Solo se puede partir de un modelo que tenga artefactos: el warm start lee
    # el estado del anterior, y una carpeta vacía no lo tiene. Con los conjuntos,
    # igual: hay que tener BD para poder copiarla o entrenar sobre ella.
    origenes = [v for v in catalogo.variantes(usuario) if v.entidades]
    conjuntos = catalogo_datos.disponibles(usuario)
    modelos_usados, modelos_tope = catalogo.cuota(usuario)
    datos_usados, datos_tope = catalogo_datos.cuota(usuario)
    tarea_propia = _gestor().ultima(usuario)
    return {
        "enviado": enviado,
        # Los nombres se devuelven al formulario cuando se rechazan (repetido,
        # vacío tras simplificarlo): reescribirlos sería absurdo.
        "enviado_nombre": (request.values.get("nombre") or "").strip(),
        "enviado_nombre_datos": (request.values.get("nombre_datos") or "").strip(),
        "datos": resumen_datos(catalogo_datos, usuario),
        "conjuntos": conjuntos,
        # Para rotular con qué datos se entrenó cada modelo (lo guarda por slug).
        "nombres_conjunto": {
            c.slug: c.nombre for c in catalogo_datos.conjuntos(usuario)},
        "ruta_bd": _visible(ajustes.db_path),
        "modelos": resumen_modelos(catalogo, catalogo_datos, _fuentes(), usuario),
        "origenes": origenes,
        "model_dir": _visible(ajustes.model_dir),
        "tarea": tarea_propia,
        # El servidor puede estar ocupado con la tarea de OTRA cuenta: la página
        # lo dice (para explicar el 409 que va a recibir) sin decir de quién es
        # ni qué está haciendo.
        "servidor_ocupado": _gestor().hay_alguna_en_curso() and tarea_propia is None,
        # Cuotas, para los contadores y para deshabilitar los botones con la
        # razón escrita al lado (un botón gris sin explicación es un fallo).
        "cuota_modelos": {"usados": modelos_usados, "tope": modelos_tope,
                          "hay_hueco": modelos_usados < modelos_tope},
        "cuota_datos": {"usados": datos_usados, "tope": datos_tope,
                        "hay_hueco": datos_usados < datos_tope},
        "etiqueta_normalizacion": config.ETIQUETA_NORMALIZACION,
        "etiqueta_entidad": config.ETIQUETA_ENTIDAD,
        # En producción no se propone una ruta absoluta del servidor como valor
        # de defecto del formulario.
        "paquete_defecto": "" if ajustes.produccion else config.PAQUETE_DEFECTO,
        **extra,
    }


def _pagina(error: str = "", codigo: int = 200):
    return render_template("datos.html", **_contexto_pagina(error=error)), codigo


# --- Vistas -------------------------------------------------------------------
@bp_datos.get("/")
def panel():
    """Estado de los datos y de los modelos, con los formularios de gestión."""
    return _pagina()


@bp_datos.get("/tarea")
def tarea():
    """Estado de la tarea DEL USUARIO en curso (o su última). Se sondea en bucle.

    Antes devolvía la tarea global: título, log completo y línea de comandos de
    quien la hubiera lanzado. Ahora una tarea ajena es indistinguible de que no
    haya ninguna (`null`), que es justo lo que la página sabe interpretar.

    `ocupado` dice si el servidor está haciendo algo (de quien sea) sin revelar
    qué: es lo que permite a la página explicar por qué los botones no responden
    sin filtrar el trabajo de otra cuenta.
    """
    actual = _gestor().ultima(_usuario())
    return jsonify({
        "tarea": actual.como_dict() if actual else None,
        "ocupado": _gestor().hay_alguna_en_curso(),
    })


@bp_datos.post("/cancelar")
def cancelar():
    """Mata la tarea en curso, si es de quien lo pide.

    Es la salida de emergencia del «una tarea a la vez»: sin esto, un subproceso
    colgado bloquea la sección para TODAS las cuentas hasta reiniciar el
    servidor. El plazo máximo (`config.DURACION_MAX_TAREA`) cubre el caso en que
    no haya nadie mirando; este botón, el caso en que sí.
    """
    id_tarea = (request.form.get("id") or "").strip()
    try:
        muerta = _gestor().cancelar(id_tarea, _usuario())
    except TareaAjena:
        # Misma respuesta que si no existiera: cancelar la tarea de otro no debe
        # ser distinguible de cancelar una inexistente.
        return _pagina("No hay ninguna tarea tuya en curso.", codigo=404)
    current_app.logger.info("tarea %s cancelada por %s", muerta.id, _usuario())
    return redirect(url_for("datos.panel"))


@bp_datos.get("/carpetas")
def carpetas():
    """Contenido de un directorio, para el explorador de «Añadir partidos».

    Teclear a mano la ruta de un paquete es la parte más frágil del formulario
    (una barra de más y la ingesta no arranca), y el navegador no puede dar la
    ruta absoluta de una carpeta: `<input type="file" webkitdirectory>` entrega
    los ficheros, no dónde están. Así que el explorador lo sirve el servidor, que
    es quien luego tiene que abrir esa ruta.

    Solo lista DIRECTORIOS (más la marca de si el actual parece un paquete
    StatsBomb), nunca contenido de ficheros. Aun así expone la forma del árbol de
    la máquina: es aceptable con la misma suposición que el resto de la sección
    —la app corre en local para su usuario—, y no lo sería en un servidor
    compartido.
    """
    bruto = (request.args.get("ruta") or "").strip()
    actual = Path(bruto).expanduser() if bruto else Path.home()
    try:
        actual = actual.resolve()
    except OSError:
        actual = Path.home()
    if not actual.is_dir():
        return jsonify({
            "ruta": str(actual),
            "error": f"No existe la carpeta {actual}.",
            "carpetas": [], "padre": None, "unidades": _unidades(),
            "es_paquete": False, "truncado": False,
        }), 404

    try:
        hijas = sorted(
            (h for h in actual.iterdir() if h.is_dir()),
            key=lambda h: h.name.lower(),
        )
    except OSError as exc:      # permisos, unidad extraíble sin medio, etc.
        return jsonify({
            "ruta": str(actual),
            "error": f"No se puede leer {actual}: {exc}",
            "carpetas": [], "padre": _padre(actual), "unidades": _unidades(),
            "es_paquete": False, "truncado": False,
        }), 403

    return jsonify({
        "ruta": str(actual),
        "padre": _padre(actual),
        "unidades": _unidades(),
        "carpetas": [
            {"nombre": h.name, "ruta": str(h)} for h in hijas[:MAX_CARPETAS]
        ],
        "truncado": len(hijas) > MAX_CARPETAS,
        "es_paquete": es_paquete(actual),
        "error": "",
    })


def es_paquete(carpeta: Path) -> bool:
    """¿Tiene pinta de paquete StatsBomb? (para marcarlo en el explorador).

    Es una pista para el usuario, no la validación: la de verdad la hace
    `src.incremental.paquete` al lanzar «Solo validar», que es la que sabe qué
    campos exige cada fichero.
    """
    return (
        (carpeta / "competitions.json").is_file()
        and (carpeta / "matches").is_dir()
        and (carpeta / "events").is_dir()
    )


def _padre(carpeta: Path) -> str | None:
    """Carpeta de arriba, o None si ya es la raíz de su unidad."""
    padre = carpeta.parent
    return None if padre == carpeta else str(padre)


def _unidades() -> list[str]:
    """Unidades montadas (C:\\, D:\\...). En POSIX, solo la raíz.

    Sin esto no se puede saltar de disco: subiendo desde `C:\\...` se llega a
    `C:\\` y ahí se acaba el camino.
    """
    if os.name != "nt":
        return ["/"]
    return [f"{letra}:\\" for letra in string.ascii_uppercase
            if Path(f"{letra}:\\").is_dir()]


def _argumentos_paquete() -> tuple[list[str], str]:
    """Argumentos comunes de la ingesta a partir del formulario. ('', error)."""
    bruto = (request.form.get("paquete") or "").strip()
    if not bruto:
        return [], "Indica la carpeta del paquete de partidos."
    paquete = Path(bruto)
    if not paquete.is_dir():
        return [], (
            f"No existe la carpeta {paquete}. Debe ser un directorio con "
            "competitions.json, matches/, events/ y lineups/."
        )
    args = ["--paquete", str(paquete)]
    for campo, opcion in (("competicion", "--competicion"),
                          ("temporada", "--temporada"),
                          ("partidos", "--partidos")):
        valor = (request.form.get(campo) or "").strip()
        if valor:
            args += [opcion, valor]
    return args, ""


def _conjunto_pedido(campo: str = "datos_origen"):
    """Conjunto de datos elegido en el formulario. (conjunto, error).

    Se exige que exista, que tenga BD y que sea **visible para esta cuenta**:
    acaba siendo una ruta en la línea de comandos y el origen de una copia, así
    que un slug ajeno enviado a mano sería copiar los datos de otro. Que la
    plantilla no ofrezca esa opción no basta: el POST se puede escribir a mano.

    Un conjunto de otra cuenta da el mismo mensaje que uno inexistente.
    """
    slug = (request.form.get(campo) or config.VARIANTE_BASE).strip()
    for conjunto in _datos().conjuntos(_usuario()):
        if conjunto.slug == slug:
            if not conjunto.existe:
                return None, (f"El conjunto de datos «{conjunto.nombre}» todavía "
                              "no tiene base de datos.")
            return conjunto, ""
    return None, f"No existe ningún conjunto de datos llamado {slug!r}."


@bp_datos.post("/validar")
def validar():
    """Comprueba el paquete sin tocar ninguna base de datos.

    Se valida CONTRA un conjunto (el `--db` sirve para avisar de choques de
    identidad y de partidos que ya estaban), pero no escribe nada en él.
    """
    args, error = _argumentos_paquete()
    if error:
        return _pagina(error, codigo=400)
    conjunto, error = _conjunto_pedido()
    if error:
        return _pagina(error, codigo=400)
    return _lanzar("validar", "Validando el paquete",
                   [*args, "--db", str(conjunto.ruta), "--solo-validar"],
                   recordar=True)


@bp_datos.post("/ingerir")
def ingerir():
    """Crea un conjunto de datos NUEVO con los partidos del paquete incorporados.

    Mismo trato que los modelos, y por el mismo motivo: el conjunto del que se
    parte **no se toca**. Se copia, se incorporan los partidos sobre la copia y
    esta se queda con su nombre. Así los modelos ya entrenados siguen casando
    con los datos con los que se entrenaron, y se puede comparar antes/después
    en vez de tener que rehacerlo todo.

    Como la copia ES el estado anterior, la copia de seguridad de `ingesta`
    sobraría: se lanza con `--sin-copia`.
    """
    args, error = _argumentos_paquete()
    if error:
        return _pagina(error, codigo=400)
    origen, error = _conjunto_pedido()
    if error:
        return _pagina(error, codigo=400)

    catalogo_datos = _datos()
    try:
        nuevo = catalogo_datos.crear(
            request.form.get("nombre_datos") or "",
            _usuario(),
            origen=origen.slug,
            paquete=(request.form.get("paquete") or "").strip(),
        )
    except CuotaDatosSuperada as exc:
        # 409 y no 403: no es que no tenga permiso, es que el estado actual de su
        # cuenta entra en conflicto con lo que pide. Se puede resolver borrando.
        return _pagina(str(exc), codigo=409)
    except NombreInvalido as exc:
        return _pagina(str(exc), codigo=400)
    except OSError as exc:
        return _pagina(f"No se pudo crear el conjunto de datos: {exc}", codigo=500)

    extra = ["--db", str(nuevo.ruta), "--sin-copia"]
    if request.form.get("omitir_invalidos"):
        extra.append("--omitir-invalidos")
    try:
        _gestor().lanzar("ingerir", f"Creando el conjunto «{nuevo.nombre}»",
                         [*args, *extra], usuario=_usuario())
    except TareaEnCurso as exc:
        # La copia se acaba de hacer y nadie va a ampliarla: dejarla ahí sería un
        # conjunto duplicado que además ocupa el nombre.
        catalogo_datos.descartar(nuevo.slug)
        return _pagina(str(exc), codigo=409)
    return redirect(url_for("datos.panel", **_memoria()))


@bp_datos.post("/reentrenar")
def reentrenar():
    """Entrena un modelo NUEVO, con nombre, incluyendo lo que se haya incorporado.

    Tres decisiones, todas del enunciado de la sección:

    - **Siempre las dos entidades** (`--servibles`): un modelo de la app es la
      pareja jugador+equipo, no un artefacto suelto. Reentrenar solo una dejaría
      un modelo que responde a medias.
    - **Siempre warm start**: no se ofrece el ajuste en frío. Es la opción
      correcta para lo que hace esta pantalla (añadir partidos y seguir), y
      preguntarlo obligaría al usuario a decidir sobre el interior del ajuste.
      `--frio` sigue existiendo en el CLI, que es donde se compara.
    - **El resultado no pisa al origen**: se escribe en una carpeta nueva
      (`--out`), así que el modelo del que se parte queda intacto y se puede
      volver a él desde el buscador.

    Lo que sí se pregunta es **sobre qué conjunto de datos** entrenar (cuando hay
    más de uno): es la otra mitad de la identidad del modelo, y queda apuntada en
    su `variante.json` porque después hace falta para servirlo.
    """
    catalogo = _catalogo()
    usuario = _usuario()

    # El modelo de partida se valida contra los que ESTA CUENTA ve. El
    # desplegable ya solo ofrece esos, pero el POST se puede escribir a mano y
    # partir del modelo de otro le daría una copia de su ajuste.
    origen = (request.form.get("origen") or config.VARIANTE_BASE).strip()
    if not any(v.slug == origen and v.entidades
               for v in catalogo.variantes(usuario)):
        return _pagina(f"No hay ningún modelo entrenado llamado {origen!r}.", codigo=400)
    conjunto, error = _conjunto_pedido()
    if error:
        return _pagina(error, codigo=400)

    try:
        nueva = catalogo.crear_variante(
            request.form.get("nombre") or "", usuario,
            origen=origen, datos=conjunto.slug)
    except CuotaSuperada as exc:
        return _pagina(str(exc), codigo=409)
    except NombreInvalido as exc:
        return _pagina(str(exc), codigo=400)
    except OSError as exc:
        return _pagina(f"No se pudo crear la carpeta del modelo: {exc}", codigo=500)

    args = [
        "--db", str(conjunto.ruta),
        "--modelos", str(catalogo.dir_modelo(origen)),
        "--out", str(catalogo.dir_modelo(nueva.slug)),
        "--servibles",
    ]
    try:
        _gestor().lanzar("reentrenar", f"Entrenando el modelo «{nueva.nombre}»",
                         args, usuario=usuario)
    except TareaEnCurso as exc:
        # La carpeta se acaba de crear y solo tiene su `variante.json`: dejarla
        # ahí sería un modelo fantasma que ya no permite reutilizar el nombre.
        catalogo.descartar_variante(nueva.slug)
        return _pagina(str(exc), codigo=409)
    return redirect(url_for("datos.panel"))


# --- Borrado ------------------------------------------------------------------
# Sin borrado, la cuota es una condena: al tercer modelo la cuenta se queda sin
# poder entrenar nunca más. Las tres reglas son las mismas en los dos casos:
# nunca el de fábrica, nunca lo de otra cuenta, y nunca con una tarea en curso
# (podría estar escribiendo justo en esa carpeta).

def _se_puede_borrar() -> str:
    """'' si ahora se puede borrar; el motivo si no."""
    if _gestor().hay_alguna_en_curso():
        return ("Hay una tarea en curso en el servidor: espera a que termine "
                "antes de borrar nada (podría estar escribiendo justo ahí).")
    return ""


@bp_datos.post("/borrar-modelo")
def borrar_modelo():
    """Borra un modelo propio, con sus artefactos, y libera cuota."""
    error = _se_puede_borrar()
    if error:
        return _pagina(error, codigo=409)
    slug = (request.form.get("slug") or "").strip()
    try:
        borrado = _catalogo().borrar_variante(slug, _usuario())
    except ModeloNoDisponible as exc:
        # El base, uno compartido o uno de otra cuenta: todos «no existe».
        return _pagina(str(exc), codigo=404)
    except OSError as exc:
        return _pagina(f"No se pudo borrar el modelo: {exc}", codigo=500)
    current_app.logger.info("modelo %r borrado por %s", borrado.slug, _usuario())
    return redirect(url_for("datos.panel"))


@bp_datos.post("/borrar-conjunto")
def borrar_conjunto():
    """Borra un conjunto de datos propio y libera cuota.

    Los modelos entrenados sobre él NO se borran: son otra cosa, con su propia
    cuota, y siguen sirviéndose. Lo que pierden es el contexto de esa BD (equipo,
    posiciones, valores reales), porque pasan a resolverse contra el conjunto
    base. La página lo advierte antes de confirmar.
    """
    error = _se_puede_borrar()
    if error:
        return _pagina(error, codigo=409)
    slug = (request.form.get("slug") or "").strip()
    try:
        borrado = _datos().borrar(slug, _usuario())
    except ConjuntoNoDisponible as exc:
        return _pagina(str(exc), codigo=404)
    except OSError as exc:
        return _pagina(f"No se pudo borrar el conjunto de datos: {exc}", codigo=500)
    current_app.logger.info("conjunto %r borrado por %s", borrado.slug, _usuario())
    return redirect(url_for("datos.panel"))


def _memoria() -> dict[str, str]:
    """Lo que se acaba de escribir en el formulario de paquete, para la URL.

    Tras lanzar una tarea se redirige (POST-Redirect-GET, para que recargar no
    vuelva a lanzarla) y sin esto el formulario se quedaría vacío justo cuando el
    flujo natural es validar y, con lo mismo, incorporar.
    """
    return {c: v for c in CAMPOS_PAQUETE
            if (v := (request.form.get(c) or "").strip())}


def _lanzar(tipo: str, titulo: str, argumentos: list[str], recordar: bool = False):
    """Arranca la tarea y redirige al panel (POST-Redirect-GET)."""
    try:
        _gestor().lanzar(tipo, titulo, argumentos, usuario=_usuario())
    except TareaEnCurso as exc:
        return _pagina(str(exc), codigo=409)
    return redirect(url_for("datos.panel", **(_memoria() if recordar else {})))

"""Factoría de la aplicación Flask.

Patrón *application factory*: `crear_app` devuelve una instancia configurada, sin
estado global de módulo. Así las pruebas pueden levantar una app apuntando a un
directorio de modelos sintético y el servidor de desarrollo (`python -m src.app`)
usa el real, sin tocar el código.

Los catálogos viven en `app.extensions` (convención de Flask para extensiones),
de donde los leen las vistas:

- `catalogo` — modelos de `outputs/modelo/`: el base y los reentrenados.
- `datos`    — conjuntos de datos: la BD base y las creadas al añadir partidos.
- `fuentes`  — lo que aporta la BD (nombres de liga, equipo y posiciones de cada
  jugador, valores reales de las métricas), **una por conjunto de datos**: cada
  modelo se sirve con los adornos de la BD sobre la que se entrenó.
- `tareas`   — ingesta y entrenamiento en segundo plano (sección «Datos»).
- `usuarios` — las cuentas (`outputs/usuarios.json`).

Todos se releen solos cuando cambian el `.npz` o la BD (comparan mtime y
tamaño), así que después de incorporar partidos y entrenar, la app sirve lo nuevo
sin reiniciarla.

Aquí se monta además lo que hace que la app pueda salir de `localhost`:

- **Sesión firmada**. La `SECRET_KEY` sale de `SCOUTING_SECRET_KEY`. En
  producción es OBLIGATORIA y no se autogenera: con varios procesos, cada uno
  inventaría una distinta y las sesiones se rechazarían entre sí (el usuario
  entraría y a la siguiente petición estaría fuera). En desarrollo sí se genera
  una al vuelo, avisando.
- **Cabeceras de seguridad** en cada respuesta, con una CSP restrictiva: esta app
  no carga NADA externo (el radar y el campo son SVG inline, el JS y el CSS son
  suyos), así que `default-src 'self'` no rompe nada y corta de raíz la
  inyección de recursos de terceros.
- **Manejador de 500** que registra la traza en el log del servidor y devuelve
  una página sin rutas, sin nombres de módulo y sin traza: el usuario no
  necesita el interior de la máquina para saber que algo ha fallado.
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from flask import Flask, render_template

from . import auth, config as config_app, csrf, tareas as mod_tareas
from .auth import Usuarios
from .catalogo import Catalogo
from .conjuntos import CatalogoDatos
from .contexto import RADIO_MARCA, texto_trayectoria
from .fuentes import Fuentes
from .rutas import bp
from .rutas_datos import bp_datos
from .tareas import GestorTareas

# Cabeceras que se añaden a TODA respuesta. Cada una cierra una clase de ataque:
#
# - `X-Content-Type-Options: nosniff` — el navegador no adivina el tipo de un
#   recurso; sin ella, un fichero servido como texto puede acabar ejecutándose
#   como script.
# - `X-Frame-Options: DENY` — nadie mete la app en un <iframe>, que es como se
#   monta un *clickjacking* sobre los botones de `/datos`.
# - `Referrer-Policy` — al salir de la app no se filtra la URL completa, que
#   lleva el nombre de la entidad buscada y el modelo usado.
# - `Content-Security-Policy` — de dónde puede cargar el navegador cada cosa.
#   `'unsafe-inline'` en `style-src` es necesario: las figuras SVG llevan el
#   estilo calculado en el atributo `style` de cada elemento. Los scripts, en
#   cambio, son todos ficheros de `/static`, así que `script-src 'self'` va sin
#   excepciones. `frame-ancestors 'none'` dobla a X-Frame-Options para los
#   navegadores que ya solo miran la CSP.
CABECERAS_SEGURIDAD = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "form-action 'self'; "
        "base-uri 'none'; "
        "object-src 'none'; "
        "frame-ancestors 'none'"
    ),
}


def _filtro_trayectoria(app: Flask):
    """Filtro `id | trayectoria`: «Sevilla, Betis (La Liga 2015/2016) · ...».

    Cruza los dos catálogos que salen de la BD: `contexto` sabe por qué equipos y
    ligas ha pasado el jugador y en qué orden, y `ligas` sabe cómo se llama cada
    liga. Se compone aquí, que es donde se cablean las extensiones, para que
    ninguno de los dos tenga que conocer al otro.

    Los dos se piden a `fuentes` en cada llamada, no al arrancar: la BD que toca
    es la del conjunto de datos del modelo que se está sirviendo, y eso solo se
    sabe dentro de la petición.

    Devuelve cadena vacía si no hay BD, y la plantilla cae entonces a las ligas
    del artefacto, que siempre están.
    """
    def trayectoria(jugador_id: int) -> str:
        fuentes: Fuentes = app.extensions["fuentes"]
        return texto_trayectoria(
            fuentes.contexto().trayectoria(jugador_id),
            fuentes.ligas().nombre,
        )

    return trayectoria


def _filtro_ligas(app: Flask):
    """Filtro `ligas | ligas`: claves `11-27` -> «La Liga 2015/2016».

    Igual que `_filtro_trayectoria`: el catálogo se resuelve por petición.
    """
    def ligas(claves) -> str:
        return app.extensions["fuentes"].ligas().texto(claves)

    return ligas


def _configurar_log(app: Flask, nivel: str | int | None) -> None:
    """Log a la consola con un nivel configurable.

    Flask deja su logger sin configurar fuera de `app.run`, así que en un
    despliegue WSGI los avisos de esta app (login fallido, tarea huérfana, 500)
    no se verían en ningún sitio. Se añade un handler propio y solo si no hay
    ninguno: si quien despliega ya ha configurado el logging, manda el suyo.
    """
    if nivel is None:
        nivel = os.environ.get("SCOUTING_LOG", "INFO")
    if isinstance(nivel, str):
        nivel = getattr(logging, nivel.strip().upper(), logging.INFO)
    app.logger.setLevel(nivel)
    if not app.logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        app.logger.addHandler(handler)


def _clave_de_firma(app: Flask, produccion: bool) -> None:
    """Fija la `SECRET_KEY` de la sesión.

    En producción tiene que venir del entorno. Autogenerarla ahí sería un fallo
    silencioso doble: cada proceso (y cada reinicio) tendría una distinta, así
    que las sesiones emitidas por uno las rechazaría el siguiente y el usuario
    viviría expulsado sin entender por qué.
    """
    clave = os.environ.get(config_app.ENV_CLAVE, "").strip()
    if clave:
        app.config["SECRET_KEY"] = clave
        return
    if produccion:
        raise RuntimeError(
            f"Falta la variable de entorno {config_app.ENV_CLAVE}. En producción "
            "es obligatoria y no se genera sola: con varios procesos o tras un "
            "reinicio, una clave distinta invalida todas las sesiones ya "
            "emitidas. Genera una con:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
            f"y expórtala como {config_app.ENV_CLAVE}."
        )
    app.config["SECRET_KEY"] = secrets.token_urlsafe(48)
    app.logger.warning(
        "%s no está definida: se ha generado una clave de sesión efímera. Vale "
        "para desarrollo; al reiniciar, todas las sesiones se cierran.",
        config_app.ENV_CLAVE,
    )


def _avisar_de_multiproceso(app: Flask) -> None:
    """Avisa si se detecta un servidor de varios PROCESOS.

    El estado de las tareas (`GestorTareas`) y todos los catálogos viven en
    memoria del proceso. Con hilos es correcto (hay locks); con procesos, no:
    «una tarea a la vez» deja de valer, `/datos/tarea` devuelve null cuando el
    sondeo cae en otro worker —y el JS corta el sondeo en cuanto ve null, así que
    la barra de progreso muere a mitad— y la matriz S se multiplica por N en
    memoria. Ver `docs/app_web.md`.
    """
    servidor = os.environ.get("SERVER_SOFTWARE", "")
    if "gunicorn" in servidor.lower() or "gunicorn" in sys.modules:
        app.logger.warning(
            "Servidor multiproceso detectado (%s). El estado de las tareas NO se "
            "comparte entre procesos: usa waitress con hilos, o gunicorn con un "
            "único worker (-w 1). Ver docs/app_web.md.",
            servidor or "gunicorn",
        )


def _avisar_de_huerfano(app: Flask, ruta_marca: Path) -> None:
    """Avisa si quedó un subproceso de una ejecución anterior todavía vivo."""
    marca = mod_tareas.huerfano(ruta_marca)
    if marca:
        app.logger.warning(
            "Hay una tarea de una ejecución anterior TODAVÍA EN MARCHA "
            "(pid %s, tipo %r, usuario %r). Reiniciar la app no mata el "
            "subproceso: si escribe en la misma base de datos que esta app, hay "
            "riesgo de corrupción. Ciérralo antes de seguir; la marca está en %s.",
            marca.get("pid"), marca.get("tipo"), marca.get("usuario"), ruta_marca,
        )


def crear_app(
    model_dir: Path | str | None = None,
    *,
    db_path: Path | str | None = None,
    ruta_usuarios: Path | str | None = None,
    ruta_marca: Path | str | None = None,
    produccion: bool = False,
    nivel_log: str | int | None = None,
    testing: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Flask:
    """Crea la app.

    `model_dir` apunta al directorio de artefactos de `src.similitud.build`
    (por defecto `outputs/modelo/`). No se exige que exista: si falta, la app
    levanta igual y cada vista explica que hay que construir los modelos, que es
    más útil que un error de arranque. `db_path` es opcional: sin ella se pierden
    los nombres de liga, el equipo del jugador, el campo de posiciones y los
    valores reales de las métricas (que caen al z-score), pero las
    recomendaciones se sirven igual.

    `produccion` endurece la instancia: exige `SCOUTING_SECRET_KEY` y deja de
    enseñar rutas absolutas del servidor y líneas de comandos en `/datos`.
    """
    app = Flask(__name__)
    ajustes = config_app.Config(
        model_dir=Path(model_dir) if model_dir is not None
        else config_app.Config.model_dir,
        db_path=Path(db_path) if db_path is not None else config_app.Config.db_path,
        ruta_usuarios=Path(ruta_usuarios) if ruta_usuarios is not None
        else config_app.Config.ruta_usuarios,
        # Se puede mover porque hay un caso real en que estorba: varias
        # instalaciones sobre el mismo repositorio compartirían la marca y se
        # avisarían de huérfanos que no son suyos.
        ruta_marca_tarea=Path(ruta_marca) if ruta_marca is not None
        else config_app.Config.ruta_marca_tarea,
        produccion=produccion,
    )
    app.config["APP_SIMILITUD"] = ajustes
    app.config["TESTING"] = testing
    # Los nombres de entidad llevan acentos y caracteres no latin-1; sin esto
    # Flask los escaparía como \uXXXX en el JSON de la API.
    app.json.ensure_ascii = False
    _configurar_log(app, nivel_log)
    _clave_de_firma(app, produccion)
    # Cookie de sesión: inaccesible desde JavaScript (HttpOnly), no viaja en
    # peticiones cruzadas de otro sitio (SameSite=Lax) y solo por HTTPS cuando se
    # sirve por HTTPS. `Secure` se ata a `produccion` porque en local se sirve
    # por HTTP y una cookie Secure sencillamente no se guardaría: nadie podría
    # entrar.
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=produccion,
    )
    if overrides:
        app.config.update(overrides)

    app.extensions["catalogo"] = Catalogo(ajustes.model_dir)
    app.extensions["datos"] = CatalogoDatos(ajustes.db_path)
    app.extensions["fuentes"] = Fuentes(ajustes.db_path)
    app.extensions["usuarios"] = Usuarios(ajustes.ruta_usuarios)
    # Las tareas se lanzan como `python -m src.incremental...` desde la raíz del
    # repo: es donde `-m` encuentra el paquete `src` y donde las rutas relativas
    # (`open-data/data`, `outputs/`) significan lo mismo que en los CLI.
    app.extensions["tareas"] = GestorTareas(
        cwd=config_app.RAIZ_REPO,
        duracion_max=ajustes.duracion_max_tarea,
        ruta_marca=ajustes.ruta_marca_tarea,
    )
    # Las plantillas escriben `entidad.ligas | ligas` o `entidad.id | equipo` en
    # vez de arrastrar los catálogos por todos los contextos.
    app.add_template_filter(_filtro_ligas(app), "ligas")
    app.add_template_filter(_filtro_trayectoria(app), "trayectoria")
    # Constante de dibujo del campo: es la misma en toda la app, no un dato de la
    # petición, así que va como global en vez de arrastrarse por cada contexto.
    app.jinja_env.globals["radio_marca"] = RADIO_MARCA
    app.jinja_env.globals["produccion"] = produccion
    # Límites del formulario de registro. Van como globales para que el `minlength`
    # del navegador y la validación del servidor salgan de la MISMA constante: si
    # se separan, el formulario rechaza (o admite) cosas que el servidor no.
    app.jinja_env.globals["min_largo_contrasena"] = config_app.MIN_LARGO_CONTRASENA
    app.jinja_env.globals["max_largo_usuario"] = config_app.MAX_LARGO_USUARIO

    # ORDEN IMPORTANTE: el CSRF se registra ANTES que la sesión. Los
    # `before_request` corren en orden de registro, y `/login` es público para
    # `auth`: si el guardia de sesión fuera primero, el POST de login se
    # colaría sin validar el token y quedaría abierto el *login CSRF*.
    csrf.registrar(app)
    auth.registrar(app)

    app.register_blueprint(bp)
    app.register_blueprint(bp_datos)
    _registrar_manejadores(app)

    if not testing:
        _avisar_de_multiproceso(app)
        _avisar_de_huerfano(app, ajustes.ruta_marca_tarea)
    return app


def _registrar_manejadores(app: Flask) -> None:
    """Cabeceras de seguridad, CSRF rechazado y errores no previstos."""

    @app.after_request
    def _cabeceras(respuesta):
        # `setdefault`: una vista que necesite relajar algo (hoy ninguna) puede
        # hacerlo sin que esto se lo pise.
        for cabecera, valor in CABECERAS_SEGURIDAD.items():
            respuesta.headers.setdefault(cabecera, valor)
        return respuesta

    @app.errorhandler(csrf.CSRFInvalido)
    def _csrf_invalido(e: csrf.CSRFInvalido):
        app.logger.warning("POST rechazado por CSRF: %s", getattr(e, "args", ("",))[0])
        from .rutas import _contexto_minimo

        return render_template("error.html", mensaje=str(e), **_contexto_minimo()), 400

    @app.errorhandler(500)
    @app.errorhandler(Exception)
    def _error_interno(e):
        """Traza al log, página neutra al navegador.

        Se registra con `exception` para que el traceback completo quede en el
        servidor —que es donde sirve de algo— y al usuario le llega un mensaje
        fijo: una traza en pantalla revela rutas absolutas, versiones y la
        estructura del proyecto.

        Las excepciones HTTP (404, 400…) se dejan pasar: ya tienen su manejador
        y su código, y tratarlas como fallos internos convertiría cada «no
        encontrado» en un 500.
        """
        from werkzeug.exceptions import HTTPException

        if isinstance(e, HTTPException):
            return e
        app.logger.exception("error no controlado sirviendo %s", _ruta_actual())
        from .rutas import _contexto_minimo

        return render_template(
            "error.html",
            mensaje="Ha ocurrido un error interno. Vuelve a intentarlo; si "
                    "persiste, revisa el log del servidor.",
            **_contexto_minimo(),
        ), 500


def _ruta_actual() -> str:
    from flask import request

    try:
        return request.path
    except RuntimeError:
        return "(sin petición)"

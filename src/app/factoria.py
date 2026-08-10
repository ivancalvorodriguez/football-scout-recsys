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

Todos se releen solos cuando cambian el `.npz` o la BD (comparan mtime y
tamaño), así que después de incorporar partidos y entrenar, la app sirve lo nuevo
sin reiniciarla.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask

from . import config as config_app
from .catalogo import Catalogo
from .conjuntos import CatalogoDatos
from .contexto import RADIO_MARCA, texto_trayectoria
from .fuentes import Fuentes
from .rutas import bp
from .rutas_datos import bp_datos
from .tareas import GestorTareas


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


def crear_app(
    model_dir: Path | str | None = None,
    *,
    db_path: Path | str | None = None,
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
    """
    app = Flask(__name__)
    ajustes = config_app.Config(
        model_dir=Path(model_dir) if model_dir is not None
        else config_app.Config.model_dir,
        db_path=Path(db_path) if db_path is not None else config_app.Config.db_path,
    )
    app.config["APP_SIMILITUD"] = ajustes
    app.config["TESTING"] = testing
    # Los nombres de entidad llevan acentos y caracteres no latin-1; sin esto
    # Flask los escaparía como \uXXXX en el JSON de la API.
    app.json.ensure_ascii = False
    if overrides:
        app.config.update(overrides)

    app.extensions["catalogo"] = Catalogo(ajustes.model_dir)
    app.extensions["datos"] = CatalogoDatos(ajustes.db_path)
    app.extensions["fuentes"] = Fuentes(ajustes.db_path)
    # Las tareas se lanzan como `python -m src.incremental...` desde la raíz del
    # repo: es donde `-m` encuentra el paquete `src` y donde las rutas relativas
    # (`open-data/data`, `outputs/`) significan lo mismo que en los CLI.
    app.extensions["tareas"] = GestorTareas(cwd=config_app.RAIZ_REPO)
    # Las plantillas escriben `entidad.ligas | ligas` o `entidad.id | equipo` en
    # vez de arrastrar los catálogos por todos los contextos.
    app.add_template_filter(_filtro_ligas(app), "ligas")
    app.add_template_filter(_filtro_trayectoria(app), "trayectoria")
    # Constante de dibujo del campo: es la misma en toda la app, no un dato de la
    # petición, así que va como global en vez de arrastrarse por cada contexto.
    app.jinja_env.globals["radio_marca"] = RADIO_MARCA
    app.register_blueprint(bp)
    app.register_blueprint(bp_datos)
    return app

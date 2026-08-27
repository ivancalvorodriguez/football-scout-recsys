"""Constantes y configuración de la app web.

Sigue el estilo del resto del proyecto (`src/extraccion/config.py`,
`src/similitud/config.py`): todo lo ajustable vive aquí, no repartido por las
vistas. Los defaults de modelo se toman de `src.similitud.config` para que la app
y los CLI miren siempre al mismo directorio de artefactos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.extraccion.config import DATA_ROOT
from src.similitud import config as config_similitud
from src.similitud import foldin

# Raíz del repositorio: `src/app/config.py` -> subir tres niveles. Desde ahí se
# lanzan las tareas de la sección «Datos», igual que se lanzarían a mano.
RAIZ_REPO = Path(__file__).resolve().parents[2]

# Carpeta que propone el formulario de «Añadir partidos». El dataset vendido con
# el proyecto ya tiene el formato de paquete que se espera, así que sirve de
# ejemplo listo para usar.
PAQUETE_DEFECTO = str(DATA_ROOT)

# --- Universo de modelos servibles -------------------------------------------
ENTIDADES: tuple[str, ...] = ("jugador", "equipo")

# Con qué (formulación, normalización) se sirve cada entidad. Lo decide
# `src.similitud.config`, que es donde vive esa elección para todo el proyecto:
# la app no ofrece las cuatro combinaciones por entidad que sabe construir
# `build`, sino UNA por entidad. Lo que el usuario elige aquí es el MODELO
# (el de fábrica o uno reentrenado por él), no la formulación.
MODELO_BASE: dict[str, tuple[str, str]] = dict(config_similitud.MODELOS_SERVIBLES)

# --- Modelos y conjuntos de datos con nombre ----------------------------------
# Dos escalones con la misma forma: lo de fábrica vive en la raíz de su
# directorio y lo que crea el usuario, en una subcarpeta con su nombre.
#
# - Modelos (`catalogo`): `outputs/modelo/` y `outputs/modelo/variantes/<slug>/`.
# - Conjuntos de datos (`conjuntos`): `outputs/db/scouting.db` y
#   `outputs/db/conjuntos/<slug>/scouting.db`.
#
# El slug `base` y la etiqueta valen para los dos (no se cruzan: cada uno se
# resuelve dentro de su catálogo).
VARIANTE_BASE = "base"
NOMBRE_BASE = "Base"
SUBDIR_VARIANTES = "variantes"
FICHERO_VARIANTE = "variante.json"
SUBDIR_CONJUNTOS = "conjuntos"
FICHERO_CONJUNTO = "conjunto.json"
MAX_LARGO_NOMBRE = 60

# Etiquetas para la interfaz (el resto del proyecto habla en español).
ETIQUETA_ENTIDAD = {"jugador": "Jugador", "equipo": "Equipo"}
ETIQUETA_PLURAL = {"jugador": "jugadores", "equipo": "equipos"}
ETIQUETA_FORMULACION = {
    "2": "Formulación 2 — SLIM instancia-instancia",
    "5": "Formulación 5 — distribucional (MMD/Sinkhorn) + EASE",
}
ETIQUETA_NORMALIZACION = {
    "por_liga": "z-score por liga",
    "global": "z-score global",
}

# Contra qué se mide el z-score que enseña la interfaz. DEPENDE de la
# normalización del modelo servido: con `por_liga` la media es la de su
# (competition_id, season_id) y con `global` la de todo el dataset, mezclando
# ligas (ver `src.similitud.features.construir`). Escribir «su liga» siempre
# haría que la interfaz mintiese en la mitad de los modelos, así que el texto
# sale de aquí. Redactado para encajar detrás de «frente a», «respecto a» o
# «se separa de».
REFERENCIA_Z = {
    "por_liga": "la media de su liga y temporada",
    "global": "la media global (todas las ligas)",
}
# Sin modelo elegido (páginas de error) no se sabe cuál de las dos es: se dice
# lo único cierto en ambos casos.
REFERENCIA_Z_DEFECTO = "la media de referencia del modelo"

# --- Entidades proyectadas (están en la BD y no en el modelo) -----------------
# Qué se le dice al usuario cuando la respuesta no sale de la matriz S sino de
# colocar a la entidad en el modelo (`src.similitud.foldin`). El texto vive aquí
# —y las cifras, en `foldin`— porque son dos calidades de respuesta distintas y
# la interfaz no puede limitarse a un adjetivo.
#
# Los dos textos dicen QUÉ se ha hecho y por qué es otra clase de respuesta, y
# ahí se paran. La frase final que llevaban —la cifra de solapamiento con el
# top-10 del modelo en el caso fiel, el «léelo como una orientación» en el
# aproximado— pedía al lector calibrar un ranking que todavía no ha leído; las
# cifras siguen documentadas en `src.similitud.foldin` y en `docs/app_web.md`.
TEXTO_PROYECCION: dict[str, str] = {
    foldin.FIEL: (
        "no está en el modelo: está en la base de datos elegida y se le ha "
        "colocado en la geometría que el modelo ya había aprendido, sin "
        "reentrenar nada."
    ),
    foldin.APROXIMADA: (
        "no está en el modelo y se le ha colocado en él al vuelo. En esta "
        "formulación la similitud se agrega por los dos lados y de una entidad "
        "nueva solo se puede resolver uno, así que es media magnitud."
    ),
}

# --- Cuentas de usuario -------------------------------------------------------
# El fichero de usuarios vive junto al resto de la salida. Lo escriben tanto el
# formulario de registro como el CLI `src.app.usuarios`.
RUTA_USUARIOS = RAIZ_REPO / "outputs" / "usuarios.json"

# Variables de entorno de las que se lee la configuración sensible. No hay flags
# equivalentes a propósito: una clave de firma en la línea de comandos acaba en
# el historial del shell y en la lista de procesos.
ENV_CLAVE = "SCOUTING_SECRET_KEY"
ENV_MODELO = "SCOUTING_MODELO"
ENV_BD = "SCOUTING_BD"
ENV_USUARIOS = "SCOUTING_USUARIOS"

# Qué exige sesión, por prefijo de ruta. La lista es de lo PRIVADO y no de lo
# público porque así se lee la decisión de producto: **el recomendador es
# público** —buscar, comparar, fichas, glosario y API se consultan sin cuenta— y
# lo que hace falta cuenta es GESTIONAR (crear conjuntos de datos y entrenar
# modelos), que lanza subprocesos y escribe en disco del servidor.
#
# Tiene una contrapartida que conviene tener presente: con una lista de lo
# privado, una ruta nueva nace PÚBLICA salvo que se añada aquí. Al revés (lista
# de lo público) nacería cerrada, que es el default más seguro, pero obligaría a
# apuntar aquí cada página del recomendador y la primera que se olvidase dejaría
# de servirse a los visitantes. Con este reparto, quien añada una vista que
# escriba algo tiene que acordarse de colgarla de `/datos`.
PREFIJOS_PRIVADOS: tuple[str, ...] = ("/datos",)

# Coste del PBKDF2. 600.000 iteraciones es la recomendación de OWASP para
# PBKDF2-HMAC-SHA256 (2023); se guarda en cada usuario para poder subirlo después
# sin invalidar las contraseñas ya registradas.
PBKDF2_ITERACIONES = 600_000
PBKDF2_SAL_BYTES = 16

# Longitud mínima de contraseña, compartida por el formulario de registro y el
# CLI. No se imponen reglas de composición (mayúsculas, signos): alargan poco la
# entropía real y empujan a patrones predecibles. La longitud es lo que cuenta.
MIN_LARGO_CONTRASENA = 8
# Tope del nombre de cuenta. Acaba en una carpeta de metadatos y en la cabecera.
MAX_LARGO_USUARIO = 32

# --- Cuotas por usuario -------------------------------------------------------
# Cada modelo de jugador ocupa decenas de MB y cada conjunto de datos es una copia
# entera de la BD: sin tope, unas cuantas cuentas llenan el disco. Los topes
# cuentan SOLO lo propio; el modelo y el conjunto de fábrica son compartidos y no
# le cuentan a nadie.
MAX_MODELOS_POR_USUARIO = 2
MAX_DATASETS_POR_USUARIO = 3

# --- Tareas en segundo plano --------------------------------------------------
# Tope de duración de una tarea. La cola es global (una a la vez para todo el
# servidor), así que un subproceso colgado deja `/datos` inservible para TODAS
# las cuentas hasta reiniciar: pasado este tiempo se mata y se cierra como
# fallida. Generoso a propósito — un reentrenamiento completo sobre la BD entera
# ronda la media hora.
DURACION_MAX_TAREA = 4 * 60 * 60.0     # segundos

# Marca en disco de la tarea en curso. El hilo que la vigila es daemon y muere
# con la app, pero el `Popen` es un proceso APARTE que en Windows no se entera de
# que su padre se ha ido: sin esta marca, reiniciar el servidor deja un huérfano
# escribiendo en la misma BD que la app va a abrir.
RUTA_MARCA_TAREA = RAIZ_REPO / "outputs" / "tarea_en_curso.json"

# --- Parámetros de consulta ---------------------------------------------------
TOP_K_DEFECTO = config_similitud.DEFAULT_TOP_K   # 10
TOP_K_MAX = 50
# Features que más explican cada recomendación (0 las desactiva).
N_COINCIDENCIAS = 4
# Métricas destacadas y flojas que muestra la ficha de cada entidad, por lista.
N_RASGOS_FICHA = 4
# Posiciones con reparto de minutos que se listan junto al campo de la ficha.
N_POSICIONES_FICHA = 5
# Sugerencias devueltas por el autocompletado.
MAX_SUGERENCIAS = 12
# Candidatos listados cuando un nombre es ambiguo.
MAX_CANDIDATOS_AMBIGUOS = 25


@dataclass(frozen=True)
class Config:
    """Configuración de una instancia de la app.

    `model_dir` es lo único imprescindible: los artefactos servibles ya traen
    nombres y ligas de cada entidad. `db_path` es opcional y se usa (en lectura)
    para tres adornos: traducir la clave de liga `"11-27"` a `"La Liga
    2015/2016"`, poner el equipo junto al nombre del jugador y dibujar su reparto
    de minutos por posición. Si la BD no está, la app sirve igual sin esos
    bloques.
    """

    model_dir: Path = config_similitud.DEFAULT_MODEL_DIR
    db_path: Path = config_similitud.DEFAULT_DB_PATH
    ruta_usuarios: Path = RUTA_USUARIOS
    # En producción la interfaz deja de enseñar rutas absolutas del servidor:
    # son útiles en local (el usuario está delante de esa máquina) y son
    # reconocimiento del sistema en cualquier otro sitio.
    produccion: bool = False
    duracion_max_tarea: float = DURACION_MAX_TAREA
    ruta_marca_tarea: Path = RUTA_MARCA_TAREA
    max_modelos: int = MAX_MODELOS_POR_USUARIO
    max_datasets: int = MAX_DATASETS_POR_USUARIO
    top_k_defecto: int = TOP_K_DEFECTO
    top_k_max: int = TOP_K_MAX
    n_coincidencias: int = N_COINCIDENCIAS
    n_rasgos_ficha: int = N_RASGOS_FICHA
    max_sugerencias: int = MAX_SUGERENCIAS

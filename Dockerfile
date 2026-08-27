# syntax=docker/dockerfile:1.7
#
# Imagen de la app web del TFG (recomendador de scouting por similitud).
#
# Contiene SÓLO lo que la app necesita para servir: el código y
# `requirements-app.txt` (numpy, pandas, matplotlib, flask, waitress). Los
# módulos del pipeline no tienen servicio propio ni imagen propia; se siguen
# ejecutando en el anfitrión con el venv de siempre, como documenta CLAUDE.md.
#
# Lo que sí va dentro es el paquete `src/` ENTERO, y eso no es de más:
# `src/app/tareas.py` lanza `python -m src.incremental.ingesta` y
# `... .reentrenar` como SUBPROCESO con `sys.executable`, es decir, dentro de
# este mismo intérprete. Sin el resto de `src/`, la sección `/datos` de la app
# —añadir partidos y entrenar modelos desde el navegador— fallaría al lanzar.
# Es la misma razón por la que `requirements-app.txt` incluye `requirements.txt`
# con `-r`: la app NECESITA el runtime del pipeline, no lo arrastra por comodidad.

ARG PYTHON_VERSION=3.11

# UID/GID del usuario sin privilegios. Son argumentos porque con un bind mount en
# Linux los ficheros de `outputs/` los escribe ESTE uid: si no coincide con el del
# anfitrión, la app no puede escribir en su propio volumen. En Windows y macOS da
# igual (Docker Desktop traduce los permisos), de ahí el 1000 por defecto, que es
# el uid del primer usuario en casi cualquier Linux.
ARG UID=1000
ARG GID=1000


FROM python:${PYTHON_VERSION}-slim
ARG UID
ARG GID

# PYTHONUNBUFFERED: el progreso de las tareas de `/datos` se deduce leyendo la
#   salida del subproceso línea a línea; con el buffer que Python usa cuando no
#   escribe a una consola, la barra daría saltos de minutos.
# PYTHONIOENCODING/LANG: los nombres de jugador llevan caracteres fuera de
#   latin-1 ('ğ', 'ø').
# MPLBACKEND: matplotlib sin display. Aquí no hay ninguno, y así el import no
#   sale a buscar un backend interactivo.
# MPLCONFIGDIR: matplotlib necesita un directorio de configuración ESCRIBIBLE o
#   avisa en cada arranque; el usuario del contenedor no tiene *home*.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

WORKDIR /app

# El usuario se crea sólo si ese uid/gid no existe ya en la imagen base: con
# `--build-arg UID=0`, o con un uid que Debian ya use, `useradd` fallaría y
# rompería el build por algo que no es un error.
#
# Los puntos de montaje se crean y se asignan AQUÍ, antes de que monte nada: si
# el directorio no existe, Docker lo crea al vuelo como root y el proceso sin
# privilegios no puede escribir dentro de su propio volumen.
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" scouting; fi; \
    if ! getent passwd "${UID}" >/dev/null; then \
        useradd --uid "${UID}" --gid "${GID}" --no-create-home scouting; \
    fi; \
    mkdir -p /app/outputs /app/open-data /tmp/matplotlib; \
    chown -R "${UID}:${GID}" /app /tmp/matplotlib

# Los requirements van antes que el código: cambiar una línea de `src/` no debe
# invalidar la capa de pip, que es la que tarda. Los dos se copian porque
# `requirements-app.txt` incluye al otro con `-r` y pip lo va a leer.
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY --chown=${UID}:${GID} src ./src
# Los dos puntos de entrada van en la MISMA imagen: `app` y `worker` son el mismo
# codigo con distinto arranque, y construir dos imagenes para eso duplicaria 519
# MB y abriria la puerta a que se separaran de version.
COPY docker/entrada-app.sh docker/entrada-worker.sh docker/salud.py /usr/local/bin/
RUN chmod +x /usr/local/bin/entrada-app.sh /usr/local/bin/entrada-worker.sh

# Puerto FIJO dentro del contenedor. Publicarlo en otro es cosa del compose
# (`ports: "8080:8000"`); atarlo aquí es lo que permite que el HEALTHCHECK sepa
# a dónde llamar sin leer la configuración del usuario.
EXPOSE 8000

# Arrancar no es estar sirviendo: la primera petición carga la matriz S (51 MB).
# `start-period` cubre ese arranque sin contar los fallos como reintentos.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD ["python", "/usr/local/bin/salud.py"]

USER ${UID}:${GID}
ENTRYPOINT ["/usr/local/bin/entrada-app.sh"]

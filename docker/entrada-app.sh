#!/bin/sh
#
# Arranque de la app web dentro del contenedor.
#
# Traduce variables de entorno —lo único que un compose, un systemd o un
# orquestador saben fijar— a las flags de `python -m src.app`, y resuelve las dos
# cosas que un contenedor tiene que resolver y la línea de comandos no: que los
# directorios de salida existan y que la clave de firma de la sesión sea la MISMA
# entre reinicios.
#
# Cualquier argumento que se le pase se añade al final, así que
# `docker compose run --rm app --log DEBUG` sigue funcionando.

set -eu

MODO="${SCOUTING_MODO:-local}"
PUERTO=8000                       # fijo dentro del contenedor (ver Dockerfile)
HILOS="${SCOUTING_HILOS:-8}"
MODELO="${SCOUTING_MODELO:-outputs/modelo}"
BD="${SCOUTING_BD:-outputs/db/scouting.db}"
USUARIOS="${SCOUTING_USUARIOS:-outputs/usuarios.json}"
CLAVE_PERSISTIDA="outputs/.clave_sesion"

# `outputs/` puede llegar vacío (volumen nuevo). La app y los CLI dan por hechos
# estos dos, y crearlos aquí es más barato que un error a mitad de la ingesta.
mkdir -p outputs/db outputs/modelo

# --- Clave de firma de la sesión ---------------------------------------------
# `docs/app_web.md` explica por qué la app se niega a autogenerarla en
# producción: cada proceso y cada reinicio tendría una distinta, y el usuario
# viviría expulsado sin entender por qué. La objeción es a generarla EN MEMORIA.
# Aquí se genera una vez y se PERSISTE en el volumen, así que sobrevive a
# `docker compose restart` y a `up` de nuevo; que sea la misma es justamente lo
# que se pedía. Lo que venga por entorno manda siempre sobre el fichero.
if [ -z "${SCOUTING_SECRET_KEY:-}" ]; then
    if [ ! -s "$CLAVE_PERSISTIDA" ]; then
        python -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(48))' \
            > "$CLAVE_PERSISTIDA"
        chmod 600 "$CLAVE_PERSISTIDA"
        echo "AVISO: no había SCOUTING_SECRET_KEY; se ha generado una y se ha" >&2
        echo "       guardado en $CLAVE_PERSISTIDA (dentro del volumen outputs/)." >&2
        echo "       Para un despliegue real, pásala por entorno y borra ese fichero." >&2
    fi
    SCOUTING_SECRET_KEY="$(cat "$CLAVE_PERSISTIDA")"
    export SCOUTING_SECRET_KEY
fi

# --- Modo ---------------------------------------------------------------------
# `local`      servidor de desarrollo de Flask, por HTTP. Es el modo utilizable
#              tal cual en http://localhost:PUERTO: la cookie de sesión NO va
#              marcada `Secure`, así que se puede iniciar sesión y entrar en
#              `/datos` sin montar TLS.
# `produccion` waitress (servidor WSGI de verdad, un proceso y varios hilos) y
#              configuración endurecida. La cookie va marcada `Secure`, de modo
#              que **por HTTP plano nadie puede iniciar sesión**: es intencionado
#              y exige un proxy inverso que termine TLS delante. Eso es el
#              servicio `proxy` del compose (perfil `https`):
#
#                  docker compose --profile https up -d
set -- --host 0.0.0.0 --puerto "$PUERTO" \
       --modelo "$MODELO" --bd "$BD" --usuarios "$USUARIOS" "$@"

# --- Cola de tareas -----------------------------------------------------------
# Con ella, la app ENCOLA las tareas largas de `/datos` y las ejecuta el
# contenedor `worker`; sin ella, las ejecuta este mismo proceso, que es el
# comportamiento de siempre. `python -m src.app` ya la lee del entorno, pero se
# pasa explícita para que el arranque diga en su primera línea quién ejecuta.
if [ -n "${SCOUTING_COLA:-}" ]; then
    set -- --cola "$SCOUTING_COLA" "$@"
fi

case "$MODO" in
    produccion)
        set -- --produccion --hilos "$HILOS" "$@"
        ;;
    local)
        ;;
    *)
        echo "Error: SCOUTING_MODO='$MODO' no es válido (local | produccion)." >&2
        exit 2
        ;;
esac

if [ -n "${SCOUTING_LOG:-}" ]; then
    set -- --log "$SCOUTING_LOG" "$@"
fi

exec python -m src.app "$@"

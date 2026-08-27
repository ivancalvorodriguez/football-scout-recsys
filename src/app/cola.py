"""Cola de tareas en redis: lo que permite que `app` y `worker` sean dos contenedores.

`tareas.GestorTareas` ejecuta las tareas largas **en el mismo proceso** que sirve
las páginas, y su estado (qué corre, de quién es, por dónde va) vive en memoria.
Eso es correcto mientras la app sea un proceso, y es lo que se sigue usando en
desarrollo y en el contenedor `app` cuando no hay cola configurada.

Este módulo es la otra mitad: el mismo contrato público, pero con el estado en
redis y la ejecución en otro contenedor (`src.app.worker`). La app deja de
ejecutar y pasa a **encolar**; el worker consume.

Qué se gana con la separación:

- Un reentrenamiento de media hora deja de competir por la memoria y los ciclos
  del proceso que sirve las páginas.
- Reiniciar la app (un despliegue, un fallo) deja de matar la tarea en curso: el
  worker es otro contenedor y sigue a lo suyo.

Qué NO cambia, y conviene tener claro:

- **Sigue siendo una tarea a la vez.** La garantía no la da el cerrojo, la da que
  hay UN worker consumiendo en serie; el cerrojo es lo que permite a la interfaz
  responder 409 en el momento en vez de aceptar y encolar en silencio.
- **Los artefactos y las bases de datos siguen viajando por el disco compartido**
  (`outputs/`), no por redis. En redis sólo hay estado de tareas: qué se está
  ejecutando, por dónde va y si alguien ha pedido pararla. Nada que doliera
  perder — si redis se vacía, se pierde el historial de tareas, no un modelo.
- **La app no necesita invalidar nada** cuando el worker termina de entrenar:
  `catalogo.todas_las_variantes()` relee el directorio en cada llamada y los
  artefactos cargados se revalidan por `(mtime, tamaño)` del `.npz`. El bind
  mount compartido es suficiente.

Layout de claves (todas bajo `scouting:`):

| Clave | Tipo | Para qué |
| --- | --- | --- |
| `scouting:cola` | LIST | trabajos pendientes; el worker hace `BRPOP` |
| `scouting:actual` | STRING | id de la tarea en curso. Es el cerrojo, con TTL |
| `scouting:tarea:<id>` | STRING (JSON) | estado de una tarea, para el sondeo |
| `scouting:historial:<cuenta>` | LIST | últimos ids de esa cuenta |
| `scouting:cancelar:<id>` | STRING | alguien ha pedido parar esa tarea |

El cerrojo lleva TTL y el worker lo refresca mientras trabaja (`al_latir` de
`tareas.Ejecutor`). Sin TTL, un worker que muriese a mitad dejaría el servidor
marcado como ocupado para siempre y `/datos` inservible para todas las cuentas —
que es exactamente el fallo que el plazo máximo vino a resolver dentro del
proceso.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from .tareas import MODULOS, OCUPADO_AJENO, Tarea, TareaAjena, TareaEnCurso

# --- Layout de claves ---------------------------------------------------------
PREFIJO = "scouting:"
CLAVE_COLA = PREFIJO + "cola"
CLAVE_CERROJO = PREFIJO + "actual"
PREFIJO_TAREA = PREFIJO + "tarea:"
PREFIJO_HISTORIAL = PREFIJO + "historial:"
PREFIJO_CANCELAR = PREFIJO + "cancelar:"

# Vida del cerrojo. Tiene que ser holgadamente mayor que el latido del worker
# (`tareas.INTERVALO_VIGILANCIA`, 1 s) para que una pausa del GC o un pico de
# carga no lo dejen caducar con la tarea todavía viva.
TTL_CERROJO = 60
# Vida del estado de una tarea ya terminada. Es lo que sostiene el «¿cómo fue lo
# último que lancé?» al volver a `/datos`; pasado ese plazo, se olvida.
TTL_TAREA = 24 * 3600
# Cuánto espera el worker en cada `BRPOP` antes de volver a mirar si le han
# pedido parar. No es una espera activa: si llega un trabajo, vuelve al instante.
ESPERA_COLA = 5

# Cuenta anónima. Sólo aparece en despliegues sin cuentas (`/datos` exige sesión,
# así que en la práctica siempre hay usuario), pero la clave tiene que existir.
_ANONIMO = "_anonimo"


class ColaNoDisponible(TareaEnCurso):
    """No se ha podido hablar con redis.

    Hereda de `TareaEnCurso` a propósito: para quien está delante, «la cola no
    responde» y «hay algo en curso» son el mismo hecho —no se puede lanzar ahora—
    y `rutas_datos._lanzar` ya traduce esa excepción a un 409 con el mensaje. Un
    tipo nuevo habría obligado a tocar las tres vistas que lanzan tareas para
    acabar respondiendo lo mismo.
    """


def _clave_historial(usuario: str | None) -> str:
    return PREFIJO_HISTORIAL + (usuario or _ANONIMO)


class Cola:
    """Acceso a redis con el layout de claves de arriba.

    La comparten la app (`GestorRedis`) y el worker: si las claves se escribieran
    en los dos sitios, separarlas sería cuestión de tiempo.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._cliente = None

    @property
    def cliente(self):
        """Cliente de redis, creado a la primera.

        Perezoso porque `crear_app` no debe fallar por que redis todavía no esté
        en pie: en un compose los contenedores arrancan a la vez, y una app que
        se niega a levantar porque su cola tarda dos segundos es una app que no
        se puede orquestar. `depends_on` con `service_healthy` cubre el caso
        normal; esto cubre el resto.
        """
        if self._cliente is None:
            try:
                import redis                       # noqa: PLC0415 - opcional
            except ImportError as exc:             # pragma: no cover
                raise ColaNoDisponible(
                    "La cola de tareas está configurada, pero el paquete `redis` "
                    "no está instalado (pip install -r requirements-app.txt)."
                ) from exc
            # `decode_responses`: todo lo que se guarda es JSON o un id, así que
            # trabajar con str y no con bytes evita decodificar en cada lectura.
            # `socket_timeout` TIENE que ser mayor que `ESPERA_COLA`: el worker
            # se queda en un `BRPOP` bloqueante de `ESPERA_COLA` segundos y, si
            # el socket vence antes, cada espera vacia se convierte en un error
            # de conexion y el worker no llega a consumir nada.
            self._cliente = redis.Redis.from_url(
                self.url, decode_responses=True,
                socket_timeout=ESPERA_COLA + 5, socket_connect_timeout=5,
                health_check_interval=30,
            )
        return self._cliente

    # --- Estado de una tarea --------------------------------------------------
    def guardar(self, tarea: Tarea) -> None:
        self.cliente.setex(
            PREFIJO_TAREA + tarea.id, TTL_TAREA,
            json.dumps(tarea.a_dict(), ensure_ascii=False),
        )

    def leer(self, id_tarea: str) -> Tarea | None:
        if not id_tarea:
            return None
        crudo = self.cliente.get(PREFIJO_TAREA + id_tarea)
        if not crudo:
            return None
        try:
            datos = json.loads(crudo)
        except json.JSONDecodeError:
            return None
        return Tarea.desde_dict(datos) if isinstance(datos, dict) else None

    # --- Cerrojo --------------------------------------------------------------
    def tomar_cerrojo(self, id_tarea: str) -> bool:
        """Marca el servidor como ocupado. False si ya lo estaba."""
        return bool(self.cliente.set(
            CLAVE_CERROJO, id_tarea, nx=True, ex=TTL_CERROJO))

    def poner_cerrojo(self, id_tarea: str) -> None:
        """Toma el cerrojo SIN condición.

        Lo usa el worker al empezar a trabajar: si el cerrojo caducó mientras el
        trabajo esperaba en la cola, quien manda es que la tarea está corriendo
        de verdad, no lo que quedara en la clave.
        """
        self.cliente.set(CLAVE_CERROJO, id_tarea, ex=TTL_CERROJO)

    def refrescar_cerrojo(self, id_tarea: str) -> None:
        """Alarga el cerrojo, y lo repone si se ha perdido.

        Reponerlo importa: si redis se reinicia con una tarea en marcha, la clave
        desaparece y la interfaz declararia el servidor libre mientras el worker
        sigue escribiendo en la BD. El siguiente latido lo vuelve a poner.
        """
        actual = self.cliente.get(CLAVE_CERROJO)
        if actual == id_tarea:
            self.cliente.expire(CLAVE_CERROJO, TTL_CERROJO)
        elif actual is None:
            self.cliente.set(CLAVE_CERROJO, id_tarea, ex=TTL_CERROJO)

    def soltar_cerrojo(self, id_tarea: str) -> None:
        """Libera el cerrojo si sigue siendo nuestro.

        La comprobación y el borrado no son atómicos. Se asume: para que
        importara, el cerrojo tendría que haber caducado justo entre las dos
        instrucciones y otra tarea haberlo tomado en esa rendija, con UN worker
        que además está terminando la única tarea que había.
        """
        if self.cliente.get(CLAVE_CERROJO) == id_tarea:
            self.cliente.delete(CLAVE_CERROJO)

    def id_en_curso(self) -> str | None:
        return self.cliente.get(CLAVE_CERROJO) or None

    # --- Trabajos -------------------------------------------------------------
    def encolar(self, trabajo: dict) -> None:
        self.cliente.lpush(CLAVE_COLA, json.dumps(trabajo, ensure_ascii=False))

    def desencolar(self, espera: int = ESPERA_COLA) -> dict | None:
        """Siguiente trabajo, esperando hasta `espera` segundos. None si no hay."""
        salida = self.cliente.brpop(CLAVE_COLA, timeout=espera)
        if not salida:
            return None
        try:
            trabajo = json.loads(salida[1])
        except json.JSONDecodeError:
            return None
        return trabajo if isinstance(trabajo, dict) else None

    # --- Historial ------------------------------------------------------------
    def anotar(self, usuario: str | None, id_tarea: str, tope: int) -> None:
        clave = _clave_historial(usuario)
        tubo = self.cliente.pipeline()
        tubo.lpush(clave, id_tarea)
        tubo.ltrim(clave, 0, tope - 1)
        tubo.expire(clave, TTL_TAREA)
        tubo.execute()

    def historial(self, usuario: str | None, tope: int) -> list[str]:
        return list(self.cliente.lrange(_clave_historial(usuario), 0, tope - 1))

    # --- Cancelación ----------------------------------------------------------
    def pedir_cancelacion(self, id_tarea: str) -> None:
        self.cliente.setex(PREFIJO_CANCELAR + id_tarea, TTL_CERROJO, "1")

    def cancelacion_pedida(self, id_tarea: str) -> bool:
        return bool(self.cliente.exists(PREFIJO_CANCELAR + id_tarea))

    def olvidar_cancelacion(self, id_tarea: str) -> None:
        self.cliente.delete(PREFIJO_CANCELAR + id_tarea)


class GestorRedis:
    """Mismo contrato que `tareas.GestorTareas`, pero encolando en vez de ejecutar.

    Los dos son intercambiables desde `factoria.crear_app`, y `rutas_datos` no
    sabe cuál le ha tocado: llama a `lanzar`, `ultima`, `hay_alguna_en_curso` y
    `cancelar` igual en los dos casos.

    **Ante un fallo de redis, las lecturas mienten hacia el lado seguro**:
    `hay_alguna_en_curso()` devuelve True (el servidor se declara ocupado, así
    que no se lanza ni se borra nada) y `ultima()` devuelve None. Al revés
    —declararse libre cuando no se sabe— `/datos` dejaría borrar un modelo
    mientras el worker está escribiendo justo en esa carpeta.
    """

    def __init__(self, url: str, cwd: Path, max_historial: int = 8,
                 duracion_max: float | None = None,
                 ruta_marca: Path | None = None) -> None:
        self.cola = Cola(url)
        self.cwd = Path(cwd)
        self.duracion_max = duracion_max
        # La escribe el WORKER, que es quien tiene el hijo; aquí se guarda sólo
        # para que la firma sea la misma que la del gestor local y `crear_app` no
        # tenga que distinguir.
        self.ruta_marca = Path(ruta_marca) if ruta_marca is not None else None
        self._max_historial = max_historial

    # --- Consulta -------------------------------------------------------------
    def en_curso(self) -> Tarea | None:
        try:
            id_tarea = self.cola.id_en_curso()
            if not id_tarea:
                return None
            tarea = self.cola.leer(id_tarea)
        except Exception:               # noqa: BLE001 - redis caído: ver docstring
            return None
        if tarea is None or tarea.estado != "en_curso":
            return None
        return tarea

    def ultima(self, usuario: str | None) -> Tarea | None:
        """La tarea DE ESA CUENTA que está corriendo, o la última suya.

        Igual que en el gestor local: la tarea de otro es indistinguible de que
        no haya ninguna (`None`), que es lo que la página sabe interpretar.
        """
        actual = self.en_curso()
        if actual is not None and actual.es_de(usuario):
            return actual
        try:
            for id_tarea in self.cola.historial(usuario, self._max_historial):
                tarea = self.cola.leer(id_tarea)
                if tarea is not None and tarea.es_de(usuario):
                    return tarea
        except Exception:               # noqa: BLE001
            return None
        return None

    def hay_alguna_en_curso(self) -> bool:
        """¿Está el servidor ocupado? (sin decir con qué ni de quién)."""
        try:
            id_tarea = self.cola.id_en_curso()
        except Exception:               # noqa: BLE001 - fallo = ocupado
            return True
        return bool(id_tarea)

    # --- Lanzamiento ----------------------------------------------------------
    def lanzar(self, tipo: str, titulo: str, argumentos: list[str],
               usuario: str | None = None) -> Tarea:
        """Encola la tarea. La ejecuta el worker, no este proceso."""
        if tipo not in MODULOS:
            raise ValueError(f"tipo de tarea desconocido: {tipo!r}")
        tarea = Tarea(id=uuid.uuid4().hex[:12], tipo=tipo, titulo=titulo,
                      comando=[], usuario=usuario)
        try:
            if not self.cola.tomar_cerrojo(tarea.id):
                ocupa = self.en_curso()
                # El título sólo se enseña si la tarea que ocupa es de quien
                # pregunta; si no, mensaje genérico (ver `OCUPADO_AJENO`).
                if ocupa is not None and ocupa.es_de(usuario):
                    raise TareaEnCurso(
                        f"Ya hay una tarea en curso ({ocupa.titulo}). "
                        "Espera a que termine."
                    )
                raise TareaEnCurso(OCUPADO_AJENO)
            self.cola.guardar(tarea)
            self.cola.anotar(usuario, tarea.id, self._max_historial)
            # El comando NO viaja hecho: el worker lo compone con SU `sys.executable`
            # y su propia tabla `MODULOS`. Así la lista cerrada de módulos se
            # comprueba también en el lado que ejecuta, y un intérprete distinto
            # entre contenedores deja de ser un problema.
            self.cola.encolar({
                "id": tarea.id,
                "tipo": tipo,
                "argumentos": list(argumentos),
                "duracion_max": self.duracion_max,
                "tarea": tarea.a_dict(),
            })
        except TareaEnCurso:
            raise
        except Exception as exc:        # noqa: BLE001
            # Si algo falló después de tomar el cerrojo, soltarlo: si no, el
            # servidor queda ocupado por una tarea que nunca llegó a la cola.
            try:
                self.cola.soltar_cerrojo(tarea.id)
            except Exception:           # noqa: BLE001
                pass
            raise ColaNoDisponible(
                "La cola de tareas no responde, así que ahora mismo no se puede "
                f"lanzar nada. Inténtalo en un momento. ({exc})"
            ) from exc
        return tarea

    # --- Cancelación ----------------------------------------------------------
    def cancelar(self, id_tarea: str, usuario: str | None) -> Tarea:
        """Pide al worker que mate la tarea, si es de esa cuenta.

        Aquí no se mata nada: el hijo es de otro contenedor. Se deja la petición
        en redis y el worker la recoge en su siguiente latido (1 s), que para
        quien pulsa el botón es indistinguible de matarlo aquí mismo.
        """
        actual = self.en_curso()
        if actual is None or actual.id != id_tarea or not actual.es_de(usuario):
            raise TareaAjena(f"No hay ninguna tarea tuya con id {id_tarea!r}.")
        try:
            self.cola.pedir_cancelacion(id_tarea)
        except Exception as exc:        # noqa: BLE001
            raise ColaNoDisponible(
                f"No se ha podido pedir la cancelación: {exc}"
            ) from exc
        actual.motivo = "cancelada"
        return actual


def comando_de(tipo: str, argumentos: list[str]) -> list[str]:
    """`python -m <módulo de tipo> <argumentos>`, compuesto en quien ejecuta.

    Vive aquí y no en el worker porque la comprobación contra `MODULOS` —la lista
    cerrada de lo que la interfaz puede ejecutar— tiene que ser la misma en los
    dos lados de la cola.
    """
    if tipo not in MODULOS:
        raise ValueError(f"tipo de tarea desconocido: {tipo!r}")
    return [sys.executable, "-m", MODULOS[tipo], *argumentos]

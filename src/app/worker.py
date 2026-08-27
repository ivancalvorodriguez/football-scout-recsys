"""Consumidor de la cola de tareas: el contenedor `worker`.

    python -m src.app.worker --cola redis://redis:6379/0

Es la mitad que ejecuta. La app encola (`src.app.cola.GestorRedis`) y este
proceso saca trabajos de uno en uno y los corre, que es literalmente lo mismo que
hacía el hilo de `tareas.GestorTareas` cuando todo vivía en un proceso: lanza
`python -m src.incremental.ingesta` o `... .reentrenar`, lee su salida para
deducir el progreso y lo publica.

Tres decisiones:

- **Un trabajo a la vez, en serie.** No hay pool ni concurrencia interna. Las dos
  tareas escriben en sitios compartidos (la BD y el directorio de modelos) y esa
  restricción no ha cambiado por repartirlas en contenedores: lo que la garantiza
  ahora es que este bucle es secuencial. Escalar a `--scale worker=N` rompería el
  «una tarea a la vez»; el compose no lo hace y no debe hacerlo.
- **El comando se compone aquí**, con el `sys.executable` de ESTE contenedor y
  contra la tabla `tareas.MODULOS` (`cola.comando_de`). Lo que viaja por la cola
  es un tipo y una lista de argumentos, nunca una línea de comandos hecha: así la
  lista cerrada de módulos ejecutables se comprueba también en el lado que
  ejecuta, y no en el que pide.
- **Al recibir SIGTERM se mata la tarea en curso** y se la marca fallida, en vez
  de intentar terminarla. Un reentrenamiento puede durar media hora y ningún
  `stop_grace_period` razonable lo cubre; lo que queda a medias es descartable
  por construcción (la ingesta escribe sobre una COPIA de la BD y el
  reentrenamiento sobre una carpeta de modelo recién creada), así que lo correcto
  es dejar el cerrojo libre y el estado dicho, no un proceso colgado y una
  interfaz ocupada para siempre.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from . import config as config_app
from .cola import Cola, comando_de
from .tareas import (
    INTERVALO_VIGILANCIA,
    Ejecutor,
    Tarea,
    borrar_marca,
    escribir_marca,
    matar_proceso,
)

log = logging.getLogger("scouting.worker")

# Cada cuánto se publica el avance como mucho. El hijo puede emitir una línea de
# progreso por iteración; publicarlas todas sería castigar a redis para que la
# página, que sondea cada segundo, no vea ninguna diferencia.
INTERVALO_PUBLICACION = 0.5

# Espera entre reintentos cuando redis no responde, y tope de esa espera. Que la
# cola no esté todavía en pie es lo normal al levantar un compose, no un error.
ESPERA_REINTENTO = 1.0
MAX_ESPERA_REINTENTO = 30.0


class Worker:
    """Bucle que consume la cola y ejecuta lo que salga."""

    def __init__(self, url: str, cwd: Path,
                 ruta_marca: Path | None = None) -> None:
        self.cola = Cola(url)
        self.cwd = Path(cwd)
        self.ruta_marca = ruta_marca
        self._parar = threading.Event()
        # El `Ejecutor` en marcha, para poder matar a su hijo desde el manejador
        # de la señal (que corre en el hilo principal, no en este bucle).
        self._actual: Ejecutor | None = None

    # --- Ciclo de vida --------------------------------------------------------
    def parar(self, *_señal) -> None:
        """Deja de aceptar trabajos y mata el que esté corriendo."""
        if self._parar.is_set():
            return
        log.info("parada pedida: no se aceptan más trabajos")
        self._parar.set()
        ejecutor = self._actual
        if ejecutor is not None:
            ejecutor.tarea.motivo = "detenida"
            matar_proceso(ejecutor.tarea.proceso)

    def correr(self) -> int:
        """Bucle principal. Devuelve el código de salida del proceso."""
        espera = ESPERA_REINTENTO
        log.info("worker en marcha; cola=%s cwd=%s", self.cola.url, self.cwd)
        while not self._parar.is_set():
            try:
                trabajo = self.cola.desencolar()
                espera = ESPERA_REINTENTO       # hubo conexión: se reinicia
            except Exception as exc:            # noqa: BLE001 - redis caído
                # No es un error del que haya que salir: en un compose los
                # servicios arrancan a la vez y redis puede tardar. Se reintenta
                # con espera creciente para no llenar el log a un mensaje por ms.
                log.warning("la cola no responde (%s); reintento en %.0f s",
                            exc, espera)
                self._parar.wait(espera)
                espera = min(espera * 2, MAX_ESPERA_REINTENTO)
                continue
            if trabajo is None:                 # el BRPOP venció sin nada
                continue
            try:
                self._atender(trabajo)
            except Exception:                   # noqa: BLE001
                # Una tarea que revienta no puede llevarse el worker: eso dejaría
                # el cerrojo puesto y `/datos` ocupado hasta que caducara.
                log.exception("fallo atendiendo el trabajo %r", trabajo.get("id"))
        log.info("worker detenido")
        return 0

    # --- Una tarea ------------------------------------------------------------
    def _atender(self, trabajo: dict) -> None:
        tarea = Tarea.desde_dict(trabajo.get("tarea") or {})
        tipo = str(trabajo.get("tipo") or tarea.tipo)
        argumentos = [str(a) for a in (trabajo.get("argumentos") or [])]
        duracion_max = trabajo.get("duracion_max")
        try:
            tarea.comando = comando_de(tipo, argumentos)
        except ValueError:
            # Tipo desconocido: la app ya lo valida al encolar, así que esto sólo
            # puede venir de un trabajo de otra versión. Se cierra como fallida
            # en vez de descartarse en silencio: quien la lanzó está mirando.
            log.error("tipo de tarea desconocido: %r", tipo)
            self._cerrar(tarea, codigo=-1,
                         error=f"Tipo de tarea desconocido: {tipo!r}.")
            return

        log.info("tarea %s (%s) — %s", tarea.id, tipo, tarea.titulo)
        self.cola.poner_cerrojo(tarea.id)
        self.cola.olvidar_cancelacion(tarea.id)
        tarea.estado = "en_curso"
        tarea.inicio = time.time()
        self._publicar(tarea)

        ultimo = 0.0

        def al_cambiar(t: Tarea) -> None:
            nonlocal ultimo
            ahora = time.monotonic()
            if ahora - ultimo < INTERVALO_PUBLICACION:
                return
            ultimo = ahora
            self._publicar(t)

        ejecutor = Ejecutor(
            tarea,
            cwd=self.cwd,
            duracion_max=duracion_max,
            al_arrancar=lambda t, pid: escribir_marca(self.ruta_marca, t, pid),
            al_cambiar=al_cambiar,
            cancelacion_pedida=lambda: self.cola.cancelacion_pedida(tarea.id),
            al_latir=lambda: self.cola.refrescar_cerrojo(tarea.id),
        )
        self._actual = ejecutor
        try:
            codigo, nota = ejecutor.correr()
        finally:
            self._actual = None
            borrar_marca(self.ruta_marca)
        if tarea.motivo == "detenida" and not nota:
            nota = "El worker se detuvo mientras la tarea estaba en curso."
        self._cerrar(tarea, codigo=codigo, error=nota)
        log.info("tarea %s: %s (código %s, %.1f s)",
                 tarea.id, tarea.estado, tarea.codigo, tarea.segundos)

    def _cerrar(self, tarea: Tarea, codigo: int, error: str = "") -> None:
        """Marca la tarea como acabada, la publica y suelta el cerrojo.

        Es el equivalente de `GestorTareas._cerrar`, con las mismas reglas: una
        tarea matada (plazo, cancelación o parada del worker) es fallida aunque
        el sistema devolviera 0, porque no llegó a hacer lo que se le pidió.
        """
        if error:
            tarea.lineas.append(error)
        tarea.codigo = codigo
        terminada = codigo == 0 and not tarea.motivo
        tarea.estado = "terminada" if terminada else "fallida"
        tarea.fin = time.time()
        tarea.progreso = 1.0 if terminada else tarea.progreso
        tarea.proceso = None
        if not terminada and error:
            tarea.error_fijo = error
        self._publicar(tarea)
        for accion in (lambda: self.cola.olvidar_cancelacion(tarea.id),
                       lambda: self.cola.soltar_cerrojo(tarea.id)):
            try:
                accion()
            except Exception:           # noqa: BLE001
                # Si redis no responde justo al cerrar, el cerrojo caduca solo
                # por su TTL. Insistir aquí sería bloquear el bucle.
                log.warning("no se pudo limpiar el estado de %s", tarea.id)

    def _publicar(self, tarea: Tarea) -> None:
        try:
            self.cola.guardar(tarea)
        except Exception:               # noqa: BLE001
            # Perder una publicación sólo significa que la barra de progreso se
            # queda quieta un momento; la siguiente la pone al día.
            log.debug("no se pudo publicar el estado de %s", tarea.id)


def _parsear(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.app.worker",
        description="Ejecuta las tareas largas que encola la app web.",
    )
    p.add_argument("--cola", default=None,
                   help=f"URL de redis. Por defecto, ${config_app.ENV_COLA}.")
    p.add_argument("--cwd", type=Path, default=config_app.RAIZ_REPO,
                   help="Directorio desde el que se lanzan los CLI (raíz del repo).")
    p.add_argument("--marca", type=Path, default=config_app.RUTA_MARCA_TAREA,
                   help="Fichero de marca de tarea en curso.")
    p.add_argument("--log", default="INFO", help="Nivel de log (INFO por defecto).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parsear(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    url = args.cola or config_app.url_cola()
    if not url:
        # Sin cola no hay nada que consumir. Es un error de configuración y se
        # dice como tal: un worker en pie que no consume nada es peor que uno
        # que no arranca, porque la interfaz parecería sana y nada se ejecutaría.
        print(f"Error: falta la URL de la cola (--cola o ${config_app.ENV_COLA}).",
              file=sys.stderr)
        return 2

    worker = Worker(url, cwd=args.cwd, ruta_marca=args.marca)
    # SIGTERM es lo que manda `docker stop`; SIGINT, un Ctrl-C en primer plano.
    for señal in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(señal, worker.parar)
        except (ValueError, OSError):   # pragma: no cover - no hay señales aquí
            pass
    return worker.correr()


if __name__ == "__main__":              # pragma: no cover
    raise SystemExit(main())

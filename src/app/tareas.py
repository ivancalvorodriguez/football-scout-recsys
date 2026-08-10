"""Tareas largas lanzadas desde la interfaz: ingesta y reentrenamiento.

Incorporar partidos tarda decenas de segundos y reentrenar puede tardar media
hora: ninguna de las dos cabe dentro de una petición HTTP. Este módulo las
ejecuta en segundo plano y expone su estado para que la página lo vaya
consultando.

Decisiones:

- **Subproceso, no hilo llamando a la librería.** Se lanza literalmente
  ``python -m src.incremental.ingesta ...``, el mismo comando documentado en
  `docs/incremental.md`. Así la interfaz no es un segundo camino que pueda
  divergir del CLI, el trabajo pesado no compite con el servidor por el GIL, y
  un fallo grave no se lleva por delante a la app. La página enseña el comando
  ejecutado, que además es la forma de reproducirlo a mano.
- **Una tarea a la vez.** Las dos escriben en sitios compartidos (la BD y el
  directorio de modelos); dos a la vez podrían pisarse. El gestor rechaza lanzar
  una segunda mientras haya una en curso.
- **El comando no se construye con texto del usuario.** El ejecutable y el módulo
  son constantes de este fichero y los argumentos van como lista (nunca
  ``shell=True``), así que no hay forma de inyectar un comando distinto. Lo que sí
  controla el usuario es la RUTA del paquete, que se valida antes de lanzar.

Limitación asumida: el estado vive en memoria del proceso, así que sirve para el
servidor de desarrollo de un solo proceso que usa este TFG. Con varios workers
cada uno vería sus propias tareas.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Módulos que la interfaz puede ejecutar. Lista cerrada a propósito.
MODULOS = {
    "validar": "src.incremental.ingesta",
    "ingerir": "src.incremental.ingesta",
    "reentrenar": "src.incremental.reentrenar",
}

# Cuántas líneas de salida se conservan (las últimas). El reentrenamiento emite
# muchas de progreso; con esto la página no crece sin límite.
MAX_LINEAS = 400

# `    ajuste:  54%  (27313/50579)` del reentrenamiento.
_PROGRESO_PCT = re.compile(r"^\s*(\w+):\s+(\d{1,3})%\s+\((\d+)/(\d+)\)")
# `  [12/306] 3890561 Hoffenheim vs Schalke 04 — 28 jugadores (nuevo)` de la ingesta.
_PROGRESO_PASO = re.compile(r"^\s*\[(\d+)/(\d+)\]")
# `[formulacion 2 | jugador | por_liga]` del reentrenamiento.
_ETIQUETA = re.compile(r"^\[(formulacion .+)\]$")


class TareaEnCurso(RuntimeError):
    """Ya hay una tarea corriendo; no se admite una segunda."""


@dataclass
class Tarea:
    """Una ejecución en segundo plano y lo que se sabe de ella."""

    id: str
    tipo: str
    titulo: str
    comando: list[str]
    estado: str = "en_curso"          # en_curso | terminada | fallida
    codigo: int | None = None
    lineas: list[str] = field(default_factory=list)
    progreso: float | None = None     # 0..1, None si no se puede estimar
    paso: str = ""
    inicio: float = field(default_factory=time.time)
    fin: float | None = None

    @property
    def segundos(self) -> float:
        return (self.fin or time.time()) - self.inicio

    def como_dict(self) -> dict:
        return {
            "id": self.id,
            "tipo": self.tipo,
            "titulo": self.titulo,
            "comando": " ".join(self.comando),
            "estado": self.estado,
            "codigo": self.codigo,
            "lineas": list(self.lineas),
            "progreso": self.progreso,
            "paso": self.paso,
            "segundos": round(self.segundos, 1),
        }


class GestorTareas:
    """Lanza y vigila las tareas largas. Una a la vez."""

    def __init__(self, cwd: Path, max_historial: int = 8) -> None:
        self.cwd = Path(cwd)
        self._max_historial = max_historial
        self._tareas: list[Tarea] = []
        self._actual: Tarea | None = None
        self._lock = threading.Lock()

    # --- Consulta -------------------------------------------------------------
    def en_curso(self) -> Tarea | None:
        with self._lock:
            return self._actual

    def ultima(self) -> Tarea | None:
        """La que está corriendo o, si no hay ninguna, la más reciente.

        Es lo único que la página necesita: enseña una tarea, la de ahora o la
        que acaba de terminar. Se guardan unas cuantas anteriores solo para no
        perder el rastro si se encadenan varias, no para listarlas.
        """
        with self._lock:
            return self._actual or (self._tareas[-1] if self._tareas else None)

    # --- Lanzamiento ----------------------------------------------------------
    def lanzar(self, tipo: str, titulo: str, argumentos: list[str]) -> Tarea:
        """Arranca `python -m <modulo de tipo> <argumentos>` en segundo plano."""
        if tipo not in MODULOS:
            raise ValueError(f"tipo de tarea desconocido: {tipo!r}")
        comando = [sys.executable, "-m", MODULOS[tipo], *argumentos]
        tarea = Tarea(id=uuid.uuid4().hex[:12], tipo=tipo, titulo=titulo,
                      comando=comando)
        with self._lock:
            if self._actual is not None:
                raise TareaEnCurso(
                    f"Ya hay una tarea en curso ({self._actual.titulo}). "
                    "Espera a que termine."
                )
            self._actual = tarea
            self._tareas.append(tarea)
            del self._tareas[:-self._max_historial]
        threading.Thread(target=self._ejecutar, args=(tarea,), daemon=True).start()
        return tarea

    # --- Ejecución ------------------------------------------------------------
    def _ejecutar(self, tarea: Tarea) -> None:
        # PYTHONUNBUFFERED para ver el progreso en vivo (si no, el hijo bufferiza
        # al no escribir a una consola) y PYTHONIOENCODING porque los nombres de
        # jugador llevan caracteres fuera de latin-1.
        entorno = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            proceso = subprocess.Popen(
                tarea.comando,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=entorno,
            )
        except OSError as exc:
            self._cerrar(tarea, codigo=-1, ultima_linea=f"No se pudo lanzar: {exc}")
            return

        # En modo texto Python traduce `\r` como fin de línea, así que la barra de
        # progreso del hijo (que reescribe la misma línea con `\r`) llega aquí
        # como una línea por actualización, sin tratamiento especial.
        assert proceso.stdout is not None
        for linea in proceso.stdout:
            self._registrar(tarea, linea.rstrip("\n"))
        codigo = proceso.wait()
        self._cerrar(tarea, codigo=codigo)

    def _registrar(self, tarea: Tarea, linea: str) -> None:
        if not linea.strip():
            return
        with self._lock:
            progreso, paso = _leer_progreso(linea)
            if progreso is not None:
                tarea.progreso = progreso
            if paso:
                tarea.paso = paso
            # Las líneas de progreso se sustituyen entre sí en vez de acumularse:
            # el reentrenamiento emite una cada 5% por modelo y llenarían el log.
            es_progreso = _PROGRESO_PCT.match(linea) is not None
            if es_progreso and tarea.lineas and _PROGRESO_PCT.match(tarea.lineas[-1]):
                tarea.lineas[-1] = linea
            else:
                tarea.lineas.append(linea)
                del tarea.lineas[:-MAX_LINEAS]

    def _cerrar(self, tarea: Tarea, codigo: int, ultima_linea: str = "") -> None:
        with self._lock:
            if ultima_linea:
                tarea.lineas.append(ultima_linea)
            tarea.codigo = codigo
            tarea.estado = "terminada" if codigo == 0 else "fallida"
            tarea.fin = time.time()
            tarea.progreso = 1.0 if codigo == 0 else tarea.progreso
            if self._actual is tarea:
                self._actual = None


def _leer_progreso(linea: str) -> tuple[float | None, str]:
    """(fracción completada, etiqueta del paso) que se deduce de una línea."""
    m = _PROGRESO_PCT.match(linea)
    if m:
        hechos, total = int(m.group(3)), int(m.group(4))
        return (hechos / total if total else None), ""
    m = _PROGRESO_PASO.match(linea)
    if m:
        hechos, total = int(m.group(1)), int(m.group(2))
        return (hechos / total if total else None), f"partido {hechos} de {total}"
    m = _ETIQUETA.match(linea.strip())
    if m:
        return None, m.group(1)
    return None, ""

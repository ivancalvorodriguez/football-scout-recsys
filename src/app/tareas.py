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
  un fallo grave no se lleva por delante a la app.
- **La salida del hijo no se enseña.** Se lee y se conserva (`lineas`), pero solo
  para deducir de ella el progreso y, si la tarea falla, la razón en UNA línea:
  la página muestra barra, paso y tiempo, no una consola. El volcado del log y la
  línea de comandos completa eran ruido para quien usa la interfaz y, fuera de
  local, reconocimiento del sistema (rutas absolutas del servidor); por eso
  `como_dict` ya no los devuelve y no hay forma de pedirlos desde el navegador.
- **Una tarea a la vez.** Las dos escriben en sitios compartidos (la BD y el
  directorio de modelos); dos a la vez podrían pisarse. El gestor rechaza lanzar
  una segunda mientras haya una en curso.
- **El comando no se construye con texto del usuario.** El ejecutable y el módulo
  son constantes de este fichero y los argumentos van como lista (nunca
  ``shell=True``), así que no hay forma de inyectar un comando distinto. Lo que sí
  controla el usuario es la RUTA del paquete, que se valida antes de lanzar.

- **Cada tarea tiene dueño.** Con cuentas, el estado de la tarea (su título, su
  progreso, su error) es información de quien la lanzó: `ultima` solo la devuelve
  a su dueño y `cancelar` solo la acepta de él.
- **Tope de duración y cancelación.** Antes, `_ejecutar` iteraba sobre la salida
  del hijo y hacía `wait()` sin límite: un subproceso colgado dejaba `_actual`
  ocupado para siempre y `/datos` inservible para TODAS las cuentas hasta
  reiniciar el servidor. Ahora hay un plazo máximo y un botón de cancelar.
- **Marca en disco** (`outputs/tarea_en_curso.json`) mientras corre. El hilo que
  vigila es daemon y muere con la app, pero el `Popen` es un proceso aparte: en
  Windows no se entera de que su padre se fue y sigue escribiendo en el mismo
  SQLite que la app va a abrir al arrancar de nuevo. La marca es lo que permite
  detectarlo y avisar.

Limitación asumida: el estado vive en memoria del proceso, así que sirve para un
despliegue de UN proceso con varios hilos (waitress), que es el soportado. Con
varios procesos (gunicorn -w N) cada uno vería sus propias tareas y el «una a la
vez» dejaría de valer; está documentado en `docs/app_web.md`.

Límite conocido y NO resuelto: la cola es global —una tarea a la vez para todo el
servidor— y el segundo usuario recibe 409 en vez de esperar turno. Aislar la
vista por cuenta no reparte la cola. Una cola FIFO o paralelismo por usuario
queda fuera de alcance.
"""

from __future__ import annotations

import json
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

# Cuántas líneas de salida se conservan (las últimas). Ya no se enseñan, pero de
# ellas salen el progreso y el mensaje de error; con esto una tarea larga no se
# come la memoria del servidor guardando su log entero.
MAX_LINEAS = 400

# `    ajuste:  54%  (27313/50579)` del reentrenamiento.
_PROGRESO_PCT = re.compile(r"^\s*(\w+):\s+(\d{1,3})%\s+\((\d+)/(\d+)\)")
# `  [12/306] 3890561 Hoffenheim vs Schalke 04 — 28 jugadores (nuevo)` de la ingesta.
_PROGRESO_PASO = re.compile(r"^\s*\[(\d+)/(\d+)\]")
#
# El reentrenamiento emite además `[formulacion 2 | jugador | por_liga]` al
# empezar cada artefacto. NO se lee: eso es el reparto interno del pipeline —qué
# artefacto toca, con qué formulación y con qué normalización—, y la interfaz no
# habla ese idioma. Quien entrena desde el navegador elige un NOMBRE de modelo y
# un conjunto de datos; no elige formulación ni normalización (las fija
# `similitud.config.MODELOS_SERVIBLES`) ni entidad (van siempre las dos, con
# `--servibles`), así que enseñarle esos tres campos era exponer un vocabulario
# sobre el que no tiene ninguna decisión que tomar.


# Mensaje cuando la tarea en curso es de OTRA cuenta. Sin título y sin nombre: el
# título dice qué está entrenando quién («Entrenando el modelo «Fichajes 2027»»),
# que es justo lo que el aislamiento por cuenta viene a esconder.
OCUPADO_AJENO = ("Hay otra tarea en curso en el servidor, espera a que termine.")


class TareaEnCurso(RuntimeError):
    """Ya hay una tarea corriendo; no se admite una segunda."""


class TareaAjena(LookupError):
    """La tarea pedida no es de quien la pide (se responde como si no existiera)."""


@dataclass
class Tarea:
    """Una ejecución en segundo plano y lo que se sabe de ella."""

    id: str
    tipo: str
    titulo: str
    comando: list[str]
    # Cuenta que la lanzó. `None` solo en tareas creadas fuera de una petición.
    usuario: str | None = None
    estado: str = "en_curso"          # en_curso | terminada | fallida
    codigo: int | None = None
    lineas: list[str] = field(default_factory=list)
    progreso: float | None = None     # 0..1, None si no se puede estimar
    paso: str = ""
    inicio: float = field(default_factory=time.time)
    fin: float | None = None
    # Por qué acabó, cuando no fue por su cuenta: "timeout" | "cancelada".
    motivo: str = ""
    # Proceso vivo, para poder matarlo. No se serializa (no es del dominio de la
    # página, y un `Popen` no es JSON).
    proceso: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def segundos(self) -> float:
        return (self.fin or time.time()) - self.inicio

    @property
    def error(self) -> str:
        """Por qué falló, en una línea. Vacío mientras no haya fallado.

        Sustituye al volcado del log: la página ya no enseña la salida del hijo,
        pero cerrar una tarea en «fallida» sin decir nada más deja al usuario sin
        forma de saber si se equivocó de carpeta o se quedó sin disco. La última
        línea es el mensaje de error del CLI (o la nota que escribe `_cerrar` al
        vencer el plazo o al cancelar), que es justo lo que hace falta.
        """
        if self.estado != "fallida":
            return ""
        for linea in reversed(self.lineas):
            if linea.strip():
                return linea.strip()
        return f"La tarea terminó con código {self.codigo}."

    def es_de(self, usuario: str | None) -> bool:
        return self.usuario == usuario

    def como_dict(self) -> dict:
        """Lo que ve la página: qué es, cómo va y —si falló— por qué.

        Ni `lineas` ni `comando`: la interfaz no tiene consola donde pintarlos y
        los dos llevan rutas absolutas del servidor, así que enviarlos al
        navegador era exposición sin contrapartida.
        """
        return {
            "id": self.id,
            "tipo": self.tipo,
            "titulo": self.titulo,
            "estado": self.estado,
            "codigo": self.codigo,
            "progreso": self.progreso,
            "paso": self.paso,
            "motivo": self.motivo,
            "error": self.error,
            "segundos": round(self.segundos, 1),
        }


class GestorTareas:
    """Lanza y vigila las tareas largas. Una a la vez."""

    def __init__(self, cwd: Path, max_historial: int = 8,
                 duracion_max: float | None = None,
                 ruta_marca: Path | None = None) -> None:
        self.cwd = Path(cwd)
        self.duracion_max = duracion_max
        self.ruta_marca = Path(ruta_marca) if ruta_marca is not None else None
        self._max_historial = max_historial
        self._tareas: list[Tarea] = []
        self._actual: Tarea | None = None
        self._lock = threading.Lock()

    # --- Consulta -------------------------------------------------------------
    def en_curso(self) -> Tarea | None:
        with self._lock:
            return self._actual

    def ultima(self, usuario: str | None) -> Tarea | None:
        """La tarea DE ESA CUENTA que está corriendo o la última suya.

        Antes era global y `/datos/tarea` devolvía el título, el log y la línea
        de comandos de quien fuera. Ahora una tarea de otro es como si no
        existiera: `None`. La página trata ese `None` como «no hay nada que
        enseñar», que es exactamente lo que debe ver.
        """
        with self._lock:
            if self._actual is not None and self._actual.es_de(usuario):
                return self._actual
            for tarea in reversed(self._tareas):
                if tarea.es_de(usuario):
                    return tarea
            return None

    def hay_alguna_en_curso(self) -> bool:
        """¿Está el servidor ocupado? (sin decir con qué ni de quién)."""
        return self.en_curso() is not None

    # --- Lanzamiento ----------------------------------------------------------
    def lanzar(self, tipo: str, titulo: str, argumentos: list[str],
               usuario: str | None = None) -> Tarea:
        """Arranca `python -m <modulo de tipo> <argumentos>` en segundo plano."""
        if tipo not in MODULOS:
            raise ValueError(f"tipo de tarea desconocido: {tipo!r}")
        comando = [sys.executable, "-m", MODULOS[tipo], *argumentos]
        tarea = Tarea(id=uuid.uuid4().hex[:12], tipo=tipo, titulo=titulo,
                      comando=comando, usuario=usuario)
        with self._lock:
            if self._actual is not None:
                # El título solo se enseña si la tarea que ocupa es de quien
                # pregunta; si no, mensaje genérico (ver `OCUPADO_AJENO`).
                if self._actual.es_de(usuario):
                    raise TareaEnCurso(
                        f"Ya hay una tarea en curso ({self._actual.titulo}). "
                        "Espera a que termine."
                    )
                raise TareaEnCurso(OCUPADO_AJENO)
            self._actual = tarea
            self._tareas.append(tarea)
            del self._tareas[:-self._max_historial]
        threading.Thread(target=self._ejecutar, args=(tarea,), daemon=True).start()
        return tarea

    # --- Cancelación ----------------------------------------------------------
    def cancelar(self, id_tarea: str, usuario: str | None) -> Tarea:
        """Mata la tarea en curso si es de esa cuenta. `TareaAjena` si no lo es.

        Matar es lo único que se puede hacer: el hijo es un proceso aparte y no
        tiene ningún protocolo de parada ordenada. Como escribe sobre una COPIA
        de la BD (la ingesta) o sobre una carpeta de modelo recién creada (el
        reentrenamiento), lo que queda a medias es descartable y no toca nada de
        lo anterior.
        """
        with self._lock:
            actual = self._actual
            if actual is None or actual.id != id_tarea or not actual.es_de(usuario):
                raise TareaAjena(f"No hay ninguna tarea tuya con id {id_tarea!r}.")
            actual.motivo = "cancelada"
            proceso = actual.proceso
        # Fuera del lock: `kill` puede tardar, y el hilo que lee la salida
        # necesita el lock para registrar sus últimas líneas.
        if proceso is not None:
            self._matar(proceso)
        return actual

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

        with self._lock:
            tarea.proceso = proceso
        # La marca se escribe con el pid del HIJO, que es el que sobrevive a un
        # reinicio de la app; por eso se escribe aquí y no en `lanzar`, donde
        # todavía no existe.
        self._escribir_marca(tarea, proceso.pid)

        # Vigilante del plazo. Es un temporizador y no una comprobación dentro
        # del bucle porque el bucle está BLOQUEADO leyendo la salida del hijo: si
        # el hijo se cuelga sin escribir nada, ese bucle no vuelve a ejecutarse
        # nunca y cualquier comprobación suya llegaría tarde para siempre.
        vigilante = None
        if self.duracion_max:
            vigilante = threading.Timer(
                self.duracion_max, self._vencer, args=(tarea, proceso))
            vigilante.daemon = True
            vigilante.start()

        try:
            # En modo texto Python traduce `\r` como fin de línea, así que la
            # barra de progreso del hijo (que reescribe la misma línea con `\r`)
            # llega aquí como una línea por actualización, sin tratamiento
            # especial.
            assert proceso.stdout is not None
            for linea in proceso.stdout:
                self._registrar(tarea, linea.rstrip("\n"))
            codigo = proceso.wait()
        finally:
            if vigilante is not None:
                vigilante.cancel()

        # El texto se compone solo si hace falta: `duracion_max` puede ser None
        # (sin plazo) y formatearlo entonces reventaría el hilo justo al cerrar.
        nota = ""
        if tarea.motivo == "timeout":
            nota = (f"La tarea excedió el máximo de {self.duracion_max:.0f} s "
                    f"y se ha detenido.")
        elif tarea.motivo == "cancelada":
            nota = "Cancelada desde la interfaz."
        # Un proceso matado devuelve un código distinto de 0 en todos los
        # sistemas, así que `_cerrar` ya la marcaría fallida; el motivo es para
        # que la página diga POR QUÉ y no solo que falló.
        self._cerrar(tarea, codigo=codigo, ultima_linea=nota)

    def _vencer(self, tarea: Tarea, proceso: subprocess.Popen) -> None:
        """Se cumplió el plazo: se mata al hijo y se anota el motivo."""
        with self._lock:
            if tarea.estado != "en_curso" or tarea.motivo:
                return          # ya terminó, o ya la estaban cancelando
            tarea.motivo = "timeout"
        self._matar(proceso)

    @staticmethod
    def _matar(proceso: subprocess.Popen) -> None:
        """`kill` tolerante: que el hijo ya haya muerto no es un error."""
        try:
            proceso.kill()
        except (OSError, ValueError):
            pass

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
            # el reentrenamiento emite una cada 5% por modelo y, si no, la línea
            # que queda al final (la que explica un fallo) sería una de ellas.
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
            # Una tarea matada (plazo o cancelación) es fallida aunque el sistema
            # devolviera 0 por lo que sea: no llegó a hacer lo que se le pidió.
            terminada = codigo == 0 and not tarea.motivo
            tarea.estado = "terminada" if terminada else "fallida"
            tarea.fin = time.time()
            tarea.progreso = 1.0 if terminada else tarea.progreso
            tarea.proceso = None
            if self._actual is tarea:
                self._actual = None
        self._borrar_marca()

    # --- Marca en disco -------------------------------------------------------
    def _escribir_marca(self, tarea: Tarea, pid: int) -> None:
        """Deja constancia en disco de que hay un subproceso vivo.

        Se escribe en cuanto el hijo existe y se borra al cerrar la tarea. Si al
        levantar la app la marca sigue ahí con un pid vivo, es un huérfano de una
        ejecución anterior: la app lo avisa (ver `leer_marca` y `pid_vivo`),
        porque dos procesos escribiendo el mismo SQLite es riesgo de corrupción.

        `pid` es el del HIJO, no el de la app: el hijo es el que sobrevive al
        reinicio. Se guarda también `pid_app` para saber quién lo lanzó.

        Un fallo al escribirla no impide lanzar la tarea: es un aviso, no un
        cerrojo.
        """
        if self.ruta_marca is None:
            return
        try:
            self.ruta_marca.parent.mkdir(parents=True, exist_ok=True)
            self.ruta_marca.write_text(
                json.dumps({
                    "pid": pid,
                    "pid_app": os.getpid(),
                    "tipo": tarea.tipo,
                    "usuario": tarea.usuario,
                    "inicio": tarea.inicio,
                    "id": tarea.id,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _borrar_marca(self) -> None:
        if self.ruta_marca is None:
            return
        try:
            self.ruta_marca.unlink(missing_ok=True)
        except OSError:
            pass


def leer_marca(ruta: Path | None) -> dict | None:
    """La marca de tarea en curso que haya quedado en disco, o None."""
    if ruta is None:
        return None
    try:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return datos if isinstance(datos, dict) else None


def pid_vivo(pid: int) -> bool:
    """¿Sigue existiendo ese proceso?

    En POSIX, la señal 0 no hace nada y solo comprueba si se puede señalar.
    **En Windows NO se puede usar `os.kill(pid, 0)`**: allí cualquier señal que
    no sea CTRL_C_EVENT/CTRL_BREAK_EVENT se traduce a `TerminateProcess`, así que
    «comprobar» mataría el proceso — justo lo contrario de lo que se quiere. Se
    abre un handle con `OpenProcess` y se mira si sale válido.

    Un pid puede haberse reciclado y dar un falso positivo. Se asume: esto
    alimenta un AVISO al arrancar, no una decisión automática.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:     # existe, pero es de otro usuario
        return True
    return True


def huerfano(ruta: Path | None) -> dict | None:
    """Marca de una tarea cuyo proceso SIGUE VIVO, o None.

    Es lo que se consulta al arrancar: reiniciar la app no mata al subproceso
    (el hilo que lo vigilaba era daemon, pero el `Popen` es un proceso aparte y
    en Windows no muere con el padre). Un huérfano escribiendo en el mismo SQLite
    que la app acaba de abrir es riesgo de corrupción, así que se avisa.
    """
    marca = leer_marca(ruta)
    if not marca:
        return None
    try:
        pid = int(marca.get("pid", 0))
    except (TypeError, ValueError):
        return None
    return marca if pid_vivo(pid) else None


def _leer_progreso(linea: str) -> tuple[float | None, str]:
    """(fracción completada, etiqueta del paso) que se deduce de una línea.

    El paso solo lo pone la ingesta («partido 12 de 306»), que es un avance que
    se entiende sin saber nada del pipeline. El reentrenamiento aporta barra pero
    no paso: lo único que tenía que decir era qué artefacto estaba ajustando, y
    eso no se enseña (ver la nota de `_PROGRESO_PASO`, arriba).
    """
    m = _PROGRESO_PCT.match(linea)
    if m:
        hechos, total = int(m.group(3)), int(m.group(4))
        return (hechos / total if total else None), ""
    m = _PROGRESO_PASO.match(linea)
    if m:
        hechos, total = int(m.group(1)), int(m.group(2))
        return (hechos / total if total else None), f"partido {hechos} de {total}"
    return None, ""

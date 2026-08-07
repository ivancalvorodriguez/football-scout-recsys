"""Descubrimiento y cacheo de los modelos servibles de `outputs/modelo/`.

La app no sabe a priori qué modelos existen: `src.similitud.build` puede haber
generado solo una formulación, o solo una entidad. Este módulo escanea el
directorio, deduce las combinaciones disponibles a partir del nombre del
artefacto y las carga bajo demanda (`src.similitud.modelo.cargar_modelo`),
guardándolas en memoria: la matriz S de jugadores es de 2176x2176, releerla en
cada petición sería absurdo.

Junto al modelo se cachea su `MatrizFases` (el perfil por fases de TODAS las
entidades). Se calcula aquí y no en la vista porque los percentiles necesitan el
universo entero: es una propiedad del artefacto, no de la consulta, y comparte
con él la invalidación.

La caché se invalida por fecha de modificación del `.npz`, para que reconstruir
los modelos con el servidor levantado se refleje sin reiniciarlo (útil en
desarrollo y sin coste apreciable: un `stat` por petición).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

from src.similitud.modelo import ModeloSimilitud, cargar_modelo

from . import config
from .fases import MatrizFases, construir_matriz

# `formulacion<N>_<entidad>[_<normalizacion>].npz`. El artefacto sin sufijo es el
# formato antiguo y `cargar_modelo` lo trata como `por_liga`; se replica aquí esa
# misma convención para no ofrecer combinaciones que luego no se puedan cargar.
_PATRON = re.compile(
    r"^formulacion(?P<formulacion>\d+)_(?P<entidad>[a-z]+)"
    r"(?:_(?P<normalizacion>por_liga|global))?$"
)


class ModeloNoDisponible(LookupError):
    """No hay ningún artefacto para la combinación pedida."""


@dataclass(frozen=True, order=True)
class ClaveModelo:
    """Identifica un artefacto servible. Ordenable para listados estables."""

    entidad: str
    formulacion: str
    normalizacion: str

    @property
    def etiqueta(self) -> str:
        return (
            f"{config.ETIQUETA_ENTIDAD.get(self.entidad, self.entidad)} · "
            f"F{self.formulacion} · "
            f"{config.ETIQUETA_NORMALIZACION.get(self.normalizacion, self.normalizacion)}"
        )


@dataclass(frozen=True)
class Cargado:
    """Un artefacto ya en memoria: el modelo y su perfil por fases."""

    modelo: ModeloSimilitud
    fases: MatrizFases


class Catalogo:
    """Modelos disponibles en un directorio, con carga perezosa y cacheada."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        # clave -> (huella del npz, cargado). El lock protege la caché: el
        # servidor de desarrollo de Flask es multihilo y cargar el mismo modelo
        # dos veces a la vez desperdicia memoria y tiempo.
        self._cache: dict[ClaveModelo, tuple[tuple[float, int], Cargado]] = {}
        self._lock = threading.Lock()

    # --- Descubrimiento ------------------------------------------------------

    def disponibles(self) -> list[ClaveModelo]:
        """Combinaciones presentes en disco, ordenadas y sin duplicados.

        Un `.npz` sin su `.json` no cuenta: el JSON trae los nombres de las
        entidades, sin los cuales el modelo no se puede consultar por nombre.
        """
        if not self.model_dir.is_dir():
            return []
        claves: set[ClaveModelo] = set()
        for npz in self.model_dir.glob("formulacion*.npz"):
            m = _PATRON.match(npz.stem)
            if m is None or not npz.with_suffix(".json").exists():
                continue
            entidad = m.group("entidad")
            formulacion = m.group("formulacion")
            if entidad not in config.ENTIDADES or formulacion not in config.FORMULACIONES:
                continue
            claves.add(ClaveModelo(
                entidad=entidad,
                formulacion=formulacion,
                # Sin sufijo = artefacto antiguo, servido como `por_liga`.
                normalizacion=m.group("normalizacion") or "por_liga",
            ))
        return sorted(claves)

    def entidades_disponibles(self) -> list[str]:
        """Tipos de entidad con al menos un modelo, en el orden de `config`."""
        con_modelo = {c.entidad for c in self.disponibles()}
        return [e for e in config.ENTIDADES if e in con_modelo]

    def opciones(self, entidad: str) -> list[ClaveModelo]:
        """Modelos disponibles para un tipo de entidad."""
        return [c for c in self.disponibles() if c.entidad == entidad]

    # --- Resolución ----------------------------------------------------------

    def resolver(
        self,
        entidad: str,
        formulacion: str | None = None,
        normalizacion: str | None = None,
    ) -> ClaveModelo:
        """Clave concreta a partir de una petición posiblemente incompleta.

        Lo único obligatorio es la entidad (es lo que pide el usuario: jugador o
        equipo). Formulación y normalización se completan con el orden de
        preferencia de `config` entre lo que realmente exista, para que la app
        funcione aunque solo se haya construido una parte de los modelos.
        """
        candidatos = self.opciones(entidad)
        if not candidatos:
            raise ModeloNoDisponible(
                f"No hay ningún modelo de '{entidad}' en {self.model_dir}. "
                f"Constrúyelos con: python -m src.similitud.build"
            )
        if formulacion is not None:
            candidatos = [c for c in candidatos if c.formulacion == formulacion]
        if normalizacion is not None:
            candidatos = [c for c in candidatos if c.normalizacion == normalizacion]
        if not candidatos:
            pedido = (
                f"entidad={entidad!r}, formulacion={formulacion!r}, "
                f"normalizacion={normalizacion!r}"
            )
            raise ModeloNoDisponible(f"No hay modelo para {pedido}.")

        def preferencia(clave: ClaveModelo) -> tuple[int, int]:
            return (
                _indice(config.PREFERENCIA_FORMULACION, clave.formulacion),
                _indice(config.PREFERENCIA_NORMALIZACION, clave.normalizacion),
            )

        return min(candidatos, key=preferencia)

    # --- Carga ---------------------------------------------------------------

    def cargado(self, clave: ClaveModelo) -> Cargado:
        """Modelo de `clave` y su perfil por fases, desde la caché si sigue vigente.

        La vigencia se mide con (mtime, tamaño) del `.npz`: es barato y detecta
        cualquier reconstrucción normal. No pretende ser un hash — dos
        reconstrucciones del mismo tamaño dentro de la misma marca de tiempo del
        sistema de ficheros pasarían por idénticas, algo irrelevante fuera de un
        test.
        """
        npz = self._ruta(clave)
        if not npz.exists():
            raise ModeloNoDisponible(f"Falta el artefacto {npz}.")
        estado = npz.stat()
        huella = (estado.st_mtime, estado.st_size)
        with self._lock:
            guardado = self._cache.get(clave)
            if guardado is not None and guardado[0] == huella:
                return guardado[1]
        # Fuera del lock: cargar es lento (I/O + descompresión) y no queremos
        # bloquear al resto de peticiones mientras tanto.
        modelo = cargar_modelo(
            self.model_dir,
            clave.formulacion,
            clave.entidad,
            normalizacion=clave.normalizacion,
        )
        entrada = Cargado(
            modelo=modelo,
            fases=construir_matriz(
                modelo.entidad, modelo.feat_names, modelo.feat_display
            ),
        )
        with self._lock:
            self._cache[clave] = (huella, entrada)
        return entrada

    def obtener(self, clave: ClaveModelo) -> ModeloSimilitud:
        """Solo el modelo de `clave` (atajo de `cargado`)."""
        return self.cargado(clave).modelo

    def _ruta(self, clave: ClaveModelo) -> Path:
        """Ruta del `.npz` de `clave` (con el fallback sin sufijo de `por_liga`)."""
        base = f"formulacion{clave.formulacion}_{clave.entidad}"
        con_sufijo = self.model_dir / f"{base}_{clave.normalizacion}.npz"
        if con_sufijo.exists() or clave.normalizacion != "por_liga":
            return con_sufijo
        return self.model_dir / f"{base}.npz"


def _indice(orden: tuple[str, ...], valor: str) -> int:
    """Posición de `valor` en `orden`; al final si no aparece."""
    return orden.index(valor) if valor in orden else len(orden)

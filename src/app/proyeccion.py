"""Recomendar a una entidad que está en la base de datos y no en el modelo.

Un artefacto servible es una foto: la S de quienes estaban al ajustarlo. La base
de datos, en cambio, se puede ampliar desde `/datos` —o elegir otra distinta con
el desplegable del buscador— y entonces hay jugadores y equipos que la BD conoce
y el modelo no. Hasta ahora el buscador respondía «no encontrado» a esas
consultas, que es la respuesta más pobre posible: los datos para contestar están
ahí, solo que no dentro de la matriz.

Este módulo es la capa de la app sobre `src.similitud.foldin`, que es quien sabe
colocar una entidad nueva en la geometría que el modelo ya aprendió. Aquí se
resuelve lo que el pipeline no tiene por qué saber:

- **De dónde salen las features de esa entidad**: de la BD elegida, pasadas por
  la MISMA cadena que el pipeline (`data.cargar` → `features.construir`) y con
  las **mu/sd congeladas del modelo** (`meta["estadisticas_normalizacion"]`).
  Recalcular el z-score con los datos de la BD nueva pondría a la entidad nueva y
  a las del modelo en escalas distintas, y la similitud dejaría de significar lo
  que dice significar.
- **De dónde sale el ancho de kernel** de la F5: del estado warm
  (`<stem>.warm.npz`) del artefacto, nunca estimado de la consulta (ver
  `foldin.embeddings`).
- **De dónde sale la geometría**: del mismo estado warm. El modelo servido es
  `mahalanobis`, cuyo blanqueo se aprendió de los datos del ajuste; reestimarlo
  con los de la consulta pondría a la entidad nueva en otro espacio que las del
  modelo. Sin estado warm no se proyecta (ver `foldin.espacio_del_modelo`).
- **Que el espacio sea el mismo**: si las columnas que produce la BD no son
  exactamente las del artefacto (un modelo construido con otro `config`, p. ej.
  sin el bloque `pos_*`), no se proyecta nada. Vale más un «no puedo» que un
  ranking calculado sobre un vector que no es el suyo.

Se cachea por (BD, artefacto) la matriz de features **y la referencia del modelo
ya preparada** (`foldin.preparar`), con la misma huella (mtime, tamaño) que el
resto de catálogos. Las dos cosas son caras y ninguna depende de a quién se
consulte: sin el caché, cada búsqueda vuelve a derivar las ~50.000 filas de la BD
y a mapear por RFF todas las observaciones del modelo — medido, 1,8 s por
consulta frente a decenas de milisegundos. El caché guarda pocas entradas a
propósito (`MAX_EN_CACHE`): cada una son decenas de MB.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.similitud import data, features, foldin
from src.similitud.features import EstadisticasNorm, MatrizFeatures
from src.similitud.modelo import ModeloSimilitud
from src.similitud.warm import EstadoWarm

# Cuántas matrices de features se conservan en memoria a la vez. Cada una es la
# BD entera en float64 (unos 35 MB con los jugadores de las cinco ligas), así que
# el caché es deliberadamente corto: cubre «el usuario encadena consultas» y no
# «el servidor acumula todas las combinaciones que se hayan pedido nunca».
MAX_EN_CACHE = 2


@dataclass(frozen=True)
class Proyectada:
    """Una entidad de la BD colocada en un modelo que no la contiene.

    - `proyeccion` es lo que devuelve `foldin` (puntuación contra cada entidad
      del modelo, y de qué fidelidad).
    - `display` es su vector de features z-scoreadas agregado como lo agrega el
      artefacto (`modelo.features_display`), para que la ficha, el radar y las
      coincidencias se calculen igual que con una entidad del modelo.
    - `nombre` y `ligas` salen de la BD: el artefacto no sabe nada de ella.
    """

    id: int
    nombre: str
    ligas: tuple[str, ...]
    display: np.ndarray
    proyeccion: foldin.Proyeccion

    @property
    def fidelidad(self) -> str:
        return self.proyeccion.fidelidad

    @property
    def aproximada(self) -> bool:
        return self.proyeccion.fidelidad != foldin.FIEL

    @property
    def n_observaciones(self) -> int:
        return self.proyeccion.n_observaciones


class CatalogoProyeccion:
    """Proyecta entidades de UNA base de datos sobre los modelos que se le pasen.

    Una instancia por BD (la crea `fuentes`, como los demás catálogos). El modelo
    entra en cada llamada y no en el constructor porque la misma BD se consulta
    con modelos distintos: es la pareja (BD, modelo) la que define un espacio.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        # clave -> (huella, features + referencia | None). None es «no se puede»
        # (BD ausente, esquema raro, columnas que no son las del modelo, falta el
        # estado warm): se cachea igual para no reintentarlo en cada consulta.
        self._cache: OrderedDict[tuple, tuple[tuple, "_Espacio | None"]] = \
            OrderedDict()
        self._lock = threading.Lock()

    # --- Huellas ---------------------------------------------------------------

    def _huella(self, ruta_npz: Path) -> tuple:
        """(BD, artefacto) tal y como están AHORA.

        Con las dos: reentrenar el modelo o ampliar la BD con el servidor
        levantado invalida las features cacheadas, que dependen de ambas cosas.
        """
        def marca(ruta: Path) -> tuple[float, int] | None:
            if not Path(ruta).is_file():
                return None
            estado = Path(ruta).stat()
            return (estado.st_mtime, estado.st_size)

        return (marca(self.db_path), marca(ruta_npz))

    # --- Features en el espacio del modelo -------------------------------------

    def _construir(self, modelo: ModeloSimilitud) -> MatrizFeatures | None:
        """Features de TODA la BD en el espacio del modelo, o None si no se puede.

        Se construyen todas y no solo las de la entidad consultada porque
        proyectar necesita también las observaciones de las entidades del modelo:
        el embedding (F5) y las vecinas de cada columna (F2) se calculan contra
        ellas, y tienen que salir de esta misma BD para que sean comparables.
        """
        estadisticas = EstadisticasNorm.desde_dict(
            modelo.meta.get("estadisticas_normalizacion"))
        if estadisticas is None:
            # Artefacto anterior a que se guardaran las mu/sd: sin ellas solo se
            # podría reestandarizar con los datos de la consulta, que es
            # justamente lo que invalidaría la comparación.
            return None
        if not self.db_path.is_file():
            return None
        try:
            df = data.cargar(self.db_path, modelo.entidad)
            if df.empty:
                return None
            mf = features.construir(
                df,
                modelo.entidad,
                normalizacion=str(modelo.meta.get("normalizacion") or "global"),
                estadisticas=estadisticas,
            )
        except Exception:
            # Mismo criterio que en `crudos`: esto es una capacidad extra, y una
            # BD con otro esquema no puede tumbar la búsqueda.
            return None
        if list(mf.feat_names) != list(modelo.feat_names):
            # Otro espacio de features (otro `config` al construir el modelo): la
            # proyección no sería del vector de este modelo.
            return None
        return mf

    def _preparar(self, modelo: ModeloSimilitud, ruta_npz: Path) -> "_Espacio | None":
        """Features de la BD + modelo listo para recibir proyecciones."""
        mf = self._construir(modelo)
        if mf is None:
            return None
        estado = _estado(ruta_npz)
        sigma = estado.sigma if (estado is not None and modelo.formulacion == "5") else None
        if modelo.formulacion == "5" and sigma is None:
            # Sin el ancho de kernel del ajuste no se proyecta y no se estima:
            # ver `foldin.embeddings`.
            return None
        try:
            # `estado` lleva además la geometría: un modelo `mahalanobis` sin él
            # no es proyectable y `foldin` lo dice con una excepción, que aquí se
            # traduce en «esta BD no sirve para este modelo» (409 en la vista).
            ref = foldin.preparar(modelo, mf, sigma=sigma, estado=estado)
        except foldin.EntidadNoProyectable:
            return None
        return _Espacio(mf=mf, ref=ref)

    def espacio(self, modelo: ModeloSimilitud, ruta_npz: Path) -> "_Espacio | None":
        """Pareja (features, referencia) de esta BD con `modelo`, cacheada."""
        clave = (modelo.entidad, str(ruta_npz))
        huella = self._huella(ruta_npz)
        with self._lock:
            guardado = self._cache.get(clave)
            if guardado is not None and guardado[0] == huella:
                self._cache.move_to_end(clave)
                return guardado[1]
        # Fuera del lock: es lo caro de todo esto (derivar la BD entera y, en la
        # F5, mapear por RFF todas sus observaciones).
        espacio = self._preparar(modelo, ruta_npz)
        with self._lock:
            self._cache[clave] = (huella, espacio)
            self._cache.move_to_end(clave)
            while len(self._cache) > MAX_EN_CACHE:
                self._cache.popitem(last=False)
        return espacio

    def features(self, modelo: ModeloSimilitud, ruta_npz: Path) -> MatrizFeatures | None:
        """Solo las features de la BD en el espacio de `modelo` (atajo)."""
        espacio = self.espacio(modelo, ruta_npz)
        return None if espacio is None else espacio.mf

    # --- Consulta ---------------------------------------------------------------

    def proyectar(
        self, modelo: ModeloSimilitud, ruta_npz: Path, entity_id: int
    ) -> Proyectada | None:
        """Coloca `entity_id` en `modelo` usando los datos de esta BD.

        Devuelve None cuando esta BD no sirve para proyectar sobre ese modelo
        (no está, otro esquema, otro espacio de features, o falta el estado warm
        de un modelo que lo necesita). Lanza `foldin.EntidadNoProyectable` cuando
        el que no admite la operación es el MODELO, o cuando la entidad no tiene
        observaciones aquí: son cosas distintas y la interfaz las explica
        distinto.
        """
        espacio = self.espacio(modelo, ruta_npz)
        if espacio is None:
            return None
        proyeccion = foldin.proyectar_desde(espacio.ref, espacio.mf, entity_id)
        mf = espacio.mf
        suyas = np.asarray(mf.entity_id).astype(np.int64) == int(entity_id)
        return Proyectada(
            id=int(entity_id),
            nombre=str(np.asarray(mf.entity_name)[suyas][0]),
            ligas=tuple(dict.fromkeys(str(x) for x in np.asarray(mf.league)[suyas])),
            display=_display(mf, suyas),
            proyeccion=proyeccion,
        )


@dataclass(frozen=True)
class _Espacio:
    """Lo que hay que tener montado para proyectar sobre un modelo desde una BD."""

    mf: MatrizFeatures
    ref: foldin.Referencia


def _estado(ruta_npz: Path) -> EstadoWarm | None:
    """Estado warm del artefacto, o None si no está.

    Se lee ENTERO y no solo la `sigma` porque de ahí salen las dos cosas que la
    proyección no puede reestimar sin cambiar de espacio:

    - el **ancho del kernel** de la F5 (`sigma`), que no se estima (ver `foldin`);
    - la **geometría** del ajuste (`distancia` + `espacio_L`), que con
      `mahalanobis` incluye el blanqueo aprendido de los datos del ajuste. Sin él,
      `foldin.espacio_del_modelo` rechaza la proyección en vez de blanquear otra
      vez con los datos de la consulta.

    No lleva caché propio porque solo se lee al montar el espacio, que ya está
    cacheado.
    """
    # `<stem>.npz` -> `<stem>.warm.npz`, que es donde lo deja el ajuste.
    ruta_warm = Path(ruta_npz).with_suffix(".warm.npz")
    if not ruta_warm.is_file():
        return None
    try:
        return EstadoWarm.cargar(ruta_warm)
    except (OSError, ValueError, KeyError):
        return None


def _display(mf: MatrizFeatures, filas: np.ndarray) -> np.ndarray:
    """Vector de la entidad: media de sus filas ponderada por minutos.

    Es lo mismo que `modelo.features_display` hace con las entidades del ajuste
    (misma ponderación, mismas features ya estandarizadas), aplicado a una sola.
    Sin esto, la ficha de una entidad proyectada no podría enseñar ni su radar ni
    en qué coincide con los candidatos: esas dos cosas leen `feat_display`, y ella
    no tiene fila ahí.
    """
    X = np.asarray(mf.X, dtype=float)[filas]
    peso = np.asarray(mf.weight, dtype=float)[filas]
    masa = float(peso.sum())
    if masa <= 0.0:
        return X.mean(axis=0)
    return (X * peso[:, None]).sum(axis=0) / masa

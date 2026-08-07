"""Valor real de cada métrica por entidad, sin estandarizar (desde la BD).

Los artefactos guardan `feat_display`: la media ponderada por minutos de las
features **z-scoreadas**. Va bien para ordenar («esta métrica es la que más lo
separa de la media») pero es ilegible como estadística: a un ojeador, «+2,41» no
le dice nada y «3,4 pases progresivos por 90'» sí.

El z-score no es invertible desde el artefacto (haría falta la media y la
desviación con las que se estandarizó, que no se serializan), así que el valor
real se reconstruye desde la BD con la MISMA cadena que el pipeline:
`similitud.data.cargar` → `similitud.features.derivar`, que es `construir`
parándose justo antes de estandarizar. De ahí que las columnas coincidan una a
una con `feat_names` del artefacto.

Agregación: media ponderada por minutos (jugador) o simple (equipo), la misma
que `modelo.features_display` aplica sobre las features ya estandarizadas. La
diferencia está en los NaN: un ratio sin denominador (0 regates intentados) aquí
se queda en NaN y se excluye del promedio, en vez de rellenarse a 0 como hace la
capa del modelo — un 0 significaría «0 % de acierto», que es falso.

Como el resto de lo que sale de la BD (`ligas`, `contexto`), esto es opcional:
sin BD no hay valores reales y la interfaz vuelve a enseñar el z-score, que
siempre está en el artefacto.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.similitud import data, features


@dataclass(frozen=True)
class TablaCrudos:
    """Valor real de cada métrica para todas las entidades de un tipo."""

    feat_names: list[str]
    entity_ids: np.ndarray   # (P,) ordenados
    valores: np.ndarray      # (P, d), con NaN donde la métrica no está definida

    def alinear(
        self, entity_ids: np.ndarray, feat_names: list[str]
    ) -> np.ndarray | None:
        """Reordena la tabla para que encaje fila a fila con un modelo.

        El artefacto fija su propio orden de entidades y de columnas, y no tiene
        por qué ser el de aquí (ni siquiera el mismo universo, si la BD se
        reextrajo después de construir el modelo). Lo que no se encuentra queda
        en NaN, que la interfaz ya sabe mostrar como «sin dato».

        Devuelve None si no coincide NADA: es señal de que la tabla no
        corresponde a este modelo, y es preferible no enseñar valores a enseñar
        una columna entera de huecos.
        """
        fila_por_id = {int(i): f for f, i in enumerate(self.entity_ids)}
        col_por_nombre = {n: c for c, n in enumerate(self.feat_names)}
        filas = np.array([fila_por_id.get(int(i), -1) for i in entity_ids])
        cols = np.array([col_por_nombre.get(n, -1) for n in feat_names])
        if not (filas >= 0).any() or not (cols >= 0).any():
            return None
        salida = np.full((len(entity_ids), len(feat_names)), np.nan)
        # Se indexa solo lo emparejado; el resto se queda en NaN.
        f_ok = np.flatnonzero(filas >= 0)
        c_ok = np.flatnonzero(cols >= 0)
        salida[np.ix_(f_ok, c_ok)] = self.valores[np.ix_(filas[f_ok], cols[c_ok])]
        return salida


class CatalogoCrudos:
    """Valores reales por tipo de entidad, leídos de la BD y cacheados."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._cache: dict[str, TablaCrudos | None] = {}
        self._huella: tuple[float, int] | None = None
        self._lock = threading.Lock()

    def _cargar(self, entidad: str) -> TablaCrudos | None:
        """Deriva y agrega los valores reales de un tipo de entidad.

        Cualquier fallo (BD ausente, esquema antiguo, columna que ya no existe)
        deja el tipo sin valores reales; la interfaz cae al z-score. No se deja
        escapar la excepción: esto es un adorno, y romper la página por no poder
        adornarla sería peor que la propia falta del adorno.
        """
        if not self.db_path.is_file():
            return None
        try:
            df = data.cargar(self.db_path, entidad)
            if df.empty:
                return None
            derivadas = features.derivar(df, entidad)
            pesos = features.masa(df, entidad)
        except Exception:
            return None
        ids = df["entity_id"].to_numpy()
        unicos, inverso = np.unique(ids, return_inverse=True)
        return TablaCrudos(
            feat_names=list(derivadas.columns),
            entity_ids=unicos,
            valores=_media_ponderada(
                derivadas.to_numpy(dtype=float), pesos, inverso, len(unicos)
            ),
        )

    def tabla(self, entidad: str) -> TablaCrudos | None:
        """Tabla de `entidad`, releída si la BD ha cambiado."""
        huella: tuple[float, int] | None = None
        if self.db_path.is_file():
            estado = self.db_path.stat()
            huella = (estado.st_mtime, estado.st_size)
        with self._lock:
            if self._huella != huella:
                self._cache.clear()
                self._huella = huella
            if entidad in self._cache:
                return self._cache[entidad]
        # Fuera del lock: derivar las ~42.000 observaciones de jugador tarda
        # medio segundo y no hay por qué bloquear al resto de peticiones.
        tabla = self._cargar(entidad)
        with self._lock:
            self._cache[entidad] = tabla
        return tabla

    def matriz(
        self, entidad: str, entity_ids: np.ndarray, feat_names: list[str]
    ) -> np.ndarray | None:
        """Valores reales alineados con un modelo concreto, o None si no hay."""
        tabla = self.tabla(entidad)
        return None if tabla is None else tabla.alinear(entity_ids, feat_names)


def _media_ponderada(
    X: np.ndarray, pesos: np.ndarray, grupo: np.ndarray, n_grupos: int
) -> np.ndarray:
    """Media de X por grupo, ponderada por `pesos` e ignorando los NaN.

    Cada columna lleva su propia masa acumulada porque los NaN no están en las
    mismas filas en todas: un jugador puede tener 30 partidos con pases y solo 4
    con algún regate intentado, y el % de regate tiene que promediarse sobre esos
    4. Un grupo sin ningún valor válido en una columna queda en NaN.
    """
    validos = np.isfinite(X)
    aportado = np.where(validos, X, 0.0) * pesos[:, None]
    masa = np.where(validos, pesos[:, None], 0.0)
    suma = np.zeros((n_grupos, X.shape[1]), dtype=float)
    total = np.zeros((n_grupos, X.shape[1]), dtype=float)
    np.add.at(suma, grupo, aportado)
    np.add.at(total, grupo, masa)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0.0, suma / total, np.nan)

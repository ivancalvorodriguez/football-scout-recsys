"""Geometria de una superficie del barrido: escala real de los ejes e interpolacion.

Las superficies del barrido se dibujaban con los ejes en posiciones ORDINALES: una
casilla por valor barrido, todas del mismo ancho, sin importar que el salto real
fuera `0.125 -> 0.25` o `5 -> 7.5`. Con esa convencion la figura MIENTE sobre la
forma del optimo: dos rejillas con los mismos valores medidos y saltos distintos
salen identicas, la pendiente entre casillas no es comparable con la de al lado, y
una meseta ancha en la escala real puede parecer una cresta estrecha (o al reves)
solo por como se eligieron los valores a probar.

Este modulo pone cada punto medido en su COORDENADA REAL y estima la superficie
entre ellos:

1. **Escala del eje** (`detectar`). En `auto`, cada eje elige la suya segun sus
   propios valores: `log` si son claramente multiplicativos (512/1024/2048,
   5/10/25), `lineal` si no (0.125…1.0), `ordinal` si no son numeros. Es lo que
   evita el problema por el que se puso ordinal en su dia: un eje que va de 10 a
   500 en escala lineal amontona tres cuartos de los puntos contra un borde, y en
   log queda regular. La escala usada se declara en el pie de la figura.
2. **Malla fina** (`Eje.uf`). Entre cada par de valores medidos se insertan nodos,
   repartidos en proporcion al ancho real del tramo. Los valores medidos siguen
   siendo nodos de la malla (`Eje.idx` dice cuales), asi que la superficie pasa
   exactamente por ellos.
3. **Interpolacion** (`interpolar`). Producto tensorial de PCHIP 1D (Hermite
   cubico con las pendientes de Fritsch-Carlson): primero a lo largo de X, despues
   a lo largo de Y. Se elige PCHIP y no un spline cubico natural ni Catmull-Rom
   por una razon concreta: **un spline sobrepasa**, y en una figura cuya lectura
   es "donde esta el optimo" un sobrepaso pinta un maximo que nadie ha medido.
   PCHIP preserva la monotonia local, con lo que la superficie estimada nunca sale
   del rango de los puntos que la rodean ni crea extremos nuevos.

Nada de esto convierte la estimacion en dato: lo medido son los nodos de `Eje.u`,
y las figuras los marcan aparte. Todo en numpy puro, como el resto del nucleo
numerico del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ESCALAS = ("auto", "lineal", "log", "ordinal")

# Nodos (aproximados) por eje de la malla fina. Es solo resolucion de dibujo: no
# cambia ningun numero, solo cuanto se nota el facetado. El visor HTML ordena por
# profundidad todos los cuadrilateros en cada fotograma, asi que subirlo mucho se
# paga alli antes que en los PNG.
DENSIDAD = 32

# Umbrales de `detectar`. Con menos de `_RANGO_MIN_LOG` de recorrido (max/min) la
# escala log no arregla nada y complica la lectura; y solo se prefiere si reparte
# los valores CLARAMENTE mejor que la lineal (`_VENTAJA_LOG`), porque ante la duda
# la escala honesta es la lineal.
_RANGO_MIN_LOG = 4.0
_VENTAJA_LOG = 0.7


# --------------------------------------------------------------------------- #
# Escala de un eje                                                              #
# --------------------------------------------------------------------------- #

def _numericos(valores: list[object]) -> np.ndarray | None:
    """Los valores como float, o None si alguno no es un numero real.

    Los booleanos se excluyen a proposito: `True` es un `int` para Python, pero un
    eje `USE_X = True/False` no tiene coordenada real que dibujar.
    """
    out: list[float] = []
    for v in valores:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            return None
        out.append(float(v))
    return np.asarray(out, dtype=float)


def _dispersion(pasos: np.ndarray) -> float:
    """Coeficiente de variacion de los saltos: 0 = perfectamente equiespaciados."""
    media = float(np.mean(pasos))
    return float(np.std(pasos) / media) if media > 0 else 0.0


def detectar(valores: list[object]) -> str:
    """Escala que reparte mejor esos valores: "lineal", "log" u "ordinal".

    Compara como de regulares quedan los saltos en cada escala. `F5_RFF_DIM`
    (512, 1024, 2048) es perfectamente regular en log y no en lineal; `F2_L1`
    (0.125 … 1.0) esta casi igual de repartido en las dos y se queda en lineal.
    """
    x = _numericos(valores)
    if x is None or x.size < 2 or np.any(np.diff(x) <= 0):
        # No numerico, un solo valor o sin orden estricto: no hay coordenada real.
        return "ordinal"
    if x.size == 2:
        # Con dos puntos cualquier escala monotona dibuja exactamente lo mismo
        # (una recta entre dos nodos): declarar "log" solo confundiria el pie.
        return "lineal"
    if x[0] <= 0 or x[-1] / x[0] < _RANGO_MIN_LOG:
        return "lineal"
    if _dispersion(np.diff(np.log(x))) <= _VENTAJA_LOG * _dispersion(np.diff(x)):
        return "log"
    return "lineal"


def _coordenadas(valores: list[object], escala: str) -> np.ndarray:
    """Coordenada de dibujo de cada valor medido, en la escala pedida."""
    if escala == "ordinal":
        return np.arange(len(valores), dtype=float)
    x = _numericos(valores)
    if x is None:                                    # no deberia pasar: `eje` lo filtra
        return np.arange(len(valores), dtype=float)
    return np.log10(x) if escala == "log" else x


def _malla_fina(u: np.ndarray, densidad: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodos intermedios entre los medidos. Devuelve (coordenadas, indices medidos).

    Los nodos se reparten en PROPORCION al ancho real de cada tramo, que es justo
    lo que hace que la superficie estimada sea homogenea: un tramo `5 -> 7.5` no
    recibe la misma resolucion que un `1 -> 5`. Los valores medidos siempre son
    nodos de la malla, asi que la superficie pasa exactamente por ellos.
    """
    u = np.asarray(u, dtype=float)
    if u.size < 2 or densidad <= 0:
        return u.copy(), np.arange(u.size)

    anchos = np.diff(u)
    objetivo = max(int(densidad), u.size)
    trozos_por_tramo = np.maximum(2, np.rint(anchos / anchos.sum() * (objetivo - 1)))

    trozos: list[np.ndarray] = []
    indices = [0]
    for a, b, k in zip(u[:-1], u[1:], trozos_por_tramo.astype(int)):
        trozos.append(np.linspace(a, b, k + 1)[:-1])   # sin el extremo: lo abre el siguiente
        indices.append(indices[-1] + int(k))
    trozos.append(np.asarray([u[-1]]))
    return np.concatenate(trozos), np.asarray(indices)


# `eq=False`: los campos son arrays de numpy, y el `__eq__`/`__hash__` que generaria
# el dataclass fallaria sobre ellos. Un eje se identifica por su nombre, no se compara.
@dataclass(frozen=True, eq=False)
class Eje:
    """Un eje de la superficie: sus valores medidos y donde se dibuja cada uno."""

    nombre: str
    valores: list[object]
    escala: str                 # "lineal" | "log" | "ordinal"
    u: np.ndarray               # coordenada de dibujo de cada valor medido
    uf: np.ndarray              # coordenadas de la malla fina (incluye las de `u`)
    idx: np.ndarray             # posicion de cada valor medido dentro de `uf`

    @property
    def etiquetas(self) -> list[str]:
        """Etiqueta de cada tick: SIEMPRE el valor real, sea cual sea la escala."""
        return [f"{v:g}" if isinstance(v, float) else str(v) for v in self.valores]

    def normalizar(self, x) -> np.ndarray:
        """Coordenadas llevadas a [-1, 1] (lo que espera el visor HTML)."""
        lo, hi = float(self.u.min()), float(self.u.max())
        x = np.asarray(x, dtype=float)
        if hi <= lo:
            return np.zeros_like(x)
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def medidos(self) -> np.ndarray:
        """Mascara sobre `uf`: True en los nodos que son un valor realmente barrido."""
        m = np.zeros(self.uf.size, dtype=bool)
        m[self.idx] = True
        return m

    def descripcion(self) -> str:
        """Como se dibuja este eje, para el pie de la figura."""
        etqs = self.etiquetas
        rango = f"{etqs[0]} … {etqs[-1]}" if len(etqs) > 1 else etqs[0]
        if self.escala == "ordinal":
            return (f"{self.nombre} en posiciones ordinales ({rango}; sus valores no "
                    "admiten coordenada real, los saltos del dibujo son iguales)")
        como = "logaritmica" if self.escala == "log" else "lineal"
        return f"{self.nombre} en escala real {como} ({rango})"


def eje(nombre: str, valores: list[object], escala: str = "auto",
        densidad: int = DENSIDAD, avisar=print) -> Eje:
    """Construye el eje: elige escala (si `auto`), coordenadas y malla fina."""
    valores = list(valores)
    if escala == "auto":
        elegida = detectar(valores)
    else:
        elegida = escala
        x = _numericos(valores)
        if elegida != "ordinal" and x is None:
            avisar(f"  [aviso] el eje {nombre} no tiene valores numericos: se dibuja "
                   "en posiciones ordinales pese a --escala.")
            elegida = "ordinal"
        elif elegida == "log" and x is not None and float(np.min(x)) <= 0:
            avisar(f"  [aviso] el eje {nombre} tiene valores <= 0: la escala "
                   "logaritmica no es posible, se dibuja lineal.")
            elegida = "lineal"

    u = _coordenadas(valores, elegida)
    uf, idx = _malla_fina(u, densidad)
    return Eje(nombre, valores, elegida, u, uf, idx)


# --------------------------------------------------------------------------- #
# Interpolacion PCHIP (Hermite cubico monotono, Fritsch-Carlson)                #
# --------------------------------------------------------------------------- #

def _pendiente_extremo(h0: float, h1: float, d0: float, d1: float) -> float:
    """Pendiente en el primer/ultimo nodo, recortada para no sobrepasar.

    Extrapolacion cuadratica de las dos primeras diferencias, con los dos frenos
    clasicos: si apunta al contrario que la diferencia adyacente se anula, y si
    supera el triple de esa diferencia se recorta. Sin ellos la superficie
    despegaria en los bordes, que es justo donde se lee si el optimo se sale de la
    rejilla.
    """
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(m) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(m) > abs(3.0 * d0):
        return 3.0 * d0
    return float(m)


def _pendientes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pendientes de Fritsch-Carlson: media armonica ponderada, 0 en los extremos locales.

    Anular la pendiente donde la serie cambia de sentido es lo que impide que la
    curva se pase del punto medido: el maximo de la superficie estimada esta
    siempre en un punto medido, no entre dos.
    """
    n = x.size
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.empty(n, dtype=float)
    if n == 2:
        m[:] = d[0]
        return m

    mismo_signo = np.sign(d[:-1]) * np.sign(d[1:]) > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        armonica = (w1 + w2) / (w1 / d[:-1] + w2 / d[1:])
    m[1:-1] = np.where(mismo_signo, armonica, 0.0)
    m[0] = _pendiente_extremo(h[0], h[1], d[0], d[1])
    m[-1] = _pendiente_extremo(h[-1], h[-2], d[-1], d[-2])
    return m


def _hermite(x: np.ndarray, y: np.ndarray, m: np.ndarray, xf: np.ndarray) -> np.ndarray:
    """Evalua el Hermite cubico de nodos `x`, valores `y` y pendientes `m` en `xf`."""
    j = np.clip(np.searchsorted(x, xf, side="right") - 1, 0, x.size - 2)
    h = x[j + 1] - x[j]
    t = (xf - x[j]) / h
    t2 = t * t
    t3 = t2 * t
    return ((2 * t3 - 3 * t2 + 1) * y[j]
            + (t3 - 2 * t2 + t) * h * m[j]
            + (-2 * t3 + 3 * t2) * y[j + 1]
            + (t3 - t2) * h * m[j + 1])


def _linea(x: np.ndarray, y: np.ndarray, xf: np.ndarray) -> np.ndarray:
    """PCHIP sobre los puntos FINITOS de `y`, evaluado en `xf`.

    Fuera del tramo que esos puntos cubren se devuelve NaN: la rejilla del barrido
    puede tener huecos (una combinacion evaluada con otras fases, una metrica no
    calculable para ese modelo) y extrapolar ahi seria inventar. Con menos de dos
    puntos finitos no hay nada que interpolar y la linea entera queda a NaN; los
    puntos medidos siguen dibujandose aparte.
    """
    ok = np.isfinite(y)
    if int(ok.sum()) < 2:
        return np.full(xf.shape, np.nan)
    xs, ys = x[ok], y[ok]
    fuera = (xf < xs[0]) | (xf > xs[-1])
    out = _hermite(xs, ys, _pendientes(xs, ys), np.clip(xf, xs[0], xs[-1]))
    out[fuera] = np.nan
    return out


def interpolar(Z: np.ndarray, eje_x: Eje, eje_y: Eje) -> np.ndarray:
    """Estima la metrica en toda la malla fina a partir de los puntos medidos.

    Producto tensorial de PCHIP: primero cada fila medida a lo largo de X, despues
    cada columna resultante a lo largo de Y. Devuelve una matriz
    `[len(eje_y.uf), len(eje_x.uf)]` que coincide con `Z` en los nodos medidos.
    """
    Z = np.asarray(Z, dtype=float)
    if eje_x.uf.size == eje_x.u.size and eje_y.uf.size == eje_y.u.size:
        return Z.copy()                              # densidad 0: la rejilla tal cual

    por_filas = np.empty((Z.shape[0], eje_x.uf.size), dtype=float)
    for i in range(Z.shape[0]):
        por_filas[i] = _linea(eje_x.u, Z[i], eje_x.uf)

    fina = np.empty((eje_y.uf.size, eje_x.uf.size), dtype=float)
    for j in range(eje_x.uf.size):
        fina[:, j] = _linea(eje_y.u, por_filas[:, j], eje_y.uf)
    return fina

"""Distancias entre observaciones: la GEOMETRIA sobre la que se ajusta el modelo.

Hasta ahora habia una sola distancia y no estaba en ninguna funcion: la euclidea
vivia repartida por el nucleo numerico como el truco
``||a||^2 + ||b||^2 - 2 a.b``, escrito a mano en cinco sitios distintos
(seleccion de vecinos de la F2, ancho del kernel RBF, coste de transporte de la
F5, proyeccion fuera de muestra). Este modulo la saca a una funcion, le pone al
lado otras tres —**mahalanobis**, **coseno** y **manhattan**— y las convierte en
una eleccion del modelo, al mismo nivel que la normalizacion del z-score.

## Que decide la distancia

Todo lo que en el pipeline significa «cerca»:

1. **La seleccion de vecinos (kNN) de la Formulacion 2** — que observaciones
   compiten por reconstruir a cada una (`slim.vecinos_mas_cercanos`).
2. **La minimizacion de SLIM** — el residuo ``y - A w`` se mide en esta metrica,
   no siempre en la euclidea (ver «Como entra en el ajuste», abajo).
3. **El kernel de la etapa 1 de la F5** (MMD): tanto el ancho como el propio
   kernel salen de la distancia (`ancho_kernel`, `omega`).
4. **El coste de transporte optimo** de la F5 con Sinkhorn (`coste_ot`).
5. **La proyeccion fuera de muestra** (`foldin`), que tiene que usar la misma
   geometria con la que se ajusto el modelo o no estaria comparando nada.

No decide la `feat_display` del artefacto (sigue siendo la media z-scoreada de
siempre: es para MOSTRAR) ni la diversidad intra-lista de la Fase 5, que se mide
en euclidea a proposito para que siga siendo comparable entre modelos con
distancias distintas.

## El invariante: transformar UNA vez

Tres de las cuatro distancias son euclideas en un espacio transformado, y esa es
la unica razon por la que se pueden meter en un coordinate descent sin reescribir
el nucleo:

| distancia | `transformar(X)` | lo que mide despues |
|---|---|---|
| `euclidea` | identidad | `||a-b||_2` |
| `mahalanobis` | `X @ L`, con `L L^T = Sigma^-1` | `||a-b||_Sigma^-1` |
| `coseno` | normaliza cada fila a norma 1 | la cordal `sqrt(2(1-cos))`, monotona en `1-cos` |
| `manhattan` | identidad | `||a-b||_1` |

De ahi el contrato de este modulo, que hay que respetar para no medir dos veces:
**`Espacio.transformar(X)` se llama una sola vez, al principio del ajuste, y
TODO lo de despues (vecinos, SLIM, RFF, Sinkhorn) trabaja sobre esa `Xt`.** Los
metodos de `Espacio` (`distancias`, `coste_ot`, `ancho_kernel`, `omega`)
esperan filas YA transformadas. Las cuatro funciones publicas de arriba
(`euclidea`, `mahalanobis`, `coseno`, `manhattan`), en cambio, trabajan sobre
vectores crudos: son la definicion de cada distancia, para consultarla o
comprobarla suelta.

## Como entra en el ajuste (y donde deja de ser exacto)

El objetivo de SLIM por columna es ``min_w ||y - A w||^2 + (beta/2)||w||^2 +
l1||w||_1``, con el residuo sobre las d dimensiones de feature. Cambiar la
distancia es cambiar la norma de ese residuo:

- `euclidea`: el de siempre, sin tocar nada.
- `mahalanobis`: ``||y - A w||^2_Sigma^-1 = ||L^T y - L^T A w||^2``. Transformar
  las filas de X **es** aplicar la metrica al residuo, asi que el coordinate
  descent de `slim.elasticnet_no_negativo` resuelve el problema EXACTO sin un
  solo cambio. Lo mismo vale para `coseno` (reconstruccion de los vectores ya
  normalizados: geometria angular).
- `manhattan`: ``||y - A w||_1`` NO es un minimo cuadrado y no tiene
  actualizacion cerrada por coordenada. Se resuelve por **IRLS**
  (`slim.elasticnet_residuo_l1`): unas pocas pasadas de minimos cuadrados
  ponderados con ``u_i = 1/max(|r_i|, eps)``, que es la mayorizacion clasica de
  la norma 1. Es un metodo iterativo, asi que ahi la solucion es aproximada
  (converge, pero no es de forma cerrada) y ademas `beta` y `l1` quedan en otra
  escala efectiva: el termino de datos pasa de cuadratico a lineal.

## Coste

`manhattan` no tiene el truco del producto escalar: la matriz de distancias entre
M observaciones cuesta O(M^2 d) de verdad, mientras que la euclidea se resuelve
con un unico GEMM. En la F5 da igual (el ancho del kernel se estima sobre una
submuestra y el RFF es lineal), pero el kNN de la **F2 sobre jugador** —50.000
observaciones— es del orden de horas. El barrido avisa antes de empezar.

`mahalanobis` cuesta una descomposicion espectral de la covarianza (d x d, con d
~ 60: milisegundos) mas un producto por fila; `coseno`, una division por fila.

## Que rompe mahalanobis (declarado, no es un bug)

El blanqueo reescala TODAS las direcciones a varianza 1, asi que deshace la
ponderacion deliberada del bloque de posicion (`config.POSITION_SCALING =
"zscore_sqrt"`, las 25 columnas `pos_*` divididas por sqrt(25) para que pesen
como una sola feature): despues de blanquear, esas 25 direcciones vuelven a
pesar lo que la covarianza diga. Es coherente con lo que mahalanobis promete
—decorrelacionar y equiparar— pero no es lo mismo que el modelo base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Distancias implementadas. `euclidea` es la de siempre y el default de todo el
# pipeline: un modelo construido sin elegir distancia es identico —numero a
# numero— al que se construia antes de existir este modulo.
DISTANCIAS_VALIDAS: tuple[str, ...] = (
    "euclidea", "mahalanobis", "coseno", "manhattan",
)
POR_DEFECTO = "euclidea"

# Piso RELATIVO (fraccion del autovalor medio) que se suma a los autovalores de
# la covarianza antes de invertirla. Sin el, una direccion casi constante —una
# columna `pos_*` de una posicion que casi no aparece— se amplificaria hasta
# gobernar la distancia entera. Va aqui y no en `config` a proposito: es parte de
# la definicion de la distancia, y como `distancias.py` entra en el hash de
# codigo de la huella, tocarlo invalida la cache igual que tocar el nucleo.
MAHALANOBIS_RIDGE = 1e-3

# Pasadas de IRLS con las que se resuelve el residuo L1 (ver el docstring del
# modulo). Pocas: la mayorizacion converge rapido y cada pasada es un coordinate
# descent completo.
IRLS_ITERS = 8
IRLS_EPS = 1e-6

# Presupuesto de memoria (bytes) del bloque temporal con el que se calculan las
# distancias L1 por pares. La euclidea no lo necesita (es un producto de
# matrices); la manhattan materializa |a_i - b_j| por dimension y sin trocear
# reventaria la memoria en cuanto M pasa de unos miles.
MEMORIA_BLOQUE = 64_000_000


# --------------------------------------------------------------------------- #
# Las cuatro distancias, sobre vectores CRUDOS                                 #
# --------------------------------------------------------------------------- #

def euclidea(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """``||a - b||_2`` entre cada fila de `A` y cada fila de `B` (B=None -> A vs A).

    Se calcula con la identidad ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b``, que
    convierte el problema en un unico producto de matrices (BLAS). Es el truco
    que estaba escrito a mano en `slim`, `distributional` y `foldin`; el negativo
    residual del error de redondeo se recorta a 0 antes de la raiz.

    Ese truco tiene la contrapartida de siempre: entre vectores muy parecidos la
    resta es catastrofica y la distancia sale con ~1e-8 de ruido en vez de 0
    exacto. Da igual para lo que hace el pipeline (ordenar vecinos y estimar una
    mediana), pero conviene saberlo si se compara el valor contra
    `np.linalg.norm(a - b)`.
    """
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    return np.sqrt(np.maximum(_euclidea2(A, B), 0.0))


def _euclidea2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distancia euclidea AL CUADRADO (sin la raiz: ordena igual y es mas barata)."""
    sa = np.einsum("ij,ij->i", A, A)
    sb = np.einsum("ij,ij->i", B, B)
    return sa[:, None] + sb[None, :] - 2.0 * (A @ B.T)


def manhattan(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """``||a - b||_1 = sum_j |a_j - b_j||`` entre cada fila de `A` y cada fila de `B`.

    No hay identidad que la reduzca a un producto de matrices, asi que se
    materializa la diferencia por dimension. Se trocea en dos niveles —filas de
    `A` y bloques de columnas— para que el temporal quepa en `MEMORIA_BLOQUE`:
    el trabajo total es el mismo (O(n_A n_B d)), lo que se acota es el pico.
    """
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    n_a, d = A.shape
    n_b = B.shape[0]
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=float)

    # Columnas por trozo y filas por bloque, repartiendo el presupuesto.
    cols = max(1, min(d, MEMORIA_BLOQUE // max(1, 8 * n_b)))
    filas = max(1, MEMORIA_BLOQUE // max(1, 8 * n_b * cols))
    salida = np.zeros((n_a, n_b), dtype=float)
    for ini in range(0, n_a, filas):
        fin = min(ini + filas, n_a)
        acumulado = salida[ini:fin]
        for c in range(0, d, cols):
            trozo = np.abs(A[ini:fin, None, c:c + cols] - B[None, :, c:c + cols])
            acumulado += trozo.sum(axis=2)
    return salida


def coseno(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """``1 - cos(a, b)``: distancia angular, ciega a la magnitud del vector.

    Sobre features ya z-scoreadas equivale a la distancia de correlacion. Dos
    observaciones con el mismo PERFIL y distinto volumen distan 0: es
    exactamente lo que la hace interesante como alternativa (mide forma) y lo que
    hay que tener presente al leerla (en scouting el volumen tambien es senal).

    Una fila de norma 0 (todo a la media) se deja como esta: su distancia a
    cualquier otra sale 1, que es la neutral.
    """
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    return 1.0 - (_normalizar_filas(A) @ _normalizar_filas(B).T)


def mahalanobis(
    A: np.ndarray, B: np.ndarray | None = None, L: np.ndarray | None = None
) -> np.ndarray:
    """``sqrt((a-b)^T Sigma^-1 (a-b))``, con `L` tal que ``L L^T = Sigma^-1``.

    `L` sale de `blanqueo(X)` sobre los datos del ajuste y hay que CONGELARLO:
    proyectar una entidad nueva con un blanqueo estimado de sus propios datos la
    pondria en otro espacio que las del modelo (mismo motivo por el que las mu/sd
    del z-score y el ancho del kernel viajan con el artefacto).
    """
    if L is None:
        raise ValueError(
            "mahalanobis necesita la raiz de Sigma^-1; calculala con `blanqueo(X)` "
            "sobre los datos del ajuste y congelala con el modelo"
        )
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    return euclidea(A @ L, B @ L)


def _normalizar_filas(X: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(X, axis=1)
    normas[normas == 0.0] = 1.0
    return X / normas[:, None]


def blanqueo(X: np.ndarray, ridge: float = MAHALANOBIS_RIDGE) -> np.ndarray:
    """Raiz simetrica (ZCA) de ``Sigma^-1``: la `L` que consume `mahalanobis`.

    ``Sigma`` es la covarianza de las observaciones. Se diagonaliza con `eigh`
    (es simetrica), se le suma a los autovalores un piso proporcional a su media
    (`ridge`) y se devuelve ``V diag(1/sqrt(lambda + piso)) V^T``.

    Se usa la raiz SIMETRICA y no la de Cholesky ni la de PCA porque las tres dan
    la misma distancia —difieren en una rotacion, y la euclidea es invariante a
    rotaciones— pero la simetrica deja el espacio transformado alineado con las
    features originales, que es lo que hace legibles los pesos de SLIM.

    Con una covarianza degenerada (menos observaciones que features, o todas
    iguales) el piso salva la inversion; si ni asi hay senal, se devuelve la
    identidad, que degrada mahalanobis a euclidea en vez de romper el ajuste.
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    if X.shape[0] < 2 or d == 0:
        return np.eye(d)
    centrada = X - X.mean(axis=0)
    sigma = (centrada.T @ centrada) / (X.shape[0] - 1)
    lam, V = np.linalg.eigh(sigma)
    lam = np.maximum(lam, 0.0)
    media = float(lam.mean())
    if not np.isfinite(media) or media <= 0.0:
        return np.eye(d)
    piso = ridge * media
    return (V * (1.0 / np.sqrt(lam + piso))) @ V.T


# --------------------------------------------------------------------------- #
# Geometria ya ajustada a unos datos                                           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Espacio:
    """Una distancia YA ajustada a unos datos, lista para que la use el pipeline.

    Guarda lo que la distancia haya tenido que aprender de la matriz de features
    (hoy solo el blanqueo de mahalanobis) y expone la operacion que necesita cada
    etapa. Todos sus metodos esperan filas **ya transformadas** con
    `transformar`: ver el invariante en el docstring del modulo.

    Es serializable (`como_dict` / `desde_dict`) porque viaja en el estado warm
    del artefacto: el reentrenamiento incremental y la proyeccion fuera de
    muestra tienen que usar exactamente esta geometria y no una reestimada.
    """

    nombre: str
    L: np.ndarray | None = None      # raiz de Sigma^-1 (solo mahalanobis)

    # --- Identidad ------------------------------------------------------------
    @property
    def euclidea_transformada(self) -> bool:
        """¿La distancia es la euclidea una vez aplicado `transformar`?

        True para `euclidea`, `mahalanobis` y `coseno`; False para `manhattan`.
        Es la propiedad de la que cuelga todo lo demas: con ella, la seleccion de
        vecinos sigue siendo un producto de matrices y el residuo del ajuste
        sigue siendo un minimo cuadrado (el coordinate descent de siempre
        resuelve el problema exacto). Sin ella hay que medir la L1 fila a fila y
        resolver el ajuste por IRLS (`slim.elasticnet_residuo_l1`).
        """
        return self.nombre != "manhattan"

    @property
    def es_identidad(self) -> bool:
        """¿`transformar` deja X tal cual? (atajo para no copiar matrices grandes)."""
        return self.nombre in ("euclidea", "manhattan")

    # --- Cambio de espacio ----------------------------------------------------
    def transformar(self, X: np.ndarray) -> np.ndarray:
        """Lleva las filas de X al espacio donde se mide. Se llama UNA vez."""
        X = np.asarray(X, dtype=float)
        if self.nombre == "mahalanobis":
            if self.L is None:
                raise ValueError("espacio mahalanobis sin blanqueo: usa `preparar`")
            return X @ self.L
        if self.nombre == "coseno":
            return _normalizar_filas(X)
        return X

    # --- Medidas sobre filas YA transformadas ---------------------------------
    def distancias(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        """Matriz de distancias entre filas ya transformadas."""
        if self.nombre == "manhattan":
            return manhattan(A, B)
        A = np.asarray(A, dtype=float)
        B = A if B is None else np.asarray(B, dtype=float)
        return np.sqrt(np.maximum(_euclidea2(A, B), 0.0))

    def distancias2(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        """Lo que ORDENA a los vecinos: d^2 en las euclideas, d en la manhattan.

        No es «la distancia al cuadrado» siempre: en la manhattan elevar al
        cuadrado seria trabajo de mas que no cambia ningun orden. Lo unico que se
        promete es que es monotona creciente en la distancia, que es lo que
        necesitan la seleccion de vecinos y el `argpartition`.
        """
        if self.nombre == "manhattan":
            return manhattan(A, B)
        A = np.asarray(A, dtype=float)
        B = A if B is None else np.asarray(B, dtype=float)
        return _euclidea2(A, B)

    def coste_ot(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Coste de transporte entre dos nubes (etapa 1 de la F5 con Sinkhorn).

        Cuadratico en las euclideas —lo que da la 2-Wasserstein de siempre— y
        lineal en la manhattan, que da la 1-Wasserstein: mas estable cuando una
        nube trae un partido atipico, que es justo el caso de una entidad con
        pocas observaciones.
        """
        return np.maximum(self.distancias2(A, B), 0.0)

    def ancho_kernel(self, Xs: np.ndarray) -> float:
        """Ancho del kernel por la heuristica de la mediana, sobre una submuestra.

        En las euclideas es la formula de siempre —``sqrt(mediana(d^2)/2)`` para
        el RBF—; en la manhattan es la mediana de las distancias L1, que es la
        escala natural del kernel laplaciano ``exp(-||x-y||_1/sigma)``.

        Sin ningun par del que sacar distancias (una sola observacion, o todas
        identicas) la mediana se toma como 1, que es la convencion que tenia el
        calculo original: no se devuelve directamente 1.0, porque en el caso
        euclideo eso saltaria la raiz y daria otro ancho.
        """
        n = Xs.shape[0]
        triu = np.triu_indices(n, k=1)
        if self.nombre == "manhattan":
            d = manhattan(Xs)[triu]
            d = d[d > 0]
            return (float(np.median(d)) if d.size else 1.0) or 1.0
        d2 = _euclidea2(Xs, Xs)[triu]
        d2 = d2[d2 > 0]
        med = np.median(d2) if d2.size else 1.0
        return float(np.sqrt(med / 2.0)) or 1.0

    def omega(
        self, d: int, dim: int, sigma: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Frecuencias del mapa Random Fourier del kernel que toca a esta distancia.

        Por el teorema de Bochner, un kernel invariante a traslaciones se aproxima
        con ``sqrt(2/D) cos(x.Omega + b)`` muestreando Omega de su densidad
        espectral. Cambia la densidad, no el mapa:

        - euclideas -> kernel RBF ``exp(-||x-y||^2 / 2 sigma^2)``, Omega gaussiana
          (lo de siempre, con la misma llamada al generador para que el resultado
          sea identico al de antes).
        - manhattan -> kernel **laplaciano** ``exp(-||x-y||_1 / sigma)``, cuya
          densidad espectral es una Cauchy. Es el kernel que corresponde a la L1,
          y ademas decae mas despacio que el RBF, que en dimension alta satura.
        """
        if self.nombre == "manhattan":
            return rng.standard_cauchy(size=(d, dim)) / sigma
        return rng.normal(0.0, 1.0 / sigma, size=(d, dim))

    # --- Serializacion --------------------------------------------------------
    def como_dict(self) -> dict:
        return {"distancia": self.nombre,
                "L": None if self.L is None else np.asarray(self.L).tolist()}

    @classmethod
    def desde_dict(cls, datos: dict | None) -> "Espacio":
        if not datos:
            return Espacio(POR_DEFECTO)
        L = datos.get("L")
        return cls(nombre=str(datos.get("distancia") or POR_DEFECTO),
                   L=None if L is None else np.asarray(L, dtype=float))


def validar(distancia: str | None) -> str:
    """Normaliza y valida el nombre de una distancia. None -> la de por defecto."""
    nombre = POR_DEFECTO if distancia is None else str(distancia)
    if nombre not in DISTANCIAS_VALIDAS:
        raise ValueError(
            f"distancia desconocida: {distancia!r} (usa {DISTANCIAS_VALIDAS})")
    return nombre


def preparar(X: np.ndarray, distancia: str | None = POR_DEFECTO) -> Espacio:
    """`Espacio` de esa distancia ajustado a `X` (lo que haya que aprender de ella).

    Hoy solo mahalanobis aprende algo (el blanqueo); las otras tres son
    parametricas y su `Espacio` no depende de los datos. Aun asi todas pasan por
    aqui, para que quien construye no tenga que saber cual es cual.
    """
    nombre = validar(distancia)
    if nombre == "mahalanobis":
        return Espacio(nombre, L=blanqueo(np.asarray(X, dtype=float)))
    return Espacio(nombre)


def vecinos(
    espacio: Espacio,
    base: np.ndarray,
    consulta: np.ndarray,
    k: int,
    propia: np.ndarray | None = None,
    paso: int = 512,
) -> np.ndarray:
    """Indices de las `k` filas de `base` mas cercanas a cada fila de `consulta`.

    Las dos matrices tienen que venir YA transformadas (ver el invariante del
    modulo). `propia[i]` es la fila de `base` que ES la consulta `i` (-1 si no
    esta ahi) y se excluye: es el ``w_ss = 0`` de SLIM.

    Se recorre la consulta por bloques porque la matriz completa es
    (n_consulta x n_base) y con las ~50.000 observaciones de la BD no cabe en
    memoria. Devuelve los `k` mas cercanos SIN ordenar entre si (`argpartition`),
    que es lo que necesita el diccionario de candidatas.
    """
    n_base, n_consulta = base.shape[0], consulta.shape[0]
    k = min(k, n_base - (0 if propia is None else 1))
    if k <= 0:
        return np.empty((n_consulta, 0), dtype=np.int64)
    salida = np.empty((n_consulta, k), dtype=np.int64)
    for ini in range(0, n_consulta, paso):
        fin = min(ini + paso, n_consulta)
        d2 = espacio.distancias2(consulta[ini:fin], base)
        if propia is not None:
            for r in range(fin - ini):
                if propia[ini + r] >= 0:
                    d2[r, propia[ini + r]] = np.inf
        if k == d2.shape[1]:
            salida[ini:fin] = np.argsort(d2, axis=1)
        else:
            salida[ini:fin] = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    return salida

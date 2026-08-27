"""Etapa 1 de la Formulacion 5: similitud distribucional entidad-entidad.

Cada entidad es la NUBE de sus observaciones por-partido. Se compara nube contra
nube SIN promediar las features crudas, obteniendo una matriz S (P x P) que luego
alimenta a EASE (etapa 2). Dos metodos, ambos en numpy puro:

- `mmd`: Maximum Mean Discrepancy via kernel mean embedding con Random Fourier
  Features (RBF). El embedding de una entidad es la media de phi(x) sobre sus
  observaciones, donde phi es un mapa NO LINEAL: eso representa la DISTRIBUCION
  (conserva varianza/multimodalidad), no es un colapso de features. Escala a
  miles de entidades (S = coseno de los mu, ver `similitud_mmd`). Robusto con 1
  sola observacion.
- `sinkhorn`: transporte optimo entropico exacto (log-domain) entre las dos
  nubes. Barato cuando hay pocas entidades (equipos). Preferido por el PDF para
  robustez con pocos partidos frente a Bures (que exige >= d+1 partidos).

**La distancia entre observaciones es un parametro** (`src.similitud.distancias`)
y las dos etapas la respetan: el kernel del MMD es el que le corresponde a esa
distancia —RBF para las euclideas, laplaciano para la manhattan, via la densidad
espectral que muestrea `Espacio.omega`— y el coste del Sinkhorn es su
`coste_ot` (cuadratico -> W2 en las euclideas, lineal -> W1 en la manhattan).
Convenio de este modulo: **las funciones publicas reciben la `MatrizFeatures`
CRUDA y transforman ellas mismas**; las privadas (`_rff`, `_sinkhorn_costo`)
esperan filas ya transformadas.
"""

from __future__ import annotations

import numpy as np

from . import config
from .distancias import Espacio, POR_DEFECTO as DISTANCIA_POR_DEFECTO
from .features import MatrizFeatures


def _espacio(espacio: Espacio | None) -> Espacio:
    """El espacio pedido, o el euclideo (que deja todo como estaba)."""
    return espacio if espacio is not None else Espacio(DISTANCIA_POR_DEFECTO)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return np.squeeze(out, axis=axis)


def _sigma_mediana(
    X: np.ndarray, seed: int, muestra: int = 2000, espacio: Espacio | None = None
) -> float:
    """Ancho del kernel por la heuristica de la mediana de distancias.

    `X` viene YA transformada. La mediana se mide en la distancia del espacio
    (`Espacio.ancho_kernel`): con la euclidea es el calculo de siempre, y con la
    manhattan la mediana de las L1, que es la escala del kernel laplaciano.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(muestra, n), replace=False)
    return _espacio(espacio).ancho_kernel(X[idx])


def _rff(
    X: np.ndarray, dim: int, sigma: float, seed: int,
    espacio: Espacio | None = None,
) -> np.ndarray:
    """Random Fourier Features: z(x) ~ tal que z(x).z(y) ~ k(x,y). `X` transformada.

    El mapa es el mismo para cualquier distancia —``sqrt(2/D) cos(x.Omega + b)``,
    Bochner— y lo unico que cambia es de que densidad se muestrea Omega, que es
    lo que decide QUE kernel se esta aproximando (`Espacio.omega`: gaussiana para
    el RBF de las euclideas, Cauchy para el laplaciano de la manhattan).
    """
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    Omega = _espacio(espacio).omega(d, dim, sigma, rng)
    b = rng.uniform(0.0, 2.0 * np.pi, size=dim)
    return np.sqrt(2.0 / dim) * np.cos(X @ Omega + b[None, :])


def _embed_entidades(
    Z: np.ndarray, ent_idx: np.ndarray, weight: np.ndarray, n_ent: int
) -> np.ndarray:
    """Kernel mean embedding por entidad: media ponderada por minutos de phi(x)."""
    dim = Z.shape[1]
    suma = np.zeros((n_ent, dim), dtype=float)
    masa = np.zeros(n_ent, dtype=float)
    np.add.at(suma, ent_idx, Z * weight[:, None])
    np.add.at(masa, ent_idx, weight)
    masa[masa == 0.0] = 1.0
    return suma / masa[:, None]


def sigma_kernel(X: np.ndarray, espacio: Espacio | None = None) -> float:
    """Ancho del kernel que usaria un ajuste en frio sobre `X` (matriz CRUDA).

    Expuesto aparte para poder CONGELARLO al reentrenar: el ancho sale de la
    mediana de distancias de todo el dataset, asi que anadir partidos lo mueve y
    con el se mueve el embedding de TODAS las entidades, incluidas las que no han
    cambiado. Reutilizando el sigma anterior, las entidades antiguas conservan su
    embedding exacto y las nuevas entran en el mismo espacio.

    El ancho depende de la DISTANCIA (se mide en ella), asi que un sigma
    congelado solo vale para el espacio en el que se estimo. Lo garantiza
    `warm.compatible`: la distancia esta en los hiperparametros del estado.
    """
    esp = _espacio(espacio)
    return _sigma_mediana(esp.transformar(X), config.F5_RFF_SEED, espacio=esp)


def similitud_mmd(
    mf: MatrizFeatures, ent_idx: np.ndarray, n_ent: int, sigma: float | None = None,
    espacio: Espacio | None = None,
) -> np.ndarray:
    """S[a,b] = <mu_a, mu_b> / (||mu_a|| ||mu_b||): COSENO entre kernel mean embeddings.

    El producto interno crudo <mu_a, mu_b> escala con ||mu_a||*||mu_b||, y
    ||mu_a||^2 = E[k(x,x')] sobre la propia nube = cuan CONCENTRADA es la
    distribucion del jugador (consistencia partido a partido). Eso inflaba a los
    perfiles compactos como similares a todos, mezclando "parecido de forma" con
    "consistencia bruta". Normalizando por las normas nos quedamos solo con el
    ANGULO entre embeddings = parecido de FORMA de las distribuciones. Con kernel
    RBF, phi tiene componentes >0, asi que el coseno queda en ~[0,1] (mismo orden
    de magnitud que el producto crudo, que ya estaba acotado por k<=1: EASE no
    necesita recalibrar lambda). Diagonal a 0 (se ignora al servir).

    ``sigma`` fija el ancho del kernel en vez de estimarlo de `mf.X` (ver
    `sigma_kernel`); None = estimarlo, que es el ajuste en frio de siempre.
    ``espacio`` es la distancia con la que se mide (None = euclidea).
    """
    esp = _espacio(espacio)
    Xt = esp.transformar(mf.X)
    if sigma is None:
        sigma = _sigma_mediana(Xt, config.F5_RFF_SEED, espacio=esp)
    Z = _rff(Xt, config.F5_RFF_DIM, sigma, config.F5_RFF_SEED, espacio=esp)
    mu = _embed_entidades(Z, ent_idx, mf.weight, n_ent)
    norms = np.linalg.norm(mu, axis=1)
    norms[norms == 0.0] = 1.0  # entidad sin masa (no deberia ocurrir): evita 0/0
    mu = mu / norms[:, None]
    S = mu @ mu.T
    np.fill_diagonal(S, 0.0)
    return S


def _sinkhorn_costo(
    Xa: np.ndarray, wa: np.ndarray, Xb: np.ndarray, wb: np.ndarray,
    reg: float, iters: int, espacio: Espacio | None = None,
) -> float:
    """Coste de transporte optimo entropico entre dos nubes (log-domain).

    `Xa`/`Xb` vienen YA transformadas; la matriz de coste la fija la distancia
    (`Espacio.coste_ot`): cuadratica en las euclideas —la 2-Wasserstein de
    siempre— y lineal en la manhattan, que da la 1-Wasserstein.
    """
    C = _espacio(espacio).coste_ot(Xa, Xb)
    a = wa / wa.sum()
    b = wb / wb.sum()
    logK = -C / reg
    u = np.zeros(len(a))
    v = np.zeros(len(b))
    log_a, log_b = np.log(a), np.log(b)
    for _ in range(iters):
        u = log_a - _logsumexp(logK + v[None, :], axis=1)
        v = log_b - _logsumexp(logK + u[:, None], axis=0)
    logP = logK + u[:, None] + v[None, :]
    return float(np.sum(np.exp(logP) * C))


def costos_sinkhorn(
    mf: MatrizFeatures,
    ent_idx: np.ndarray,
    n_ent: int,
    progreso=None,
    previos: np.ndarray | None = None,
    espacio: Espacio | None = None,
) -> np.ndarray:
    """Matriz (n_ent x n_ent) de coste de transporte optimo entre cada par de nubes.

    ``previos`` (misma forma, NaN donde no se sabe) permite REUTILIZAR costes de
    un ajuste anterior: los pares con valor finito no se recalculan. Es
    reutilizacion exacta, no una aproximacion — el coste OT entre dos nubes
    depende solo de esas dos nubes, asi que si ninguna ha cambiado el numero es
    identico. Quien lo pasa es responsable de poner NaN en todo par que involucre
    una nube tocada (ver `src.incremental`).

    ``progreso`` (opcional): callable ``progreso(hecho, total)`` invocado por cada
    entidad del bucle externo (O(n_ent^2) pares); permite seguir el avance.
    """
    esp = _espacio(espacio)
    X = esp.transformar(mf.X)   # una sola vez para todas las nubes
    filas_por_ent = [np.where(ent_idx == e)[0] for e in range(n_ent)]
    reg, iters = config.F5_SINKHORN_REG, config.F5_SINKHORN_ITERS
    costo = np.zeros((n_ent, n_ent), dtype=float)
    for a in range(n_ent):
        Ra = filas_por_ent[a]
        Xa, wa = X[Ra], mf.weight[Ra]
        for b in range(a + 1, n_ent):
            if previos is not None and np.isfinite(previos[a, b]):
                c = float(previos[a, b])
            else:
                Rb = filas_por_ent[b]
                c = _sinkhorn_costo(Xa, wa, X[Rb], mf.weight[Rb], reg, iters,
                                    espacio=esp)
            costo[a, b] = costo[b, a] = c
        if progreso is not None:
            progreso(a + 1, n_ent)
    return costo


def kernel_desde_costos(costo: np.ndarray) -> np.ndarray:
    """Coste OT -> similitud. Kernel gaussiano con ancho = mediana de los costes.

    En ~30 dimensiones estandarizadas el coste vale decenas y `exp(-coste)` haria
    underflow (toda S ~ 0, sin contraste); la mediana lo lleva a un rango util.
    La mediana se recalcula siempre sobre los costes vigentes: es una escala
    global, asi que anadir entidades la mueve un poco aunque los costes antiguos
    sean identicos (efecto monotono, no altera el orden de un top-k).
    """
    n_ent = costo.shape[0]
    triu = costo[np.triu_indices(n_ent, k=1)]
    escala = np.median(triu) if triu.size else 1.0
    escala = escala or 1.0
    S = np.exp(-costo / escala)
    np.fill_diagonal(S, 0.0)
    return S


def similitud_sinkhorn(
    mf: MatrizFeatures,
    ent_idx: np.ndarray,
    n_ent: int,
    progreso=None,
    previos: np.ndarray | None = None,
    espacio: Espacio | None = None,
) -> np.ndarray:
    """S[a,b] = exp(-OT(nube_a, nube_b)/escala); OT entropico exacto por pares."""
    costo = costos_sinkhorn(mf, ent_idx, n_ent, progreso=progreso,
                            previos=previos, espacio=espacio)
    return kernel_desde_costos(costo)


def construir_S(
    mf: MatrizFeatures,
    ent_idx: np.ndarray,
    n_ent: int,
    metodo: str,
    progreso=None,
    sigma: float | None = None,
    costos_previos: np.ndarray | None = None,
    espacio: Espacio | None = None,
) -> np.ndarray:
    """Matriz de similitud distribucional P x P segun el metodo elegido.

    ``progreso`` solo aplica al metodo ``sinkhorn`` (bucle costoso); ``mmd`` esta
    vectorizado y no lo necesita. ``sigma`` (mmd) y ``costos_previos`` (sinkhorn)
    son las dos vias de reaprovechar un ajuste anterior; None en ambas = frio.
    ``espacio`` es la distancia entre observaciones (None = euclidea).
    """
    if metodo == "mmd":
        return similitud_mmd(mf, ent_idx, n_ent, sigma=sigma, espacio=espacio)
    if metodo == "sinkhorn":
        return similitud_sinkhorn(
            mf, ent_idx, n_ent, progreso=progreso, previos=costos_previos,
            espacio=espacio)
    raise ValueError(f"metodo distribucional desconocido: {metodo!r}")

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
"""

from __future__ import annotations

import numpy as np

from . import config
from .features import MatrizFeatures


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return np.squeeze(out, axis=axis)


def _sigma_mediana(X: np.ndarray, seed: int, muestra: int = 2000) -> float:
    """Ancho del kernel RBF por la heuristica de la mediana de distancias."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(muestra, n), replace=False)
    Xs = X[idx]
    sq = np.einsum("ij,ij->i", Xs, Xs)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)
    d2 = d2[np.triu_indices(len(Xs), k=1)]
    d2 = d2[d2 > 0]
    med = np.median(d2) if d2.size else 1.0
    return float(np.sqrt(med / 2.0)) or 1.0


def _rff(X: np.ndarray, dim: int, sigma: float, seed: int) -> np.ndarray:
    """Random Fourier Features del kernel RBF: z(x) ~ tal que z(x).z(y) ~ k(x,y)."""
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    Omega = rng.normal(0.0, 1.0 / sigma, size=(d, dim))
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


def similitud_mmd(
    mf: MatrizFeatures, ent_idx: np.ndarray, n_ent: int
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
    """
    sigma = _sigma_mediana(mf.X, config.F5_RFF_SEED)
    Z = _rff(mf.X, config.F5_RFF_DIM, sigma, config.F5_RFF_SEED)
    mu = _embed_entidades(Z, ent_idx, mf.weight, n_ent)
    norms = np.linalg.norm(mu, axis=1)
    norms[norms == 0.0] = 1.0  # entidad sin masa (no deberia ocurrir): evita 0/0
    mu = mu / norms[:, None]
    S = mu @ mu.T
    np.fill_diagonal(S, 0.0)
    return S


def _sinkhorn_costo(
    Xa: np.ndarray, wa: np.ndarray, Xb: np.ndarray, wb: np.ndarray,
    reg: float, iters: int,
) -> float:
    """Coste de transporte optimo entropico entre dos nubes (log-domain)."""
    sa = np.einsum("ij,ij->i", Xa, Xa)
    sb = np.einsum("ij,ij->i", Xb, Xb)
    C = sa[:, None] + sb[None, :] - 2.0 * (Xa @ Xb.T)
    C = np.maximum(C, 0.0)
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


def similitud_sinkhorn(
    mf: MatrizFeatures, ent_idx: np.ndarray, n_ent: int, progreso=None
) -> np.ndarray:
    """S[a,b] = exp(-OT(nube_a, nube_b)); OT entropico exacto por pares.

    ``progreso`` (opcional): callable ``progreso(hecho, total)`` invocado por cada
    entidad del bucle externo (O(n_ent^2) pares de transporte optimo); permite
    seguir el avance. None (por defecto) no hace nada.
    """
    filas_por_ent = [np.where(ent_idx == e)[0] for e in range(n_ent)]
    reg, iters = config.F5_SINKHORN_REG, config.F5_SINKHORN_ITERS
    costo = np.zeros((n_ent, n_ent), dtype=float)
    for a in range(n_ent):
        Ra = filas_por_ent[a]
        Xa, wa = mf.X[Ra], mf.weight[Ra]
        for b in range(a + 1, n_ent):
            Rb = filas_por_ent[b]
            c = _sinkhorn_costo(Xa, wa, mf.X[Rb], mf.weight[Rb], reg, iters)
            costo[a, b] = costo[b, a] = c
        if progreso is not None:
            progreso(a + 1, n_ent)
    # Kernel gaussiano sobre el coste OT con ancho = mediana de los costes: en
    # ~30 dimensiones estandarizadas el coste vale decenas y exp(-coste) haria
    # underflow (toda S ~ 0, sin contraste); la mediana lo lleva a un rango util.
    triu = costo[np.triu_indices(n_ent, k=1)]
    escala = np.median(triu) if triu.size else 1.0
    escala = escala or 1.0
    S = np.exp(-costo / escala)
    np.fill_diagonal(S, 0.0)
    return S


def construir_S(
    mf: MatrizFeatures, ent_idx: np.ndarray, n_ent: int, metodo: str, progreso=None
) -> np.ndarray:
    """Matriz de similitud distribucional P x P segun el metodo elegido.

    ``progreso`` solo aplica al metodo ``sinkhorn`` (bucle costoso); ``mmd`` esta
    vectorizado y no lo necesita.
    """
    if metodo == "mmd":
        return similitud_mmd(mf, ent_idx, n_ent)
    if metodo == "sinkhorn":
        return similitud_sinkhorn(mf, ent_idx, n_ent, progreso=progreso)
    raise ValueError(f"metodo distribucional desconocido: {metodo!r}")

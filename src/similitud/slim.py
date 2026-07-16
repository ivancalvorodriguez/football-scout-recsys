"""Nucleo SLIM: piezas aisladas y sustituibles del ajuste.

Todo en numpy puro (no hay sklearn/scipy en el entorno; el enunciado lo permite
y el propio PDF describe el metodo: coordinate descent + soft-thresholding,
glmnet/Friedman-Hastie-Tibshirani 2010).

Contiene:
- `elasticnet_no_negativo`: regresion dispersa L1+L2 con w>=0 por coordenadas,
  el ladrillo comun de SLIM (una columna) y de la variante feature-space.
- `vecinos_mas_cercanos`: seleccion de candidatos tipo fsSLIM (kNN euclideo) que
  hace tratable la W (M x M) de la Formulacion 2.
- `slim_instancia`: aprende W instancia-instancia (Formulacion 2). Reconstruye
  cada OBSERVACION como combinacion dispersa no-negativa de sus vecinas.
- `agregar_W_a_entidades`: agrega los bloques de W a similitud entidad-entidad
  (la agregacion cae sobre los PESOS aprendidos, no sobre el input).
- `ease`: SLIM de forma cerrada (Steck 2019) usado como capa de re-ranking en la
  Formulacion 5.
"""

from __future__ import annotations

import numpy as np


def elasticnet_no_negativo(
    A: np.ndarray,
    y: np.ndarray,
    beta: float,
    l1: float,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> np.ndarray:
    """Resuelve  min_w (1/2)||y - A w||^2 + (beta/2)||w||^2 + l1*||w||_1,  w>=0.

    Coordinate descent con soft-thresholding (parte positiva por la restriccion
    w>=0). `A` es (n_muestras, n_vars), `y` es (n_muestras,). Devuelve w (n_vars,).
    Esta es exactamente la subrutina por columna de SLIM (Ec. 4 del paper).
    """
    n_vars = A.shape[1]
    w = np.zeros(n_vars, dtype=float)
    if n_vars == 0:
        return w
    # Norma al cuadrado de cada columna + ridge (denominador de la actualizacion).
    col_sq = np.einsum("ij,ij->j", A, A) + beta
    col_sq[col_sq == 0.0] = 1.0
    residual = y - A @ w  # = y (w=0)
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(n_vars):
            w_j = w[j]
            a_j = A[:, j]
            # rho = a_j . (residual + w_j a_j) = correlacion parcial con y.
            rho = a_j @ residual + w_j * col_sq[j] - beta * w_j
            new_wj = max(0.0, rho - l1) / col_sq[j]  # soft-threshold + no-neg
            delta = new_wj - w_j
            if delta != 0.0:
                residual -= delta * a_j
                w[j] = new_wj
                max_delta = max(max_delta, abs(delta))
        if max_delta < tol:
            break
    return w


def vecinos_mas_cercanos(X: np.ndarray, k: int) -> np.ndarray:
    """Indices de los k vecinos euclideos mas cercanos de cada fila (sin la propia).

    Seleccion de candidatos estilo fsSLIM: cada observacion solo se reconstruye
    desde sus ~k vecinas, evitando resolver contra las M observaciones. Se calcula
    por bloques para no materializar la matriz M x M completa.
    """
    M = X.shape[0]
    k = min(k, M - 1)
    sq = np.einsum("ij,ij->i", X, X)  # ||x_i||^2
    vecinos = np.empty((M, k), dtype=np.int64)
    paso = 512
    for ini in range(0, M, paso):
        fin = min(ini + paso, M)
        # d^2(i,j) = ||x_i||^2 + ||x_j||^2 - 2 x_i.x_j  (constante ||x_i||^2 no ordena)
        d2 = sq[None, :] - 2.0 * (X[ini:fin] @ X.T)
        for r in range(fin - ini):
            d2[r, ini + r] = np.inf  # excluir la propia observacion
        idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
        vecinos[ini:fin] = idx
    return vecinos


def slim_instancia(
    X: np.ndarray,
    n_neighbors: int,
    beta: float,
    l1: float,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Formulacion 2: aprende W (M x M) instancia-instancia.

    Variante feature-space del PDF: las COLUMNAS a reconstruir son las
    observaciones (matriz d x M con features en filas). Cada observacion `s`
    (vector de d features) se reconstruye como combinacion dispersa no-negativa
    de sus vecinas: min ||x_s - sum_r w_rs x_r||^2 + reg, con w_ss = 0.

    `W_rs` = cuanto contribuye la observacion `r` a reconstruir la `s` = similitud
    APRENDIDA observacion-observacion. Se devuelve dispersa por columnas: para cada
    `s`, (indices de filas r con peso, pesos). Las "muestras" del minimo-cuadrados
    son las d dimensiones de feature; por eso la ponderacion por minutos no entra
    aqui como sample_weight, sino en la agregacion (ver `agregar_W_a_entidades`,
    opcion (b) del PDF).
    """
    M = X.shape[0]
    vecinos = vecinos_mas_cercanos(X, n_neighbors)
    cols_idx: list[np.ndarray] = []
    cols_val: list[np.ndarray] = []
    for s in range(M):
        cand = vecinos[s]                 # ya excluye s (w_ss = 0)
        A = X[cand].T                     # (d, n_cand): features en filas
        y = X[s]                          # (d,): observacion objetivo
        w = elasticnet_no_negativo(A, y, beta=beta, l1=l1, max_iter=max_iter, tol=tol)
        nz = w > 0.0
        cols_idx.append(cand[nz].astype(np.int64))
        cols_val.append(w[nz])
    return cols_idx, cols_val


def agregar_W_a_entidades(
    cols_idx: list[np.ndarray],
    cols_val: list[np.ndarray],
    row_entity: np.ndarray,
    weight: np.ndarray,
    n_entities: int,
) -> np.ndarray:
    """Agrega bloques de W a similitud entidad-entidad (Formulacion 2).

    S[a,b] = (1/Z_ab) * sum_{r en a} sum_{s en b} m_r * m_s * W_rs, con
    m = masa por minutos y Z_ab = (sum_{r en a} m_r) * (sum_{s en b} m_s). Dividir
    por Z neutraliza que un jugador con muchos partidos aporte mas masa (el PDF:
    "con normalizacion por N_p, no domina"). La agregacion cae SOBRE W (pesos
    aprendidos), nunca sobre las features de entrada: ese es el principio central.

    `row_entity[i]` = indice de entidad (0..n_entities-1) de la observacion i.
    Se recorre W por columnas (coste O(nnz)).
    """
    num = np.zeros((n_entities, n_entities), dtype=float)
    # Masa total por entidad (denominador Z).
    masa = np.zeros(n_entities, dtype=float)
    np.add.at(masa, row_entity, weight)
    masa_prod = np.outer(masa, masa)
    masa_prod[masa_prod == 0.0] = 1.0

    for s, (idx, val) in enumerate(zip(cols_idx, cols_val)):
        if idx.size == 0:
            continue
        b = row_entity[s]
        m_s = weight[s]
        a = row_entity[idx]
        contrib = weight[idx] * m_s * val
        np.add.at(num[:, b], a, contrib)

    S = num / masa_prod
    # Similitud simetrizada: la Formulacion 2 aprende una W asimetrica
    # (contribucion direccional); se promedia para un ranking estable.
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    return S


def ease(S: np.ndarray, lam: float) -> np.ndarray:
    """EASE^R (Steck 2019): SLIM de forma cerrada. min_B ||S - S B||^2 + lam||B||^2.

    Solucion: P = (S^T S + lam I)^-1;  B = -P / diag(P);  diag(B) = 0. Usado como
    capa de re-ranking aprendido en la Formulacion 5 sobre la matriz de similitud
    entidad-entidad ya calculada en la etapa distribucional.
    """
    G = S.T @ S
    n = G.shape[0]
    P = np.linalg.inv(G + lam * np.eye(n))
    diag = np.diag(P).copy()
    diag[diag == 0.0] = 1e-12
    B = -P / diag[None, :]
    np.fill_diagonal(B, 0.0)
    return B

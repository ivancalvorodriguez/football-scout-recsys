"""Nucleo SLIM: piezas aisladas y sustituibles del ajuste.

Todo en numpy puro (no hay sklearn/scipy en el entorno; el enunciado lo permite
y el propio PDF describe el metodo: coordinate descent + soft-thresholding,
glmnet/Friedman-Hastie-Tibshirani 2010).

Contiene:
- `elasticnet_no_negativo`: regresion dispersa L1+L2 con w>=0 por coordenadas,
  el ladrillo comun de SLIM (una columna) y de la variante feature-space.
- `elasticnet_residuo_l1`: la misma regresion con el residuo medido en norma 1
  (IRLS sobre la anterior), que es lo que pide la distancia `manhattan`.
- `vecinos_mas_cercanos`: seleccion de candidatos tipo fsSLIM (kNN) que hace
  tratable la W (M x M) de la Formulacion 2.
- `slim_instancia`: aprende W instancia-instancia (Formulacion 2). Reconstruye
  cada OBSERVACION como combinacion dispersa no-negativa de sus vecinas.
- `agregar_W_a_entidades`: agrega los bloques de W a similitud entidad-entidad
  (la agregacion cae sobre los PESOS aprendidos, no sobre el input).
- `ease`: SLIM de forma cerrada (Steck 2019) usado como capa de re-ranking en la
  Formulacion 5.

**La distancia es un parametro.** Tanto la seleccion de vecinos como la norma del
residuo salen de `src.similitud.distancias`: sin `espacio` (o con el euclideo)
esto hace exactamente lo que hacia antes, y con otro se ajusta en su geometria.
Ver el docstring de aquel modulo, sobre todo el invariante de transformar una
sola vez: aqui `slim_instancia` es quien lo hace, y todo lo que llama despues
trabaja ya sobre la matriz transformada.
"""

from __future__ import annotations

import numpy as np

from . import distancias
from .distancias import Espacio


def elasticnet_no_negativo(
    A: np.ndarray,
    y: np.ndarray,
    beta: float,
    l1: float,
    max_iter: int = 200,
    tol: float = 1e-4,
    w_init: np.ndarray | None = None,
) -> np.ndarray:
    """Resuelve  min_w (1/2)||y - A w||^2 + (beta/2)||w||^2 + l1*||w||_1,  w>=0.

    Coordinate descent con soft-thresholding (parte positiva por la restriccion
    w>=0). `A` es (n_muestras, n_vars), `y` es (n_muestras,). Devuelve w (n_vars,).
    Esta es exactamente la subrutina por columna de SLIM (Ec. 4 del paper).

    ``w_init`` es el punto de arranque (WARM START): la solucion de un ajuste
    anterior sobre datos parecidos. Con ``beta > 0`` el objetivo es estrictamente
    convexo, asi que el optimo es UNICO y arrancar mas cerca solo ahorra
    iteraciones — no cambia la solucion (mas alla de `tol`). Los negativos se
    recortan a 0 para no arrancar fuera de la region factible.
    """
    n_vars = A.shape[1]
    if w_init is None:
        w = np.zeros(n_vars, dtype=float)
    else:
        w = np.maximum(np.asarray(w_init, dtype=float), 0.0)
        if w.shape != (n_vars,):
            raise ValueError(
                f"w_init tiene forma {w.shape} y se esperaba ({n_vars},)")
    if n_vars == 0:
        return w
    # Norma al cuadrado de cada columna + ridge (denominador de la actualizacion).
    col_sq = np.einsum("ij,ij->j", A, A) + beta
    col_sq[col_sq == 0.0] = 1.0
    residual = y - A @ w  # = y cuando se arranca en frio (w=0)
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


def elasticnet_residuo_l1(
    A: np.ndarray,
    y: np.ndarray,
    beta: float,
    l1: float,
    max_iter: int = 200,
    tol: float = 1e-4,
    w_init: np.ndarray | None = None,
    iters_irls: int = distancias.IRLS_ITERS,
    eps: float = distancias.IRLS_EPS,
) -> np.ndarray:
    """Resuelve  min_w ||y - A w||_1 + (beta/2)||w||^2 + l1*||w||_1,  w>=0.

    Es `elasticnet_no_negativo` con el residuo medido en norma 1 en vez de en
    norma 2, que es lo que significa ajustar con la distancia `manhattan`: el
    residuo se mide en la misma metrica en la que se eligen los vecinos.

    No hay actualizacion cerrada por coordenada para la norma 1, asi que se
    resuelve por **IRLS** (iteratively reweighted least squares): |r| se mayoriza
    con r^2/(2|r|), de modo que cada pasada es un minimo cuadrado PONDERADO con
    pesos ``u_i = 1/max(|r_i|, eps)``, y eso ya lo sabe hacer el coordinate
    descent de siempre sin tocarlo — basta con escalar las filas por sqrt(u).
    Cada pasada arranca de la anterior, asi que las ultimas son casi gratis.

    Dos consecuencias que conviene tener presentes:

    - La solucion es ITERATIVA: converge (la mayorizacion no aumenta el
      objetivo), pero a diferencia del caso euclideo no es el optimo exacto de
      una funcion estrictamente convexa resuelta hasta `tol`.
    - `beta` y `l1` quedan en OTRA escala efectiva, porque el termino de datos
      pasa de cuadratico a lineal. Los valores afinados para la euclidea no son
      directamente trasladables: hay que barrerlos de nuevo.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    w = elasticnet_no_negativo(A, y, beta=beta, l1=l1, max_iter=max_iter,
                               tol=tol, w_init=w_init)
    for _ in range(max(0, iters_irls - 1)):
        residual = y - A @ w
        raiz_u = 1.0 / np.sqrt(np.maximum(np.abs(residual), eps))
        nueva = elasticnet_no_negativo(
            A * raiz_u[:, None], y * raiz_u,
            beta=beta, l1=l1, max_iter=max_iter, tol=tol, w_init=w)
        cambio = float(np.max(np.abs(nueva - w))) if nueva.size else 0.0
        w = nueva
        if cambio < tol:
            break
    return w


def vecinos_mas_cercanos(
    X: np.ndarray, k: int, espacio: Espacio | None = None
) -> np.ndarray:
    """Indices de los k vecinos mas cercanos de cada fila (sin la propia).

    Seleccion de candidatos estilo fsSLIM: cada observacion solo se reconstruye
    desde sus ~k vecinas, evitando resolver contra las M observaciones. Se calcula
    por bloques para no materializar la matriz M x M completa.

    `X` tiene que venir YA en el espacio de la distancia (`Espacio.transformar`);
    `espacio` solo dice con que metrica se mide ahi. Con None —o con cualquier
    distancia que sea euclidea tras transformar: `euclidea`, `mahalanobis`,
    `coseno`— se usa la via rapida de siempre, un unico producto de matrices por
    bloque. La `manhattan` no tiene esa identidad y va por
    `distancias.vecinos`, que mide fila a fila.
    """
    if espacio is not None and not espacio.euclidea_transformada:
        return distancias.vecinos(
            espacio, X, X, k, propia=np.arange(X.shape[0], dtype=np.int64))
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
    progreso=None,
    inicial=None,
    espacio: Espacio | None = None,
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

    ``inicial`` (opcional) es el WARM START: un callable ``inicial(s, cand)`` que
    devuelve el vector de arranque para la columna `s` alineado con `cand`, o
    None para arrancar en frio esa columna. La traduccion entre los indices del
    ajuste anterior y los de ahora (las observaciones se renumeran al anadir
    partidos, y los vecinos de cada una pueden cambiar) es responsabilidad de
    quien lo pasa — aqui no se guarda estado de ejecuciones previas.

    ``espacio`` es la DISTANCIA con la que se ajusta (`src.similitud.distancias`).
    Aqui es donde se aplica el cambio de espacio, UNA vez para toda la matriz: de
    ahi para abajo —seleccion de vecinos y residuo del minimo cuadrado— ya se
    trabaja en la geometria elegida. None = euclidea, y entonces esto hace
    literalmente lo que hacia antes de que la distancia fuera un parametro.
    """
    M = X.shape[0]
    esp = espacio if espacio is not None else distancias.Espacio(distancias.POR_DEFECTO)
    X = X if esp.es_identidad else esp.transformar(X)
    resolver = (elasticnet_no_negativo if esp.euclidea_transformada
                else elasticnet_residuo_l1)
    vecinos = vecinos_mas_cercanos(X, n_neighbors, espacio=esp)
    cols_idx: list[np.ndarray] = []
    cols_val: list[np.ndarray] = []
    for s in range(M):
        cand = vecinos[s]                 # ya excluye s (w_ss = 0)
        A = X[cand].T                     # (d, n_cand): features en filas
        y = X[s]                          # (d,): observacion objetivo
        w0 = None if inicial is None else inicial(s, cand)
        w = resolver(
            A, y, beta=beta, l1=l1, max_iter=max_iter, tol=tol, w_init=w0)
        nz = w > 0.0
        cols_idx.append(cand[nz].astype(np.int64))
        cols_val.append(w[nz])
        if progreso is not None:  # avance del bucle mas caro del ajuste F2
            progreso(s + 1, M)
    return cols_idx, cols_val


def agregar_W_a_entidades(
    cols_idx: list[np.ndarray],
    cols_val: list[np.ndarray],
    row_entity: np.ndarray,
    weight: np.ndarray,
    n_entities: int,
    simetrizar: bool = True,
) -> np.ndarray:
    """Agrega bloques de W a similitud entidad-entidad (Formulacion 2).

    S[a,b] = (1/Z_ab) * sum_{r en a} sum_{s en b} m_r * m_s * W_rs, con
    m = masa por minutos y Z_ab = (sum_{r en a} m_r) * (sum_{s en b} m_s). Dividir
    por Z neutraliza que un jugador con muchos partidos aporte mas masa (el PDF:
    "con normalizacion por N_p, no domina"). La agregacion cae SOBRE W (pesos
    aprendidos), nunca sobre las features de entrada: ese es el principio central.

    `row_entity[i]` = indice de entidad (0..n_entities-1) de la observacion i.
    Se recorre W por columnas (coste O(nnz)).

    ``simetrizar`` (por defecto True) devuelve la S servible ``0.5*(S+S^T)``. Con
    False se devuelve la S CRUDA direccional (sin simetrizar), que el harness de
    evaluacion usa para medir la asimetria que el pipeline corrige (sanity check
    de la Fase 0); en produccion siempre se simetriza.
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
    if simetrizar:
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

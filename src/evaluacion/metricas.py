"""Metricas de evaluacion en numpy puro (no hay scipy/sklearn en el entorno).

Dos familias:

- Comparacion de rankings top-k: RBO (top-weighted, Webber et al. 2010), Kendall
  tau, MRR y recall@k para la auto-similitud y la estabilidad.
- Tests estadisticos: correlacion de Spearman, test de Mantel por permutacion
  (comparar dos matrices de similitud), test de permutacion pareado (denoising
  F5) y F1 macro (clasificacion posicional downstream).

Todo se implementa aqui porque el proyecto es numpy/pandas puro por diseno.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Comparacion de rankings                                                      #
# --------------------------------------------------------------------------- #

def rbo(lista_a: list[int], lista_b: list[int], p: float) -> float:
    """Rank-Biased Overlap extrapolado (Webber, Moffat & Zobel 2010).

    Metrica top-weighted para comparar dos rankings (posiblemente de distinta
    longitud y sin compartir todos los items): pondera mas las primeras
    posiciones via el parametro geometrico ``p`` (persistencia del usuario).
    Devuelve un valor en [0, 1] (1 = identicos). Se usa la variante extrapolada
    (RBO_EXT) que reparte el residual asumiendo que el solapamiento en la
    profundidad evaluada continua.
    """
    if not lista_a and not lista_b:
        return 1.0
    if not lista_a or not lista_b:
        return 0.0
    k = max(len(lista_a), len(lista_b))
    set_a: set[int] = set()
    set_b: set[int] = set()
    solap = np.zeros(k + 1, dtype=float)  # solap[d] = |A_d interseccion B_d|
    suma = 0.0
    for d in range(1, k + 1):
        if d <= len(lista_a):
            x = lista_a[d - 1]
            # Si x ya esta en B_{d-1} incrementa el solapamiento.
            if x in set_b:
                solap[d] = solap[d - 1] + 1.0
            else:
                solap[d] = solap[d - 1]
            set_a.add(x)
        else:
            solap[d] = solap[d - 1]
        if d <= len(lista_b):
            y = lista_b[d - 1]
            if y in set_a:
                solap[d] += 1.0
            set_b.add(y)
        # Agreement en profundidad d = solapamiento / d.
        suma += (solap[d] / d) * (p ** (d - 1))
    # Termino de suma ponderada.
    rbo_min = (1.0 - p) * suma
    # Extrapolacion del residual con el agreement a profundidad k.
    x_k = solap[k]
    ext = (x_k / k) * (p ** k)
    return float(rbo_min + ext)


def kendall_tau(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Kendall tau-b entre dos vectores de rangos (O(n^2), n pequeno)."""
    n = len(rank_a)
    if n < 2:
        return 1.0
    concordantes = discordantes = 0
    empates_a = empates_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a[i] - rank_a[j]
            db = rank_b[i] - rank_b[j]
            prod = da * db
            if prod > 0:
                concordantes += 1
            elif prod < 0:
                discordantes += 1
            else:
                if da == 0:
                    empates_a += 1
                if db == 0:
                    empates_b += 1
    n0 = n * (n - 1) / 2
    den = np.sqrt((n0 - empates_a) * (n0 - empates_b))
    if den == 0:
        return 0.0
    return float((concordantes - discordantes) / den)


def rango_del_objetivo(scores: np.ndarray, objetivo: int) -> float:
    """Rango 1-based del candidato ``objetivo`` con empates a rango medio (midrank).

    rango = (nº con score ESTRICTAMENTE mayor) + (nº de empatados, incluido el
    objetivo, + 1) / 2. El midrank es la eleccion NEUTRAL ante empates: con la S
    muy dispersa de la F2 (muchos ceros) un objetivo empatado a 0 no recibe el
    mejor rango del grupo (rango minimo = interpretacion optimista, que inflaba
    top-k/MRR de la F2) ni el peor, sino el del medio. Asi la comparacion cara a
    cara con la S densa de la F5 (casi sin empates) no queda sesgada a favor de la
    F2. Devuelve un float (los empates dan medios rangos), valido para recall@k y
    MRR.
    """
    s = scores[objetivo]
    mayores = int(np.sum(scores > s))
    iguales = int(np.sum(scores == s))  # incluye al propio objetivo (>= 1)
    return mayores + (iguales + 1) / 2.0


def recall_at_k(rangos: np.ndarray, k: int) -> float:
    """Fraccion de objetivos recuperados en el top-k (recall@k = top-k accuracy)."""
    if rangos.size == 0:
        return float("nan")
    return float(np.mean(rangos <= k))


def mrr(rangos: np.ndarray) -> float:
    """Mean Reciprocal Rank de los objetivos."""
    if rangos.size == 0:
        return float("nan")
    return float(np.mean(1.0 / rangos))


# --------------------------------------------------------------------------- #
# Tests estadisticos                                                           #
# --------------------------------------------------------------------------- #

def _rankdata(x: np.ndarray) -> np.ndarray:
    """Rangos con promedio de empates (equivalente a scipy 'average'), vectorizado.

    Sin bucles Python: imprescindible porque el test de Mantel llama a esto miles
    de veces sobre vectores de >100k elementos (triangulo superior de S).
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    if n == 0:
        return np.empty(0, dtype=float)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)
    arr_ord = arr[sorter]
    obs = np.r_[True, arr_ord[1:] != arr_ord[:-1]]
    dense = obs.cumsum()[inv]                      # rango denso 1..k
    count = np.r_[np.flatnonzero(obs), n]          # posiciones de fin de cada grupo
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if den == 0:
        return 0.0
    return float(np.sum(x * y) / den)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Correlacion de Spearman = Pearson sobre los rangos."""
    return pearson(_rankdata(x), _rankdata(y))


def _triu_vals(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    return M[iu]


def mantel(
    A: np.ndarray, B: np.ndarray, permutaciones: int, seed: int
) -> tuple[float, float]:
    """Test de Mantel (1967) entre dos matrices simetricas PxP.

    Correlacion de Spearman entre los triangulos superiores de A y B, con
    significancia por permutacion conjunta de filas/columnas (barajar las
    etiquetas de una matriz). Devuelve ``(r, p)``. Precaucion (documentada en el
    PDF): el test puede inflar la significancia ante autocorrelacion estructural
    (posicional); priorizar la MAGNITUD de r sobre el p-valor.
    """
    n = A.shape[0]
    ra = _rankdata(_triu_vals(A))          # se rankea una sola vez (matriz fija)
    r_obs = pearson(ra, _rankdata(_triu_vals(B)))
    rng = np.random.default_rng(seed)
    ge = 1  # cuenta el observado (test de una cola por |r|)
    for _ in range(permutaciones):
        perm = rng.permutation(n)
        Bp = B[np.ix_(perm, perm)]
        r_perm = pearson(ra, _rankdata(_triu_vals(Bp)))
        if abs(r_perm) >= abs(r_obs):
            ge += 1
    p = ge / (permutaciones + 1)
    return float(r_obs), float(p)


def permutacion_pareado(
    dif: np.ndarray, permutaciones: int, seed: int
) -> tuple[float, float]:
    """Test de permutacion pareado por cambio de signo (antes vs despues).

    ``dif`` = diferencias pareadas (p. ej. metrica_post - metrica_pre por
    entidad). Bajo H0 (sin efecto) el signo de cada diferencia es intercambiable.
    Devuelve ``(media_dif, p_dos_colas)``. Alternativa numerica al Wilcoxon
    signed-rank que sugiere el PDF, sin dependencias externas.
    """
    dif = np.asarray(dif, dtype=float)
    dif = dif[dif != 0.0]
    n = len(dif)
    if n == 0:
        return 0.0, 1.0
    obs = float(np.mean(dif))
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(permutaciones):
        signos = rng.choice((-1.0, 1.0), size=n)
        if abs(float(np.mean(signos * np.abs(dif)))) >= abs(obs):
            ge += 1
    return obs, ge / (permutaciones + 1)


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 macro (media no ponderada del F1 por clase)."""
    clases = np.unique(y_true)
    f1s = []
    for c in clases:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def ic_bootstrap(valores: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confianza percentil (para agregar metricas por bootstrap)."""
    valores = np.asarray(valores, dtype=float)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(valores, 100 * alpha / 2))
    hi = float(np.percentile(valores, 100 * (1 - alpha / 2)))
    return (lo, hi)

"""Reentrenamiento incremental: incluir lo nuevo sin repetir el trabajo.

El escenario es el real: una BD con dos ligas, un ajuste en frio, llega una
tercera liga y se reentrena. Lo que se exige:

1. Las entidades nuevas entran en la S y son alcanzables desde las antiguas.
2. El ajuste warm da el MISMO resultado que uno en frio equivalente. Sin esto, el
   ahorro no vale nada: estariamos sirviendo otro modelo.
3. Se reutiliza de verdad (los contadores no son decorativos).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.incremental.reentrenar import reentrenar_uno
from src.similitud import build, warm
from src.similitud.modelo import cargar_modelo
from src.tests import factories

pytestmark = [pytest.mark.integracion, pytest.mark.lento]

COMBINACIONES = [("2", "equipo"), ("5", "equipo"), ("2", "jugador"), ("5", "jugador")]


@pytest.fixture(scope="module")
def escenario(tmp_path_factory) -> dict:
    """BD con 2 ligas y BD con 3, compartiendo fila a fila las dos primeras."""
    base = tmp_path_factory.mktemp("incremental")
    antes = base / "antes.db"
    despues = base / "despues.db"
    factories.crear_bd_sintetica(antes, ligas=factories.LIGAS_SINTETICAS)
    factories.crear_bd_sintetica(
        despues, ligas=factories.LIGAS_SINTETICAS + (factories.LIGA_EXTRA,))
    return {"antes": antes, "despues": despues, "dir": base}


@pytest.fixture(scope="module")
def modelos_iniciales(escenario) -> Path:
    """Ajuste en frio sobre las dos ligas: el punto de partida del incremental."""
    out = escenario["dir"] / "modelo"
    for formulacion, entidad in COMBINACIONES:
        build._construir_uno(escenario["antes"], out, formulacion, entidad, "por_liga")
    return out


def _reentrenar(escenario, modelos_iniciales, formulacion, entidad, destino, **kw):
    return reentrenar_uno(
        escenario["despues"], modelos_iniciales, destino,
        formulacion, entidad, "por_liga", **kw)


# --- El estado queda escrito por el ajuste en frio ----------------------------
@pytest.mark.parametrize("formulacion,entidad", COMBINACIONES)
def test_build_deja_estado_para_la_proxima(modelos_iniciales, formulacion, entidad):
    ruta = warm.ruta_estado(modelos_iniciales, formulacion, entidad, "por_liga")
    assert ruta.exists()
    estado = warm.EstadoWarm.cargar(ruta)
    assert estado.formulacion == formulacion and estado.entidad == entidad
    assert estado.obs_hash.size == estado.n_observaciones


def test_el_modelo_guarda_las_estadisticas_de_normalizacion(modelos_iniciales):
    meta = cargar_modelo(modelos_iniciales, "5", "equipo", "por_liga").meta
    assert meta["estadisticas_normalizacion"]["normalizacion"] == "por_liga"


# --- Las entidades nuevas aparecen -------------------------------------------
@pytest.mark.parametrize("formulacion,entidad", COMBINACIONES)
def test_las_entidades_nuevas_entran_en_el_modelo(
    escenario, modelos_iniciales, tmp_path, formulacion, entidad
):
    destino = tmp_path / f"warm-{formulacion}-{entidad}"
    _reentrenar(escenario, modelos_iniciales, formulacion, entidad, destino)

    antes = cargar_modelo(modelos_iniciales, formulacion, entidad, "por_liga")
    despues = cargar_modelo(destino, formulacion, entidad, "por_liga")
    nuevas = set(despues.entity_ids.tolist()) - set(antes.entity_ids.tolist())

    assert nuevas, "la liga nueva deberia aportar entidades"
    assert set(antes.entity_ids.tolist()) <= set(despues.entity_ids.tolist())
    assert despues.S.shape == (len(despues.entity_ids),) * 2


def test_una_entidad_nueva_es_recomendable_para_una_antigua(
    escenario, modelos_iniciales, tmp_path
):
    """Sin esto el reentrenamiento no serviria de nada: las nuevas quedarian aisladas."""
    destino = tmp_path / "cruce"
    _reentrenar(escenario, modelos_iniciales, "5", "equipo", destino)

    antes = cargar_modelo(modelos_iniciales, "5", "equipo", "por_liga")
    despues = cargar_modelo(destino, "5", "equipo", "por_liga")
    viejas = set(antes.entity_ids.tolist())
    es_nueva = np.array([int(i) not in viejas for i in despues.entity_ids])
    filas_viejas = ~es_nueva

    # Alguna entidad antigua puntua a alguna nueva por encima de cero.
    assert (despues.S[np.ix_(filas_viejas, es_nueva)] > 0).any()
    # Y al reves: las nuevas tienen candidatos entre las antiguas.
    assert (despues.S[np.ix_(es_nueva, filas_viejas)] > 0).any()


# --- Warm frente a frio -------------------------------------------------------
def _divergencia(a, b, k: int = 5) -> tuple[float, float]:
    """(diferencia relativa maxima, solape medio del top-k) entre dos S."""
    escala = max(float(np.abs(b.S).max()), 1e-12)
    aciertos = 0
    for i in range(a.S.shape[0]):
        fa, fb = a.S[i].copy(), b.S[i].copy()
        fa[i] = fb[i] = -np.inf
        aciertos += len(set(np.argsort(fa)[::-1][:k]) & set(np.argsort(fb)[::-1][:k]))
    return float(np.abs(a.S - b.S).max()) / escala, aciertos / (a.S.shape[0] * k)


@pytest.mark.parametrize("formulacion,entidad", COMBINACIONES)
def test_el_warm_start_llega_al_mismo_sitio_que_el_frio(
    escenario, modelos_iniciales, tmp_path, formulacion, entidad
):
    """Warm y frio resuelven el MISMO problema; deben servir el mismo ranking.

    La igualdad no es bit a bit y no se finge que lo sea:

    - Formulacion 2: el coordinate descent para cuando el mayor cambio de peso
      baja de `F2_TOL` (1e-4). Dos arranques distintos paran en puntos distintos
      DENTRO de esa tolerancia — la diferencia encoge proporcionalmente al bajar
      `tol`, que es la firma de un criterio de parada y no la de un error.
    - Formulacion 5: `por_liga` + estadisticas congeladas reproduce el frio
      exactamente en sinkhorn; con MMD interviene ademas el kernel congelado (ver
      la prueba de `--refrescar-kernel`).

    Por eso el top-k tampoco se exige identico en la Formulacion 2: sobre datos
    reales el solape del top-10 medido es 0,992 (ver `docs/incremental.md`), es
    decir se intercambian posiciones de la cola entre candidatos casi empatados.
    Exigir 1,0 aqui seria afirmar algo que no se cumple.
    """
    caliente = tmp_path / "caliente"
    frio = tmp_path / "frio"
    informe = _reentrenar(escenario, modelos_iniciales, formulacion, entidad, caliente,
                          refrescar_kernel=True)
    _reentrenar(escenario, modelos_iniciales, formulacion, entidad, frio, frio=True)

    a = cargar_modelo(caliente, formulacion, entidad, "por_liga")
    b = cargar_modelo(frio, formulacion, entidad, "por_liga")
    relativa, solape = _divergencia(a, b)

    assert np.array_equal(a.entity_ids, b.entity_ids)
    assert informe["estadisticas_congeladas"]
    if formulacion == "5":
        assert relativa < 1e-9     # reutilizacion exacta
        assert solape == 1.0
    else:
        assert relativa < 1e-2     # del orden de F2_TOL propagado a la S
        assert solape >= 0.95      # casi todo el top-5, no necesariamente todo


def test_la_diferencia_de_la_formulacion_2_es_el_criterio_de_parada(
    escenario, modelos_iniciales, tmp_path, monkeypatch
):
    """Con una tolerancia mas fina, warm y frio convergen entre si."""
    from src.similitud import config as config_sim

    monkeypatch.setattr(config_sim, "F2_TOL", 1e-8)
    monkeypatch.setattr(config_sim, "F2_MAX_ITER", 5000)
    caliente = _reentrenar(
        escenario, modelos_iniciales, "2", "equipo", tmp_path / "c")
    _reentrenar(escenario, modelos_iniciales, "2", "equipo", tmp_path / "f", frio=True)
    del caliente

    relativa, _ = _divergencia(
        cargar_modelo(tmp_path / "c", "2", "equipo", "por_liga"),
        cargar_modelo(tmp_path / "f", "2", "equipo", "por_liga"))
    assert relativa < 1e-6


# EASE bajo el que se mide el efecto del kernel congelado. NO es el del modelo
# servido (0,045, apenas re-ranking): con una lambda tan pequeña la S es casi la
# similitud distribucional en crudo y, en este escenario de 9 entidades, las dos
# variantes quedan a un 0,4 % una de otra — el efecto existe pero no se distingue
# del ruido de la escala. Se fija aqui para que la prueba mida lo que dice medir
# y no dependa de un valor que el barrido puede mover.
EASE_MEDIBLE = 50.0


@pytest.fixture
def modelos_con_ease(escenario, tmp_path, monkeypatch) -> Path:
    """Ajuste inicial de F5/jugador con EASE activo, y lo deja activo.

    Tiene que construirse DENTRO del monkeypatch, no solo reentrenarse: el estado
    warm guarda la huella de los hiperparametros con los que se ajusto, asi que un
    inicial a lambda 0 y un reentrenamiento a lambda 50 no se reconocerian y los
    dos caminos acabarian en un ajuste en frio identico — la prueba pasaria sin
    ejercitar nada.

    Se parchea tambien `HIPERPARAMETROS_SERVIBLES`, y no solo el default del
    modulo: F5/jugador/por_liga ES la celda servible, asi que `reentrenar_uno` le
    pone los valores del artefacto servido y esos ganarian al monkeypatch. Sin
    esto, el reentrenamiento ajustaria con otra lambda que el inicial y volveria a
    caer en el ajuste en frio de los dos lados.
    """
    from src.similitud import config as config_sim

    monkeypatch.setattr(config_sim, "F5_EASE_LAMBDA", EASE_MEDIBLE)
    monkeypatch.setitem(
        config_sim.HIPERPARAMETROS_SERVIBLES, "jugador",
        {**config_sim.HIPERPARAMETROS_SERVIBLES["jugador"],
         "F5_EASE_LAMBDA": EASE_MEDIBLE},
    )
    out = tmp_path / "inicial-ease"
    build._construir_uno(escenario["antes"], out, "5", "jugador", "por_liga")
    return out


def test_congelar_el_kernel_estabiliza_a_las_entidades_intactas(
    escenario, modelos_con_ease, tmp_path
):
    """Con MMD, heredar el ancho del kernel mantiene a las viejas en su sitio.

    Es la unica pieza cuyo resultado se aparta del ajuste en frio, y a proposito:
    recalcular el ancho mueve el embedding de todo el mundo, incluido quien no ha
    jugado un minuto mas.
    """
    congelado = tmp_path / "congelado"
    refrescado = tmp_path / "refrescado"
    _reentrenar(escenario, modelos_con_ease, "5", "jugador", congelado)
    _reentrenar(escenario, modelos_con_ease, "5", "jugador", refrescado,
                refrescar_kernel=True)

    inicial = cargar_modelo(modelos_con_ease, "5", "jugador", "por_liga")
    viejas = set(inicial.entity_ids.tolist())

    def desplazamiento(dir_modelo: Path) -> float:
        m = cargar_modelo(dir_modelo, "5", "jugador", "por_liga")
        idx = [i for i, e in enumerate(m.entity_ids) if int(e) in viejas]
        return float(np.abs(m.S[np.ix_(idx, idx)] - inicial.S).max())

    assert desplazamiento(congelado) < desplazamiento(refrescado)


# --- Se reutiliza de verdad ---------------------------------------------------
@pytest.mark.parametrize("formulacion,entidad", COMBINACIONES)
def test_se_reutilizan_las_observaciones_que_no_han_cambiado(
    escenario, modelos_iniciales, tmp_path, formulacion, entidad
):
    informe = _reentrenar(
        escenario, modelos_iniciales, formulacion, entidad, tmp_path / "w")
    w = informe["warm"]

    assert w["aplicado"] and not w["motivo"]
    assert w["observaciones_nuevas"] > 0            # la liga nueva
    assert w["observaciones_intactas"] > 0          # las dos viejas, sin tocar
    # `por_liga` aisla a las ligas antiguas: todo lo que ya estaba sigue intacto.
    assert w["observaciones_intactas"] == w["observaciones_reutilizadas"]
    assert w["entidades_nuevas"] > 0


def test_el_modo_frio_no_reutiliza_nada(escenario, modelos_iniciales, tmp_path):
    informe = _reentrenar(
        escenario, modelos_iniciales, "5", "equipo", tmp_path / "f", frio=True)
    assert not informe["warm"]["aplicado"]
    assert informe["warm"]["observaciones_reutilizadas"] == 0
    assert not informe["estadisticas_congeladas"]


def test_sinkhorn_reutiliza_costes_de_transporte(escenario, modelos_iniciales, tmp_path):
    """El bucle O(P^2) de la etapa 1 es el coste real del equipo en la F5."""
    informe = _reentrenar(escenario, modelos_iniciales, "5", "equipo", tmp_path / "s")
    assert informe["warm"]["pares_ot_reutilizados"] > 0


def test_mmd_congela_el_ancho_del_kernel(escenario, modelos_iniciales, tmp_path):
    destino = tmp_path / "mmd"
    informe = _reentrenar(escenario, modelos_iniciales, "5", "jugador", destino)
    assert informe["warm"]["sigma_congelado"]

    previo = warm.EstadoWarm.cargar(
        warm.ruta_estado(modelos_iniciales, "5", "jugador", "por_liga"))
    nuevo = warm.EstadoWarm.cargar(warm.ruta_estado(destino, "5", "jugador", "por_liga"))
    assert nuevo.sigma == pytest.approx(previo.sigma)


# --- Reentrenar sin novedades -------------------------------------------------
def test_reentrenar_sin_datos_nuevos_reproduce_el_modelo(
    escenario, modelos_iniciales, tmp_path
):
    """Idempotencia: si la BD no ha cambiado, la S tampoco debe moverse."""
    destino = tmp_path / "igual"
    informe = reentrenar_uno(
        escenario["antes"], modelos_iniciales, destino, "5", "equipo", "por_liga")

    antes = cargar_modelo(modelos_iniciales, "5", "equipo", "por_liga")
    despues = cargar_modelo(destino, "5", "equipo", "por_liga")

    assert informe["warm"]["observaciones_nuevas"] == 0
    assert informe["warm"]["entidades_nuevas"] == 0
    assert np.allclose(antes.S, despues.S)


def test_sin_estado_previo_se_ajusta_en_frio_y_avisa(escenario, tmp_path):
    vacio = tmp_path / "sin-modelos"
    vacio.mkdir()
    informe = reentrenar_uno(
        escenario["despues"], vacio, tmp_path / "out", "5", "equipo", "por_liga")
    assert not informe["warm"]["aplicado"]
    assert not informe["estadisticas_congeladas"]
    assert warm.ruta_estado(tmp_path / "out", "5", "equipo", "por_liga").exists()


def test_estado_con_otros_hiperparametros_se_descarta(
    escenario, modelos_iniciales, tmp_path, monkeypatch
):
    from src.similitud import config as config_sim

    # Se SUMA, no se multiplica: el default de produccion es 0.0 y multiplicarlo
    # dejaria el hiperparametro donde estaba, con lo que el estado warm encajaria
    # y la prueba pasaria por el motivo contrario al que dice comprobar.
    monkeypatch.setattr(config_sim, "F5_EASE_LAMBDA", config_sim.F5_EASE_LAMBDA + 7.0)
    informe = _reentrenar(escenario, modelos_iniciales, "5", "equipo", tmp_path / "h")

    assert not informe["warm"]["aplicado"]
    assert "ease_lambda" in informe["warm"]["motivo"]

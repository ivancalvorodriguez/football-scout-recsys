"""El aviso de cobertura, visto desde las paginas que tienen que darlo.

Reproduce la situacion real que se colo en produccion: un modelo entrenado ANTES
de incorporar partidos, servido con la BD ya ampliada. La app respondia igual, y
los jugadores nuevos —los del conjunto que el usuario acababa de crear— no
aparecian en ninguna busqueda ni habia nada en pantalla que lo explicara.

Donde se DA ese aviso cambio despues: el buscador abria con dos bandas que
contaban lo mismo (esta y la de «consultando sobre otros datos») delante de una
respuesta que ademas se rotula sola. Hoy la cobertura se mide igual pero se
enseña en `/datos`, que es donde se decide reentrenar; aqui se comprueban las dos
mitades.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.app import config as config_app
from src.app.catalogo import ClaveModelo
from src.app.factoria import crear_app
from src.tests.app.conftest import (
    CONJUNTO_SLUG,
    ClienteConCSRF,
    FICHAJE,
    VARIANTE_SLUG,
    entrar,
)

pytestmark = pytest.mark.integracion

# Nombre del modelo desfasado que monta el fixture.
SLUG_VIEJO = "modelo-viejo"
NOMBRE_VIEJO = "Modelo viejo"


def _texto(respuesta) -> str:
    return respuesta.get_data(as_text=True)


@pytest.fixture
def cliente_desfasado(dir_modelos: Path, bd: Path, tmp_path: Path,
                      fichero_usuarios: Path):
    """App con un modelo que apunta al conjunto ampliado sin conocer su fichaje.

    Se compone copiando el directorio de modelos y dandole al modelo nuevo los
    artefactos del BASE (los de 7 jugadores) mientras su `variante.json` declara
    el conjunto ampliado, que ya tiene 8. Es exactamente lo que deja incorporar
    partidos y NO reentrenar.
    """
    destino = tmp_path / "modelos"
    shutil.copytree(dir_modelos, destino)
    carpeta = destino / config_app.SUBDIR_VARIANTES / SLUG_VIEJO
    carpeta.mkdir(parents=True)
    (carpeta / config_app.FICHERO_VARIANTE).write_text(
        json.dumps({
            "nombre": NOMBRE_VIEJO, "slug": SLUG_VIEJO,
            "creado": "2026-01-01T10:00:00",
            "origen": config_app.VARIANTE_BASE, "datos": CONJUNTO_SLUG,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    for entidad in config_app.ENTIDADES:
        stem = ClaveModelo(entidad).stem
        for extension in (".npz", ".json"):
            shutil.copy(destino / f"{stem}{extension}", carpeta / f"{stem}{extension}")
    app = crear_app(destino, db_path=bd, ruta_usuarios=fichero_usuarios,
                    testing=True)
    app.test_client_class = ClienteConCSRF
    cliente = app.test_client()
    entrar(cliente)
    return cliente


# --- Buscador -----------------------------------------------------------------
def test_el_buscador_ya_no_pinta_la_banda_de_cobertura(cliente_desfasado):
    """El recomendador no abre con dos avisos que dicen lo mismo.

    La cobertura incompleta y el «consultando sobre otros datos» describen la
    misma situación —modelo y BD que no cuadran— y, encima, la respuesta de una
    entidad proyectada ya se rotula ella sola («Calculado fuera del modelo»). El
    aviso se midió y se sigue midiendo: donde vive es en `/datos`, que es donde se
    decide reentrenar.
    """
    for url in (f"/?modelo={SLUG_VIEJO}",
                f"/similares?entidad=jugador&id=10&k=2&modelo={SLUG_VIEJO}"):
        assert "Cobertura incompleta" not in _texto(cliente_desfasado.get(url))


def test_al_jugador_que_falta_se_le_explica_por_que_no_se_le_puede_colocar(
    cliente_desfasado,
):
    """Ya no es un 404 mudo: o se le proyecta, o se dice por que no.

    Estos artefactos estan escritos a mano y no guardan con que se estandarizaron,
    asi que no se les puede proyectar nada (proyectar exige el MISMO espacio de
    features). Lo que se comprueba aqui es que eso se explica y se distingue de
    «ese nombre no existe»: 409 y un mensaje que dice que hay que entrenar. El
    camino en el que SI se proyecta se prueba en `test_proyeccion_vistas.py`, con
    modelos construidos de verdad.
    """
    r = cliente_desfasado.get(
        f"/similares?entidad=jugador&nombre={FICHAJE[1]}&modelo={SLUG_VIEJO}")
    assert r.status_code == 409
    assert "Datos y modelos" in _texto(r) or "Entrena" in _texto(r)
    # Y la misma app, con el modelo que si lo conoce, lo encuentra.
    ok = cliente_desfasado.get(
        f"/similares?entidad=jugador&nombre={FICHAJE[1]}&modelo={VARIANTE_SLUG}")
    assert ok.status_code == 200


def test_un_modelo_coherente_no_pinta_ningun_aviso(cliente_desfasado):
    """El caso normal no gana ruido: el aviso solo aparece cuando falta alguien."""
    html = _texto(cliente_desfasado.get(f"/?modelo={VARIANTE_SLUG}"))
    assert "Cobertura incompleta" not in html


def test_sin_bd_no_se_inventa_ningun_aviso(cliente_sin_bd):
    """Sin base de datos no hay universo con el que comparar: se calla."""
    assert "Cobertura incompleta" not in _texto(cliente_sin_bd.get("/"))


# --- Pagina de datos ----------------------------------------------------------
def test_datos_relaciona_las_dos_cifras_de_cada_modelo(cliente_desfasado):
    """Era el bug de fondo: la pagina anunciaba los jugadores de la BD y los del
    modelo sin ponerlos nunca juntos."""
    # La plantilla parte las lineas donde le conviene; lo que se comprueba es el
    # texto, no su sangrado.
    html = " ".join(_texto(cliente_desfasado.get("/datos/")).split())
    assert "6 de 7 jugadores" in html
    assert "El modelo cubre 6 de 7 jugadores de estos datos" in html


def test_datos_tambien_cuenta_lo_que_sobra(cliente_desfasado):
    """El modelo base sintetico trae un jugador sin estadisticas en la BD.

    Sobrar no impide buscar (por eso el buscador calla), pero en la pagina de
    gestion si es informacion: dice que ese modelo y esa BD no son la pareja que
    parecen.
    """
    html = _texto(cliente_desfasado.get("/datos/"))
    assert "1 jugadores del modelo no están en estos datos" in html


def test_datos_no_carga_ninguna_matriz_para_avisar(cliente_desfasado, monkeypatch):
    """La pagina lista TODOS los modelos: si midiera cargandolos, no escalaria."""
    from src.app import catalogo as modulo_catalogo

    monkeypatch.setattr(
        modulo_catalogo, "cargar_modelo",
        lambda *a, **k: pytest.fail("la pagina de datos no debe cargar modelos"),
    )
    assert cliente_desfasado.get("/datos/").status_code == 200

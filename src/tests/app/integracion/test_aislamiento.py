"""Propiedad, cuotas y aislamiento entre cuentas.

Hasta ahora no existia la nocion de dueno: los catalogos enumeraban carpetas y
todo el mundo veia todo. Lo que se comprueba aqui es lo contrario, en las dos
direcciones:

- lo de una cuenta **no se ve ni se puede pedir** desde la otra, y pedirlo da 404
  (no 403: un 403 confirma que existe, que es la mitad de lo que se protege);
- lo compartido —el modelo y el conjunto de fabrica— **si lo ven las dos**, y no
  lo borra ninguna.

Y las cuotas, que son lo que impide que una cuenta llene el disco: 3 modelos y 4
conjuntos, contando solo lo propio, con borrado para poder liberar sitio.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.app import config as config_app
from src.app.factoria import crear_app
from src.app.tareas import OCUPADO_AJENO, Tarea
from src.tests.app.conftest import (
    NOMBRE_A,
    USUARIO_A,
    USUARIO_B,
    ClienteConCSRF,
    entrar,
)
from src.tests.app.integracion.test_rutas_datos import GestorFalso

pytestmark = pytest.mark.integracion


def _texto(respuesta) -> str:
    return respuesta.get_data(as_text=True)


@pytest.fixture
def app(dir_modelos: Path, bd: Path, tmp_path: Path, fichero_usuarios: Path):
    """App sobre copias: aqui se crean y se borran modelos y conjuntos de verdad."""
    modelos = tmp_path / "modelo"
    shutil.copytree(dir_modelos, modelos)
    copia_bd = tmp_path / "db" / bd.name
    copia_bd.parent.mkdir(parents=True)
    shutil.copy2(bd, copia_bd)
    creada = crear_app(modelos, db_path=copia_bd, ruta_usuarios=fichero_usuarios,
                       testing=True)
    creada.test_client_class = ClienteConCSRF
    return creada


@pytest.fixture
def gestor(app) -> GestorFalso:
    falso = GestorFalso()
    app.extensions["tareas"] = falso
    return falso


@pytest.fixture
def cliente_a(app):
    c = app.test_client()
    entrar(c, USUARIO_A)
    return c


@pytest.fixture
def cliente_b(app):
    c = app.test_client()
    entrar(c, USUARIO_B)
    return c


def _entrenar(cliente, nombre: str, **extra):
    return cliente.post("/datos/reentrenar", data={"nombre": nombre, **extra})


def _crear_datos(cliente, nombre: str, paquete: Path, **extra):
    return cliente.post("/datos/ingerir",
                        data={"paquete": str(paquete), "nombre_datos": nombre, **extra})


# --- Propiedad: se apunta al crear ---------------------------------------------
def test_el_modelo_creado_apunta_a_su_dueno(cliente_a, gestor, app):
    _entrenar(cliente_a, "Mio")
    assert app.extensions["catalogo"].variante("mio", USUARIO_A).usuario == USUARIO_A


def test_el_conjunto_creado_apunta_a_su_dueno(cliente_a, gestor, app, tmp_path: Path):
    paquete = tmp_path / "paq"
    paquete.mkdir()
    _crear_datos(cliente_a, "Mios", paquete)
    assert app.extensions["datos"].conjunto("mios", USUARIO_A).usuario == USUARIO_A


def test_lo_que_ya_existia_sin_dueno_es_compartido(app):
    """Los modelos anteriores a las cuentas no se le adjudican a nadie: se ven
    desde todas, no los borra ninguna y no le cuentan cuota a ninguna."""
    from src.tests.app.conftest import VARIANTE_SLUG

    catalogo = app.extensions["catalogo"]
    heredado = catalogo.variante(VARIANTE_SLUG, USUARIO_A)
    assert heredado.compartido
    assert heredado.visible_para(USUARIO_A) and heredado.visible_para(USUARIO_B)
    assert not heredado.borrable_por(USUARIO_A)
    assert catalogo.cuota(USUARIO_A)[0] == 0


# --- Aislamiento: no ver lo del otro -------------------------------------------
def test_una_cuenta_no_ve_los_modelos_de_la_otra(cliente_a, cliente_b, gestor):
    _entrenar(cliente_a, "Solo De Ana")
    assert "Solo De Ana" in _texto(cliente_a.get("/datos/"))
    assert "Solo De Ana" not in _texto(cliente_b.get("/datos/"))


def test_una_cuenta_no_ve_los_conjuntos_de_la_otra(cliente_a, cliente_b, gestor,
                                                   tmp_path: Path):
    paquete = tmp_path / "paq"
    paquete.mkdir()
    _crear_datos(cliente_a, "Datos De Ana", paquete)
    assert "Datos De Ana" in _texto(cliente_a.get("/datos/"))
    assert "Datos De Ana" not in _texto(cliente_b.get("/datos/"))


def test_pedir_por_url_el_modelo_de_otro_da_404(cliente_a, cliente_b, gestor, app):
    """404 y no 403: confirmar que existe ya dice que la otra cuenta lo entreno."""
    _entrenar(cliente_a, "Solo De Ana")
    # Se le dan artefactos para que sea servible de verdad y el 404 no venga de
    # que este a medias.
    origen = app.extensions["catalogo"].dir_modelo(config_app.VARIANTE_BASE)
    destino = app.extensions["catalogo"].dir_modelo("solo-de-ana")
    for fichero in origen.glob("formulacion*"):
        shutil.copy(fichero, destino / fichero.name)

    assert cliente_a.get("/?modelo=solo-de-ana").status_code == 200
    assert cliente_b.get("/?modelo=solo-de-ana").status_code == 404


def test_un_modelo_inventado_da_el_mismo_404(cliente_b):
    """Indistinguible de pedir el de otro: si no, probando nombres se enumera."""
    assert cliente_b.get("/?modelo=no-existe-en-absoluto").status_code == 404


def test_consultar_sobre_el_conjunto_de_otro_da_404(cliente_a, cliente_b, gestor,
                                                    tmp_path: Path):
    """La BD tambien se elige desde el buscador (`?datos=`), asi que es otra
    puerta por la que se podria leer lo de otra cuenta: quien busca sobre un
    conjunto ve QUIEN esta en el (el buscador ofrece sus entidades y las
    proyecta). Mismo criterio que con el modelo: 404, no 403."""
    paquete = tmp_path / "paq_datos"
    paquete.mkdir()
    _crear_datos(cliente_a, "Base De Ana", paquete)
    assert cliente_a.get("/?datos=base-de-ana").status_code == 200
    assert cliente_b.get("/?datos=base-de-ana").status_code == 404


def test_un_conjunto_inventado_da_el_mismo_404(cliente_b):
    assert cliente_b.get("/?datos=no-existe-en-absoluto").status_code == 404


def test_el_conjunto_base_se_puede_pedir_desde_el_buscador(cliente_a, cliente_b):
    for cliente in (cliente_a, cliente_b):
        assert cliente.get(
            f"/?datos={config_app.VARIANTE_BASE}").status_code == 200


def test_el_modelo_base_lo_ven_las_dos_cuentas(cliente_a, cliente_b):
    for cliente in (cliente_a, cliente_b):
        assert cliente.get(f"/?modelo={config_app.VARIANTE_BASE}").status_code == 200


def test_el_conjunto_base_lo_ven_las_dos_cuentas(cliente_a, cliente_b):
    for cliente in (cliente_a, cliente_b):
        assert config_app.NOMBRE_BASE in _texto(cliente.get("/datos/"))


def test_no_se_puede_partir_del_conjunto_de_otro(cliente_a, cliente_b, gestor,
                                                 tmp_path: Path):
    """El desplegable no lo ofrece, pero el POST se puede escribir a mano."""
    paquete = tmp_path / "paq"
    paquete.mkdir()
    _crear_datos(cliente_a, "Datos De Ana", paquete)
    respuesta = _crear_datos(cliente_b, "Copia", paquete, datos_origen="datos-de-ana")
    assert respuesta.status_code == 400
    assert "No existe" in _texto(respuesta)


def test_no_se_puede_partir_del_modelo_de_otro(cliente_a, cliente_b, gestor):
    _entrenar(cliente_a, "Solo De Ana")
    respuesta = _entrenar(cliente_b, "Aprovechado", origen="solo-de-ana")
    assert respuesta.status_code == 400
    assert "No hay ningún modelo entrenado" in _texto(respuesta)


def test_el_desplegable_de_origen_solo_ofrece_lo_propio(cliente_a, cliente_b, gestor):
    _entrenar(cliente_a, "Solo De Ana")
    html = _texto(cliente_b.get("/datos/"))
    assert 'value="solo-de-ana"' not in html


# --- Aislamiento de las tareas -------------------------------------------------
class GestorConTareaAjena(GestorFalso):
    """Gestor ocupado con una tarea de OTRA cuenta."""

    def __init__(self) -> None:
        super().__init__(ocupado=True)
        self.tarea_ajena = Tarea(
            id="ajena", tipo="reentrenar",
            titulo="Entrenando el modelo «Fichajes secretos»",
            comando=["python", "-m", "x", "--out", "/ruta/privada"],
            usuario=USUARIO_B, lineas=["linea privada"],
        )

    def ultima(self, usuario: str | None = None):
        return self.tarea_ajena if usuario == USUARIO_B else None

    def en_curso(self):
        return self.tarea_ajena


def test_la_api_de_tarea_no_devuelve_la_tarea_ajena(cliente_a, app):
    app.extensions["tareas"] = GestorConTareaAjena()
    datos = cliente_a.get("/datos/tarea").get_json()
    assert datos["tarea"] is None
    # Pero se admite decir que el servidor esta ocupado: explica el 409 sin
    # contar de quien es la tarea.
    assert datos["ocupado"] is True


def test_su_dueno_si_la_ve(cliente_b, app):
    app.extensions["tareas"] = GestorConTareaAjena()
    datos = cliente_b.get("/datos/tarea").get_json()
    assert datos["tarea"]["titulo"] == "Entrenando el modelo «Fichajes secretos»"


def test_la_pagina_no_pinta_el_titulo_ni_el_log_ajenos(cliente_a, app):
    app.extensions["tareas"] = GestorConTareaAjena()
    html = _texto(cliente_a.get("/datos/"))
    assert "Fichajes secretos" not in html
    assert "linea privada" not in html
    assert "/ruta/privada" not in html


def test_el_mensaje_de_ocupado_no_revela_el_titulo_ajeno(app, fichero_usuarios: Path):
    """Antes decia «Ya hay una tarea en curso (Entrenando el modelo «X»)»."""
    from src.app.tareas import GestorTareas, TareaEnCurso

    gestor = GestorTareas(cwd=Path("."))
    gestor._actual = Tarea(id="a", tipo="reentrenar",
                           titulo="Entrenando el modelo «Fichajes secretos»",
                           comando=["python"], usuario=USUARIO_B)
    with pytest.raises(TareaEnCurso) as excinfo:
        gestor.lanzar("reentrenar", "Lo mio", [], usuario=USUARIO_A)
    assert str(excinfo.value) == OCUPADO_AJENO
    assert "Fichajes" not in str(excinfo.value)
    assert USUARIO_B not in str(excinfo.value)


def test_con_su_propia_tarea_si_se_le_dice_cual_es(app):
    """A su dueno no hay nada que ocultarle: saber cual es le ayuda a esperar."""
    from src.app.tareas import GestorTareas, TareaEnCurso

    gestor = GestorTareas(cwd=Path("."))
    gestor._actual = Tarea(id="a", tipo="reentrenar", titulo="Entrenando lo mio",
                           comando=["python"], usuario=USUARIO_A)
    with pytest.raises(TareaEnCurso) as excinfo:
        gestor.lanzar("reentrenar", "Otra", [], usuario=USUARIO_A)
    assert "Entrenando lo mio" in str(excinfo.value)


# --- Cuotas -------------------------------------------------------------------
def test_la_cuota_corta_el_modelo_que_pasa_del_tope(cliente_a, gestor):
    """`MAX_MODELOS_POR_USUARIO`. El número sale del config, no está escrito
    aquí: cambiarlo es un ajuste de producto y no debe romper la prueba."""
    tope = config_app.MAX_MODELOS_POR_USUARIO
    for i in range(tope):
        assert _entrenar(cliente_a, f"Modelo {i}").status_code == 302
    ultimo = _entrenar(cliente_a, "Uno De Mas")
    assert ultimo.status_code == 409
    assert f"máximo de {tope} modelos" in _texto(ultimo)


def test_la_cuota_corta_el_conjunto_que_pasa_del_tope(cliente_a, gestor, tmp_path: Path):
    """`MAX_DATASETS_POR_USUARIO`, igual que el anterior."""
    tope = config_app.MAX_DATASETS_POR_USUARIO
    paquete = tmp_path / "paq"
    paquete.mkdir()
    for i in range(tope):
        assert _crear_datos(cliente_a, f"Datos {i}", paquete).status_code == 302
    ultimo = _crear_datos(cliente_a, "Uno De Mas", paquete)
    assert ultimo.status_code == 409
    assert f"máximo de {tope} conjuntos" in _texto(ultimo)


def test_la_cuota_de_una_cuenta_no_gasta_la_de_la_otra(cliente_a, cliente_b, gestor):
    for i in range(config_app.MAX_MODELOS_POR_USUARIO):
        _entrenar(cliente_a, f"De Ana {i}")
    assert _entrenar(cliente_a, "Otro").status_code == 409
    assert _entrenar(cliente_b, "De Bruno").status_code == 302


def test_al_llegar_al_tope_no_se_copia_nada(cliente_a, gestor, app, tmp_path: Path):
    """La cuota se mira ANTES de copiar: cada copia es una BD entera."""
    paquete = tmp_path / "paq"
    paquete.mkdir()
    for i in range(config_app.MAX_DATASETS_POR_USUARIO):
        _crear_datos(cliente_a, f"Datos {i}", paquete)
    antes = len(list(app.extensions["datos"].dir_conjuntos.iterdir()))
    _crear_datos(cliente_a, "Uno De Mas", paquete)
    assert len(list(app.extensions["datos"].dir_conjuntos.iterdir())) == antes


def test_la_pagina_lleva_los_contadores(cliente_a, gestor):
    _entrenar(cliente_a, "Uno")
    html = " ".join(_texto(cliente_a.get("/datos/")).split())
    assert f"1 de {config_app.MAX_MODELOS_POR_USUARIO}" in html      # modelos
    assert f"0 de {config_app.MAX_DATASETS_POR_USUARIO}" in html     # conjuntos


def test_al_llegar_al_tope_el_boton_queda_deshabilitado_con_su_razon(cliente_a, gestor):
    for i in range(config_app.MAX_MODELOS_POR_USUARIO):
        _entrenar(cliente_a, f"Modelo {i}")
    html = " ".join(_texto(cliente_a.get("/datos/")).split())
    assert "que es el máximo por cuenta" in html
    assert "disabled" in html


# --- Borrado ------------------------------------------------------------------
def test_borrar_un_modelo_propio_libera_cuota(cliente_a, gestor, app):
    for i in range(config_app.MAX_MODELOS_POR_USUARIO):
        _entrenar(cliente_a, f"Modelo {i}")
    assert _entrenar(cliente_a, "No Cabe").status_code == 409

    assert cliente_a.post("/datos/borrar-modelo",
                          data={"slug": "modelo-0"}).status_code == 302
    assert _entrenar(cliente_a, "Ahora Si").status_code == 302


def test_borrar_un_conjunto_propio_libera_cuota(cliente_a, gestor, tmp_path: Path):
    paquete = tmp_path / "paq"
    paquete.mkdir()
    for i in range(config_app.MAX_DATASETS_POR_USUARIO):
        _crear_datos(cliente_a, f"Datos {i}", paquete)
    assert _crear_datos(cliente_a, "No Cabe", paquete).status_code == 409

    assert cliente_a.post("/datos/borrar-conjunto",
                          data={"slug": "datos-0"}).status_code == 302
    assert _crear_datos(cliente_a, "Ahora Si", paquete).status_code == 302


def test_borrar_se_lleva_la_carpeta(cliente_a, gestor, app):
    _entrenar(cliente_a, "Efimero")
    carpeta = app.extensions["catalogo"].dir_modelo("efimero")
    assert carpeta.is_dir()
    cliente_a.post("/datos/borrar-modelo", data={"slug": "efimero"})
    assert not carpeta.exists()


def test_no_se_puede_borrar_el_modelo_de_otro(cliente_a, cliente_b, gestor, app):
    _entrenar(cliente_a, "De Ana")
    respuesta = cliente_b.post("/datos/borrar-modelo", data={"slug": "de-ana"})
    assert respuesta.status_code == 404
    assert app.extensions["catalogo"].dir_modelo("de-ana").is_dir()


def test_no_se_puede_borrar_el_conjunto_de_otro(cliente_a, cliente_b, gestor,
                                                app, tmp_path: Path):
    paquete = tmp_path / "paq"
    paquete.mkdir()
    _crear_datos(cliente_a, "De Ana", paquete)
    respuesta = cliente_b.post("/datos/borrar-conjunto", data={"slug": "de-ana"})
    assert respuesta.status_code == 404
    assert app.extensions["datos"].ruta("de-ana").exists()


def test_no_se_puede_borrar_el_modelo_base(cliente_a, gestor, app):
    respuesta = cliente_a.post("/datos/borrar-modelo",
                               data={"slug": config_app.VARIANTE_BASE})
    assert respuesta.status_code == 404
    assert app.extensions["catalogo"].model_dir.is_dir()


def test_no_se_puede_borrar_el_conjunto_base(cliente_a, gestor, app):
    respuesta = cliente_a.post("/datos/borrar-conjunto",
                               data={"slug": config_app.VARIANTE_BASE})
    assert respuesta.status_code == 404
    assert app.config["APP_SIMILITUD"].db_path.exists()


def test_no_se_puede_borrar_lo_compartido(cliente_a, gestor, app):
    """Lo heredado (sin dueno) se ve desde todas las cuentas; si cualquiera
    pudiera borrarlo, «compartido» seria «de todos para destruirlo»."""
    from src.tests.app.conftest import VARIANTE_SLUG

    respuesta = cliente_a.post("/datos/borrar-modelo", data={"slug": VARIANTE_SLUG})
    assert respuesta.status_code == 404
    assert app.extensions["catalogo"].dir_modelo(VARIANTE_SLUG).is_dir()


def test_no_se_borra_con_una_tarea_en_curso(cliente_a, app):
    """Podria estar escribiendo justo en esa carpeta."""
    app.extensions["tareas"] = GestorFalso(ocupado=True)
    respuesta = cliente_a.post("/datos/borrar-modelo", data={"slug": "lo-que-sea"})
    assert respuesta.status_code == 409
    assert "tarea en curso" in _texto(respuesta)


def test_el_boton_de_borrar_solo_sale_en_lo_propio(cliente_a, cliente_b, gestor):
    _entrenar(cliente_a, "De Ana")
    assert "Borrar" in _texto(cliente_a.get("/datos/"))
    # Bruno no tiene nada suyo: solo ve el base y lo compartido, sin borrar.
    assert "Borrar" not in _texto(cliente_b.get("/datos/"))

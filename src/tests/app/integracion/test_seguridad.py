"""Login, cierre de sesion, CSRF y cabeceras, vistos desde el cliente HTTP.

Antes de esto la app no tenia autenticacion de ninguna clase y `/datos` lanzaba
subprocesos, copiaba bases de datos y entrenaba modelos. Aqui se comprueba la
puerta: que esta, que se abre con las credenciales correctas y solo con esas, y
que un POST llegado de fuera no pasa.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app import config as config_app
from src.app.factoria import CABECERAS_SEGURIDAD, crear_app
from src.tests.app.conftest import (
    CONTRASENA,
    NOMBRE_A,
    USUARIO_A,
    USUARIO_B,
    ClienteConCSRF,
    entrar,
    token_csrf,
)

pytestmark = pytest.mark.integracion


def _texto(respuesta) -> str:
    return respuesta.get_data(as_text=True)


# --- Login --------------------------------------------------------------------
def test_las_credenciales_correctas_abren_sesion(cliente_anonimo):
    r = cliente_anonimo.post("/login", data={
        "usuario": USUARIO_A, "contrasena": CONTRASENA})
    assert r.status_code == 302
    # `/datos` es lo que separa a quien tiene cuenta de quien no.
    assert cliente_anonimo.get("/datos/").status_code == 200


@pytest.mark.parametrize("usuario, contrasena", [
    (USUARIO_A, "contrasena-equivocada"),
    ("fantasma", CONTRASENA),
    ("", ""),
])
def test_las_credenciales_malas_no_abren_nada(cliente_anonimo, usuario, contrasena):
    r = cliente_anonimo.post("/login",
                             data={"usuario": usuario, "contrasena": contrasena})
    assert r.status_code == 401
    assert cliente_anonimo.get("/datos/").status_code == 302


def test_el_error_no_distingue_entre_cuenta_y_contrasena(cliente_anonimo):
    """Distinguirlos convierte el formulario en un listado de que cuentas hay."""
    inexistente = _texto(cliente_anonimo.post(
        "/login", data={"usuario": "fantasma", "contrasena": CONTRASENA}))
    equivocada = _texto(cliente_anonimo.post(
        "/login", data={"usuario": USUARIO_A, "contrasena": "otra-cosa"}))
    from src.app.auth import ERROR_CREDENCIALES

    assert ERROR_CREDENCIALES in inexistente
    assert ERROR_CREDENCIALES in equivocada


def test_la_contrasena_no_vuelve_en_la_respuesta(cliente_anonimo):
    """Ni siquiera reescrita en el formulario tras fallar."""
    html = _texto(cliente_anonimo.post(
        "/login", data={"usuario": USUARIO_A, "contrasena": CONTRASENA + "x"}))
    assert CONTRASENA not in html


def test_la_contrasena_no_llega_al_log(cliente_anonimo, caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        cliente_anonimo.post("/login",
                             data={"usuario": USUARIO_A, "contrasena": CONTRASENA})
        cliente_anonimo.post("/login",
                             data={"usuario": USUARIO_A, "contrasena": "mal"})
    assert CONTRASENA not in caplog.text


# --- Guardia de sesion --------------------------------------------------------
@pytest.mark.parametrize("ruta", [
    "/datos/", "/datos/tarea", "/datos/carpetas",
])
def test_la_gestion_sin_sesion_redirige_al_login(cliente_anonimo, ruta):
    r = cliente_anonimo.get(ruta)
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


@pytest.mark.parametrize("ruta", [
    "/", "/similares?entidad=jugador&nombre=Messi", "/jugador/10", "/equipo/1",
    "/api/sugerencias?q=Me", "/api/similares?entidad=jugador&nombre=Messi",
    "/api/ficha/jugador/10", "/glosario", "/login", "/registro",
])
def test_el_recomendador_se_usa_sin_cuenta(cliente_anonimo, ruta):
    """La portada ES el recomendador y se sirve a cualquiera."""
    assert cliente_anonimo.get(ruta).status_code == 200


def test_la_portada_sin_cuenta_trae_el_buscador(cliente_anonimo):
    html = _texto(cliente_anonimo.get("/"))
    assert "Buscar similares" in html
    assert "¿A quién se parece?" in html


def test_una_busqueda_completa_sin_cuenta(cliente_anonimo):
    html = _texto(cliente_anonimo.get(
        "/similares?entidad=jugador&nombre=Messi&k=3"))
    assert "Lionel Messi" in html


def test_el_login_devuelve_a_donde_se_iba(cliente_anonimo):
    destino = cliente_anonimo.get("/datos/").headers["Location"]
    r = cliente_anonimo.post(destino, data={
        "usuario": USUARIO_A, "contrasena": CONTRASENA})
    assert r.headers["Location"] == "/datos/"


@pytest.mark.parametrize("destino", [
    "https://otro-sitio.example/robar", "//otro-sitio.example/robar",
])
def test_no_se_puede_usar_el_login_como_redirector_abierto(cliente_anonimo, destino):
    """Un redirector abierto es la mitad de un phishing convincente."""
    r = cliente_anonimo.post(f"/login?siguiente={destino}", data={
        "usuario": USUARIO_A, "contrasena": CONTRASENA})
    assert "otro-sitio" not in r.headers["Location"]


def test_el_glosario_se_lee_sin_entrar(cliente_anonimo):
    """Es documentacion: que mide cada metrica no es dato de nadie."""
    assert cliente_anonimo.get("/glosario").status_code == 200


def test_los_estaticos_se_sirven_sin_entrar(cliente_anonimo):
    """La propia pagina de login pide su hoja de estilos."""
    assert cliente_anonimo.get("/static/estilo.css").status_code == 200


def test_una_cuenta_borrada_deja_de_valer(app, fichero_usuarios: Path):
    """La cookie firmada seguiria siendo valida: se comprueba contra el fichero."""
    from src.app.auth import Usuarios

    cliente = app.test_client()
    entrar(cliente)
    assert cliente.get("/datos/").status_code == 200
    Usuarios(fichero_usuarios).borrar(USUARIO_A)
    assert cliente.get("/datos/").status_code == 302
    # Y sigue pudiendo usar el recomendador, como cualquier visitante.
    assert cliente.get("/").status_code == 200


# --- Registro -----------------------------------------------------------------
NUEVA = {"usuario": "carla", "nombre": "Carla Vega",
         "contrasena": "contrasena-nueva-1", "contrasena2": "contrasena-nueva-1"}


def test_el_formulario_de_registro_se_sirve(cliente_anonimo):
    html = _texto(cliente_anonimo.get("/registro"))
    assert "Crear cuenta" in html
    assert 'name="csrf_token"' in html


def test_registrarse_crea_la_cuenta_y_entra(cliente_anonimo, fichero_usuarios: Path):
    """Obligar a repetir las credenciales recien escritas no aporta nada."""
    from src.app.auth import Usuarios, verificar

    r = cliente_anonimo.post("/registro", data=NUEVA)
    assert r.status_code == 302
    cuenta = Usuarios(fichero_usuarios).obtener("carla")
    assert cuenta is not None and cuenta.nombre == "Carla Vega"
    assert verificar(cuenta, NUEVA["contrasena"])
    # Y ya esta dentro: llega a la seccion que exige cuenta.
    assert cliente_anonimo.get("/datos/").status_code == 200


def test_la_cuenta_recien_creada_empieza_sin_nada(cliente_anonimo):
    cliente_anonimo.post("/registro", data=NUEVA)
    html = " ".join(_texto(cliente_anonimo.get("/datos/")).split())
    assert f"0 de {config_app.MAX_MODELOS_POR_USUARIO}" in html
    assert f"0 de {config_app.MAX_DATASETS_POR_USUARIO}" in html


def test_no_se_puede_repetir_un_usuario(cliente_anonimo, fichero_usuarios: Path):
    r = cliente_anonimo.post("/registro", data={**NUEVA, "usuario": USUARIO_A})
    assert r.status_code == 400
    assert "Ya hay una cuenta" in _texto(r)


def test_el_usuario_repetido_no_pisa_la_contrasena_del_otro(cliente_anonimo,
                                                            fichero_usuarios: Path):
    """Lo importante del caso anterior: la cuenta que ya existia queda intacta."""
    from src.app.auth import Usuarios, verificar

    cliente_anonimo.post("/registro", data={**NUEVA, "usuario": USUARIO_A})
    assert verificar(Usuarios(fichero_usuarios).obtener(USUARIO_A), CONTRASENA)


@pytest.mark.parametrize("cambio, esperado", [
    ({"usuario": ""}, "Escribe un nombre"),
    ({"usuario": "con espacio"}, "solo admite"),
    ({"contrasena": "corta", "contrasena2": "corta"}, "al menos"),
    ({"contrasena2": "otra-cosa-larga"}, "no coinciden"),
])
def test_los_datos_malos_se_rechazan_diciendo_que_falla(cliente_anonimo, cambio,
                                                        esperado):
    r = cliente_anonimo.post("/registro", data={**NUEVA, **cambio})
    assert r.status_code == 400
    assert esperado in _texto(r)


def test_el_registro_exige_token_csrf(cliente_anonimo):
    assert cliente_anonimo.post("/registro", data=NUEVA, sin_csrf=True).status_code == 400


def test_el_registro_no_devuelve_la_contrasena(cliente_anonimo):
    html = _texto(cliente_anonimo.post(
        "/registro", data={**NUEVA, "contrasena2": "otra-cosa-larga"}))
    assert NUEVA["contrasena"] not in html
    # Pero el usuario escrito si vuelve: reescribirlo seria absurdo.
    assert NUEVA["usuario"] in html


def test_sin_nombre_visible_se_usa_el_usuario(cliente_anonimo, fichero_usuarios: Path):
    from src.app.auth import Usuarios

    cliente_anonimo.post("/registro", data={**NUEVA, "nombre": ""})
    assert Usuarios(fichero_usuarios).obtener("carla").nombre == "carla"


def test_registrarse_con_sesion_abierta_no_hace_nada(cliente):
    assert cliente.get("/registro").status_code == 302


def test_la_cuenta_nueva_no_ve_lo_de_las_demas(cliente_anonimo, cliente):
    """El aislamiento vale igual para quien acaba de registrarse."""
    cliente_anonimo.post("/registro", data=NUEVA)
    html = _texto(cliente_anonimo.get("/datos/"))
    assert "Borrar" not in html       # no tiene nada propio


# --- Cabecera y salida --------------------------------------------------------
def test_la_cabecera_dice_quien_ha_entrado(cliente):
    html = _texto(cliente.get("/"))
    assert NOMBRE_A in html
    assert "Salir" in html
    # Con sesion no se ofrecen los botones de entrar ni de registrarse.
    assert "Registrarse" not in html


def test_sin_sesion_la_cabecera_ofrece_las_dos_puertas(cliente_anonimo):
    html = _texto(cliente_anonimo.get("/"))
    assert "Iniciar sesión" in html
    assert "Registrarse" in html
    assert "Salir" not in html


def test_la_navegacion_ensena_datos_aunque_no_haya_sesion(cliente_anonimo):
    """Esconderlo dejaria al visitante sin saber que existe ni por que
    registrarse; quien entre sin cuenta acaba en el login, con vuelta."""
    assert "/datos/" in _texto(cliente_anonimo.get("/"))


def test_salir_cierra_la_sesion(cliente):
    assert cliente.post("/logout").status_code == 302
    assert cliente.get("/datos/").status_code == 302
    # Pero el recomendador se le sigue sirviendo.
    assert cliente.get("/").status_code == 200


def test_salir_es_post_no_get(cliente):
    """Con un GET, cualquier <img src="/logout"> de otra pagina echaria fuera."""
    assert cliente.get("/logout").status_code in (302, 405)
    assert cliente.get("/datos/").status_code == 200   # sigue dentro


# --- CSRF ---------------------------------------------------------------------
@pytest.mark.parametrize("ruta, datos", [
    ("/datos/validar", {"paquete": "."}),
    ("/datos/ingerir", {"paquete": ".", "nombre_datos": "X"}),
    ("/datos/reentrenar", {"nombre": "X"}),
    ("/datos/cancelar", {"id": "abc"}),
    ("/datos/borrar-modelo", {"slug": "x"}),
    ("/datos/borrar-conjunto", {"slug": "x"}),
    ("/logout", {}),
])
def test_ningun_post_pasa_sin_token(cliente, ruta, datos):
    assert cliente.post(ruta, data=datos, sin_csrf=True).status_code == 400


def test_el_login_tambien_valida_el_token(cliente_anonimo):
    """Sin esto queda abierto el *login CSRF*: meter a la victima en la cuenta
    del atacante para que lo que haga despues acabe registrado ahi."""
    r = cliente_anonimo.post(
        "/login", data={"usuario": USUARIO_A, "contrasena": CONTRASENA},
        sin_csrf=True)
    assert r.status_code == 400


def test_un_token_inventado_no_cuela(cliente):
    r = cliente.post("/datos/validar",
                     data={"paquete": ".", "csrf_token": "inventado"}, sin_csrf=True)
    assert r.status_code == 400


def test_con_token_el_mismo_post_si_pasa(cliente, tmp_path: Path):
    """La prueba de que el 400 anterior era por el token y no por otra cosa."""
    r = cliente.post("/datos/validar",
                     data={"paquete": str(tmp_path),
                           "csrf_token": token_csrf(cliente)},
                     sin_csrf=True)
    assert r.status_code != 400


def test_los_formularios_llevan_el_campo_oculto(cliente):
    for ruta in ("/datos/",):
        assert 'name="csrf_token"' in _texto(cliente.get(ruta))


def test_el_login_lleva_el_campo_oculto(cliente_anonimo):
    assert 'name="csrf_token"' in _texto(cliente_anonimo.get("/login"))


# --- Cabeceras de seguridad ---------------------------------------------------
@pytest.mark.parametrize("cabecera", sorted(CABECERAS_SEGURIDAD))
def test_las_cabeceras_de_seguridad_estan_en_todas_las_respuestas(cliente, cabecera):
    for ruta in ("/", "/datos/", "/glosario", "/api/sugerencias?q=Me"):
        assert cliente.get(ruta).headers.get(cabecera) == CABECERAS_SEGURIDAD[cabecera]


def test_tambien_en_las_paginas_de_error(cliente_anonimo):
    """Una pagina de error sin cabeceras es una pagina sin proteger."""
    respuesta = cliente_anonimo.get("/login")
    assert respuesta.headers.get("X-Frame-Options") == "DENY"


def test_la_csp_no_permite_scripts_de_fuera(cliente):
    csp = cliente.get("/").headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    # La app no carga NADA externo: si algun dia hiciera falta, que se note aqui.
    assert "unsafe-eval" not in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp


def test_la_cookie_de_sesion_es_httponly(app, cliente_anonimo):
    r = cliente_anonimo.post("/login", data={
        "usuario": USUARIO_A, "contrasena": CONTRASENA})
    galleta = r.headers.get("Set-Cookie", "")
    assert "HttpOnly" in galleta
    assert "SameSite=Lax" in galleta


def test_en_produccion_la_cookie_es_secure(dir_modelos: Path, bd: Path,
                                           fichero_usuarios: Path, monkeypatch):
    monkeypatch.setenv(config_app.ENV_CLAVE, "clave-de-prueba")
    app = crear_app(dir_modelos, db_path=bd, ruta_usuarios=fichero_usuarios,
                    produccion=True, testing=True)
    assert app.config["SESSION_COOKIE_SECURE"] is True


def test_en_local_la_cookie_no_es_secure(app):
    """En local se sirve por HTTP: una cookie Secure no se guardaria y nadie
    podria entrar."""
    assert app.config["SESSION_COOKIE_SECURE"] is False


# --- Error 500 ----------------------------------------------------------------
def test_un_fallo_interno_no_filtra_la_traza(app, monkeypatch):
    """Una traza en pantalla revela rutas absolutas, versiones y estructura."""
    from src.app import rutas

    def reventar(*args, **kwargs):
        raise RuntimeError("secreto: C:/rutas/del/servidor/interno.py")

    monkeypatch.setattr(rutas, "_contexto_base", reventar)
    app.config["PROPAGATE_EXCEPTIONS"] = False
    cliente = app.test_client()
    entrar(cliente)
    respuesta = cliente.get("/")
    assert respuesta.status_code == 500
    cuerpo = _texto(respuesta)
    assert "secreto" not in cuerpo
    assert "Traceback" not in cuerpo
    assert "rutas/del/servidor" not in cuerpo
    assert "error interno" in cuerpo.lower()


def test_un_fallo_interno_si_deja_la_traza_en_el_log(app, monkeypatch, caplog):
    import logging

    from src.app import rutas

    def reventar(*args, **kwargs):
        raise RuntimeError("marca-para-el-log")

    monkeypatch.setattr(rutas, "_contexto_base", reventar)
    app.config["PROPAGATE_EXCEPTIONS"] = False
    cliente = app.test_client()
    entrar(cliente)
    with caplog.at_level(logging.ERROR):
        cliente.get("/")
    assert "marca-para-el-log" in caplog.text
    assert "Traceback" in caplog.text


def test_un_404_sigue_siendo_un_404(cliente):
    """Las excepciones HTTP no se convierten en 500 por el manejador general."""
    assert cliente.get("/ruta/que/no/existe").status_code == 404


# --- Modo produccion: lo que deja de enseñarse --------------------------------
@pytest.fixture
def cliente_produccion(dir_modelos: Path, bd: Path, fichero_usuarios: Path,
                       monkeypatch):
    monkeypatch.setenv(config_app.ENV_CLAVE, "clave-de-prueba")
    app = crear_app(dir_modelos, db_path=bd, ruta_usuarios=fichero_usuarios,
                    produccion=True, testing=True)
    app.test_client_class = ClienteConCSRF
    cliente = app.test_client()
    entrar(cliente)
    return cliente


def test_en_local_si_se_enseñan_las_rutas(cliente_sin_bd, app_sin_bd):
    """En local la ruta sirve: el usuario está delante de esa máquina y la
    necesita para saber dónde tiene que dejar la BD."""
    html = _texto(cliente_sin_bd.get("/datos/"))
    assert str(app_sin_bd.config["APP_SIMILITUD"].db_path) in html


def test_en_produccion_no_se_enseña_esa_misma_ruta(dir_modelos: Path, tmp_path: Path,
                                                   fichero_usuarios: Path, monkeypatch):
    """Una ruta absoluta dice dónde está instalada la app y bajo qué cuenta corre."""
    monkeypatch.setenv(config_app.ENV_CLAVE, "clave-de-prueba")
    sin_bd = tmp_path / "no" / "existe.db"
    app = crear_app(dir_modelos, db_path=sin_bd, ruta_usuarios=fichero_usuarios,
                    produccion=True, testing=True)
    app.test_client_class = ClienteConCSRF
    cliente = app.test_client()
    entrar(cliente)
    html = _texto(cliente.get("/datos/"))
    assert str(sin_bd) not in html
    assert str(sin_bd.parent) not in html
    # Pero se sigue diciendo QUÉ falta: ocultar la ruta no es ocultar el estado.
    assert "No hay base de datos" in html


def test_nunca_se_enseña_la_linea_de_comandos_ni_el_log(cliente_produccion, app):
    """Ya no es cosa de `--produccion`: la interfaz no los enseña en ningún modo.

    Antes se ocultaban solo fuera de local, porque en local servían para repetir
    la tarea a mano. Al quitar la consola de la página dejaron de tener sitio
    donde pintarse, así que ni la plantilla ni la API los sirven nunca. Aquí se
    comprueba en producción; que en local tampoco salgan lo cubre
    `test_rutas_datos.test_la_pagina_no_pinta_ni_la_consola_ni_el_comando`.
    """
    from src.tests.app.integracion.test_rutas_datos import GestorFalso
    from src.app.tareas import Tarea

    class ConTarea(GestorFalso):
        def ultima(self, usuario: str | None = None):
            return Tarea(id="a", tipo="ingerir", titulo="Incorporando",
                         comando=["python", "-m", "x", "--db",
                                  "C:/ruta/secreta/del/servidor.db"],
                         lineas=["escribiendo en C:/ruta/secreta/del/servidor.db"],
                         estado="terminada", codigo=0, usuario=usuario)

    cliente_produccion.application.extensions["tareas"] = ConTarea()
    html = _texto(cliente_produccion.get("/datos/"))
    assert "ruta/secreta" not in html
    # Y tampoco por la API, que es de donde la rellena el JS mientras corre.
    datos = cliente_produccion.get("/datos/tarea").get_json()["tarea"]
    assert "comando" not in datos and "lineas" not in datos


def test_en_produccion_no_se_propone_una_ruta_del_servidor(cliente_produccion):
    html = _texto(cliente_produccion.get("/datos/"))
    assert config_app.PAQUETE_DEFECTO not in html

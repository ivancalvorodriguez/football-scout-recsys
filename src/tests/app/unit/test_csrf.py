"""El token CSRF: de donde sale, donde vive y cuando se rechaza.

Lo importante de este modulo no es que valide, sino DONDE guarda el token: en la
sesion firmada y no en un diccionario de modulo. Un diccionario vive en la
memoria de un interprete, asi que con mas de un proceso el token que emite uno no
lo reconoce el de al lado. Eso se comprueba aqui recreando la app.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app import csrf
from src.app.factoria import crear_app
from src.tests.app.conftest import ClienteConCSRF, entrar

pytestmark = pytest.mark.integracion


def test_el_token_es_estable_dentro_de_la_sesion(app):
    """Regenerarlo en cada respuesta rompe las pestanas abiertas, que es como
    acaban desactivandose estas protecciones por molestas."""
    with app.test_request_context():
        primero = csrf.token()
        assert csrf.token() == primero


def test_el_token_no_es_adivinable(app):
    with app.test_request_context():
        valor = csrf.token()
    assert len(valor) >= 32


def test_dos_sesiones_tienen_tokens_distintos(app):
    a, b = app.test_client(), app.test_client()
    entrar(a)
    entrar(b, "bruno")
    from src.tests.app.conftest import token_csrf

    assert token_csrf(a) != token_csrf(b)


def test_el_token_de_una_sesion_no_vale_en_otra(app):
    """Es el ataque que esto para: el atacante puede hacer que la victima envie
    un formulario, pero no puede leer SU token."""
    from src.tests.app.conftest import token_csrf

    victima, atacante = app.test_client(), app.test_client()
    entrar(victima)
    entrar(atacante, "bruno")
    respuesta = victima.post("/datos/validar",
                             data={"paquete": ".", "csrf_token": token_csrf(atacante)})
    assert respuesta.status_code == 400


def test_el_token_vive_en_la_sesion_no_en_memoria_del_proceso(
    dir_modelos: Path, bd: Path, fichero_usuarios: Path, monkeypatch
):
    """Sobrevive a recrear la app con la MISMA clave de firma.

    Es la prueba de que no hay estado de modulo: si el token estuviera en un
    diccionario del proceso, la app nueva no reconoceria el de la sesion vieja
    — y eso es exactamente lo que pasaria entre dos workers de gunicorn.
    """
    monkeypatch.setenv("SCOUTING_SECRET_KEY", "clave-fija-de-prueba")

    def montar():
        creada = crear_app(dir_modelos, db_path=bd,
                           ruta_usuarios=fichero_usuarios, testing=True)
        creada.test_client_class = ClienteConCSRF
        return creada

    primera = montar()
    cliente = primera.test_client()
    entrar(cliente)
    from src.tests.app.conftest import token_csrf

    token_de_la_primera = token_csrf(cliente)

    # Segunda app, misma clave: se le pasa la cookie de sesion del cliente de la
    # primera, que es lo que haria el navegador al caer en el otro worker.
    segunda = montar()
    cliente_nuevo = segunda.test_client()
    for clave, galleta in cliente._cookies.items():
        cliente_nuevo._cookies[clave] = galleta

    respuesta = cliente_nuevo.post(
        "/datos/validar",
        data={"paquete": ".", "csrf_token": token_de_la_primera},
    )
    # 400 seria «token invalido». Cualquier otra cosa significa que lo acepto.
    assert respuesta.status_code != 400


def test_una_sesion_sin_token_rechaza_el_post(app):
    cliente = app.test_client()
    entrar(cliente)
    with cliente.session_transaction() as sesion:
        sesion.pop(csrf.CLAVE_SESION, None)
    respuesta = cliente.post("/datos/validar",
                             data={"paquete": ".", "csrf_token": "lo-que-sea"},
                             sin_csrf=True)
    assert respuesta.status_code == 400


def test_los_metodos_de_lectura_no_piden_token(app):
    """Un GET que cambiara algo seria el problema, no la falta de token."""
    assert "GET" not in csrf.METODOS_PROTEGIDOS
    assert "HEAD" not in csrf.METODOS_PROTEGIDOS
    assert csrf.METODOS_PROTEGIDOS >= {"POST", "DELETE"}

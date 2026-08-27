"""Cuentas: hash de la contrasena, registro en disco y verificacion.

Lo que se protege aqui no es «que el login funcione» (eso se ve en las pruebas de
integracion), sino las propiedades del almacenamiento: que la contrasena no se
puede recuperar de lo guardado, que dos cuentas con la misma contrasena no
comparten hash, y que un fichero roto deja la app CERRADA en vez de abierta.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app import config as config_app
from src.app.auth import (
    RegistroInvalido,
    Usuario,
    UsuarioDesconocido,
    Usuarios,
    crear_usuario,
    exige_sesion,
    normalizar,
    validar_alta,
    verificar,
)

CONTRASENA = "una-contrasena-larga"


@pytest.fixture
def almacen(tmp_path: Path) -> Usuarios:
    return Usuarios(tmp_path / "usuarios.json")


# --- Hash ---------------------------------------------------------------------
def test_la_contrasena_no_esta_en_lo_que_se_guarda():
    """Ni en claro ni codificada: lo unico que queda es un derivado irreversible."""
    cuenta = crear_usuario("ana", "Ana", CONTRASENA)
    serializado = json.dumps(cuenta.como_dict())
    assert CONTRASENA not in serializado
    assert CONTRASENA.encode("utf-8").hex() not in serializado


def test_dos_cuentas_con_la_misma_contrasena_no_comparten_hash():
    """La sal es por usuario: sin ella, un hash repetido delata la coincidencia
    y una tabla precalculada rompe las dos a la vez."""
    a = crear_usuario("ana", "Ana", CONTRASENA)
    b = crear_usuario("bruno", "Bruno", CONTRASENA)
    assert a.sal != b.sal
    assert a.hash != b.hash


def test_la_contrasena_correcta_verifica():
    assert verificar(crear_usuario("ana", "Ana", CONTRASENA), CONTRASENA)


@pytest.mark.parametrize("intento", ["", "otra-cosa", CONTRASENA + " ",
                                     CONTRASENA.upper()])
def test_cualquier_otra_no_verifica(intento):
    assert not verificar(crear_usuario("ana", "Ana", CONTRASENA), intento)


def test_una_cuenta_inexistente_no_verifica_nada():
    """None no es un atajo: se deriva igual (contra tiempo) y se dice que no."""
    assert not verificar(None, CONTRASENA)


def test_un_registro_corrupto_no_autentica_a_nadie():
    """Un `hash` que no es hexadecimal no debe convertirse en «cualquiera pasa»."""
    roto = Usuario(usuario="ana", nombre="Ana", sal="zz", hash="no-es-hex")
    assert not verificar(roto, CONTRASENA)
    assert not verificar(roto, "")


def test_se_guarda_el_coste_para_poder_subirlo_despues():
    cuenta = crear_usuario("ana", "Ana", CONTRASENA)
    assert cuenta.iteraciones == config_app.PBKDF2_ITERACIONES
    assert cuenta.iteraciones >= 100_000


def test_una_cuenta_con_menos_iteraciones_sigue_verificando():
    """El coste vive en el registro, no en el codigo: subirlo no invalida a nadie."""
    cuenta = crear_usuario("ana", "Ana", CONTRASENA)
    barata = Usuario(usuario="ana", nombre="Ana", sal=cuenta.sal,
                     hash=cuenta.hash, iteraciones=cuenta.iteraciones)
    assert verificar(barata, CONTRASENA)


# --- Nombres de cuenta --------------------------------------------------------
@pytest.mark.parametrize("escrito", ["Ana", " ana ", "ANA", "aNa\t"])
def test_el_nombre_de_cuenta_se_normaliza(escrito):
    """«Ana» y «ana» no pueden ser dos cuentas: es suplantacion por confusion."""
    assert normalizar(escrito) == "ana"


def test_se_guarda_normalizado_y_se_encuentra_como_se_escriba(almacen: Usuarios):
    almacen.guardar(crear_usuario("  Ana  ", "Ana Ruiz", CONTRASENA))
    assert almacen.obtener("ANA").nombre == "Ana Ruiz"
    assert almacen.obtener("ana").usuario == "ana"


def test_sin_nombre_visible_se_usa_el_de_la_cuenta():
    assert crear_usuario("ana", "   ", CONTRASENA).nombre == "ana"


# --- Registro en disco --------------------------------------------------------
def test_sin_fichero_no_hay_cuentas(almacen: Usuarios):
    """El fallo seguro es el que no deja entrar, no el que abre la puerta."""
    assert almacen.todos() == []
    assert almacen.obtener("ana") is None
    assert not almacen.hay_cuentas


def test_un_fichero_ilegible_tampoco_abre_la_puerta(tmp_path: Path):
    ruta = tmp_path / "usuarios.json"
    ruta.write_text("{esto no es json", encoding="utf-8")
    assert Usuarios(ruta).todos() == []


def test_un_json_con_otra_forma_no_rompe(tmp_path: Path):
    ruta = tmp_path / "usuarios.json"
    ruta.write_text('["una", "lista"]', encoding="utf-8")
    assert Usuarios(ruta).todos() == []


def test_guardar_y_releer(almacen: Usuarios):
    almacen.guardar(crear_usuario("ana", "Ana Ruiz", CONTRASENA))
    assert verificar(almacen.obtener("ana"), CONTRASENA)


def test_se_relee_del_disco_en_cada_consulta(almacen: Usuarios):
    """Dar de alta con el CLI tiene que valer sin reiniciar el servidor."""
    assert almacen.obtener("bruno") is None
    Usuarios(almacen.ruta).guardar(crear_usuario("bruno", "Bruno", CONTRASENA))
    assert almacen.obtener("bruno") is not None


def test_guardar_conserva_las_demas_cuentas(almacen: Usuarios):
    almacen.guardar(crear_usuario("ana", "Ana", CONTRASENA))
    almacen.guardar(crear_usuario("bruno", "Bruno", CONTRASENA))
    assert [c.usuario for c in almacen.todos()] == ["ana", "bruno"]


def test_guardar_dos_veces_la_misma_cuenta_la_reemplaza(almacen: Usuarios):
    almacen.guardar(crear_usuario("ana", "Ana", CONTRASENA))
    almacen.guardar(crear_usuario("ana", "Ana", "otra-contrasena-larga"))
    assert len(almacen.todos()) == 1
    assert not verificar(almacen.obtener("ana"), CONTRASENA)
    assert verificar(almacen.obtener("ana"), "otra-contrasena-larga")


def test_el_fichero_es_utf8_legible(almacen: Usuarios):
    """Se administra a mano de vez en cuando: tiene que poder leerse."""
    almacen.guardar(crear_usuario("ivan", "Iván Calvo Rodríguez", CONTRASENA))
    datos = json.loads(almacen.ruta.read_text(encoding="utf-8"))
    assert datos["usuarios"]["ivan"]["nombre"] == "Iván Calvo Rodríguez"


def test_no_queda_ningun_temporal(almacen: Usuarios):
    """Se escribe por temporal + `os.replace` para no truncar el fichero."""
    almacen.guardar(crear_usuario("ana", "Ana", CONTRASENA))
    assert [p.name for p in almacen.ruta.parent.iterdir()] == [almacen.ruta.name]


def test_borrar_quita_la_cuenta(almacen: Usuarios):
    almacen.guardar(crear_usuario("ana", "Ana", CONTRASENA))
    almacen.guardar(crear_usuario("bruno", "Bruno", CONTRASENA))
    almacen.borrar("ANA")
    assert [c.usuario for c in almacen.todos()] == ["bruno"]


def test_borrar_una_cuenta_que_no_esta(almacen: Usuarios):
    with pytest.raises(UsuarioDesconocido):
        almacen.borrar("fantasma")


# --- Que exige sesion y que no ------------------------------------------------
@pytest.mark.parametrize("ruta", [
    "/", "/similares", "/jugador/10", "/equipo/1", "/glosario",
    "/api/similares", "/api/sugerencias", "/static/estilo.css",
    "/login", "/registro", "/logout",
])
def test_el_recomendador_entero_se_sirve_sin_cuenta(ruta):
    """La consulta es publica: buscar no escribe nada ni cuesta mas que una
    peticion. Lo que hace falta cuenta es GESTIONAR."""
    assert not exige_sesion(ruta)


@pytest.mark.parametrize("ruta", [
    "/datos", "/datos/", "/datos/carpetas", "/datos/ingerir",
    "/datos/reentrenar", "/datos/borrar-modelo", "/datos/tarea",
])
def test_la_gestion_de_datos_y_modelos_si_la_exige(ruta):
    """Lanza subprocesos, copia bases de datos y ocupa disco del servidor."""
    assert exige_sesion(ruta)


@pytest.mark.parametrize("ruta", ["/datos-publicos", "/datosfalso", "/datoss"])
def test_el_prefijo_privado_cierra_en_barra(ruta):
    """Un `startswith` a secas protegeria rutas que nadie decidio proteger, y un
    prefijo mal escrito dejaria `/datos` abierto."""
    assert not exige_sesion(ruta)


# --- Validacion del alta ------------------------------------------------------
def test_un_alta_correcta_devuelve_el_usuario_normalizado():
    assert validar_alta("  Ana  ", CONTRASENA, CONTRASENA) == "ana"


@pytest.mark.parametrize("usuario", ["", "   ", "-empieza-por-guion", "con espacio",
                                     "acentuadó", "con/barra", "a" * 40])
def test_nombres_de_cuenta_que_no_valen(usuario):
    with pytest.raises(RegistroInvalido):
        validar_alta(usuario, CONTRASENA, CONTRASENA)


@pytest.mark.parametrize("usuario", ["ana", "ana.ruiz", "ana_ruiz", "ana-ruiz", "u2"])
def test_nombres_de_cuenta_que_si_valen(usuario):
    assert validar_alta(usuario, CONTRASENA, CONTRASENA) == usuario


def test_una_contrasena_corta_no_vale():
    corta = "a" * (config_app.MIN_LARGO_CONTRASENA - 1)
    with pytest.raises(RegistroInvalido, match="al menos"):
        validar_alta("ana", corta, corta)


def test_las_dos_contrasenas_tienen_que_coincidir():
    with pytest.raises(RegistroInvalido, match="no coinciden"):
        validar_alta("ana", CONTRASENA, CONTRASENA + "x")


def test_el_mensaje_de_error_no_lleva_la_contrasena():
    """Se le ensena tal cual al usuario y puede acabar en una captura de pantalla."""
    with pytest.raises(RegistroInvalido) as excinfo:
        validar_alta("ana", CONTRASENA, "otra")
    assert CONTRASENA not in str(excinfo.value)

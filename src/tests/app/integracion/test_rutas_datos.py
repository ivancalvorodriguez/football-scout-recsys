"""Sección «Datos»: qué se muestra, qué se valida y qué se llega a lanzar.

Las tareas NO se ejecutan de verdad aquí (eso ya lo cubren `test_tareas.py` y el
e2e del flujo incremental): se sustituye el ejecutor por uno que solo apunta lo
que se le pide. Lo que se comprueba es el pegamento — que el formulario se
traduce a los argumentos correctos del CLI y que un formulario malo no llega a
lanzar nada.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.app import config as config_app
from src.app.factoria import crear_app
from src.app.rutas_datos import resumen_bd, resumen_modelos
from src.app.tareas import Tarea, TareaEnCurso
from src.tests.app.conftest import (
    USUARIO_A,
    ClienteConCSRF,
    VARIANTE_NOMBRE,
    VARIANTE_SLUG,
    entrar,
)

pytestmark = pytest.mark.integracion


@pytest.fixture
def app(dir_modelos: Path, bd: Path, tmp_path: Path, fichero_usuarios: Path):
    """App sobre COPIAS de los modelos y de la BD: aquí se escribe de verdad.

    Entrenar un modelo reserva su carpeta y crear un conjunto de datos copia la
    BD, las dos cosas antes de lanzar la tarea. Sobre los fixtures de sesión,
    cada prueba le dejaría a la siguiente un modelo o un conjunto de más.
    """
    modelos = tmp_path / "modelo"
    shutil.copytree(dir_modelos, modelos)
    copia_bd = tmp_path / "db" / bd.name
    copia_bd.parent.mkdir(parents=True)
    shutil.copy2(bd, copia_bd)
    creada = crear_app(modelos, db_path=copia_bd, ruta_usuarios=fichero_usuarios,
                       testing=True)
    creada.test_client_class = ClienteConCSRF
    return creada


class GestorFalso:
    """Sustituto del ejecutor: registra las llamadas en vez de lanzar procesos."""

    def __init__(self, ocupado: bool = False, usuario_ocupante: str = "otro") -> None:
        self.llamadas: list[tuple[str, str, list[str], str | None]] = []
        self.ocupado = ocupado
        self.usuario_ocupante = usuario_ocupante

    def lanzar(self, tipo: str, titulo: str, argumentos: list[str],
               usuario: str | None = None) -> Tarea:
        if self.ocupado:
            raise TareaEnCurso("Ya hay una tarea en curso (Reentrenando).")
        self.llamadas.append((tipo, titulo, argumentos, usuario))
        return Tarea(id="x", tipo=tipo, titulo=titulo, comando=["python"],
                     usuario=usuario)

    def ultima(self, usuario: str | None = None):
        return None

    def en_curso(self):
        return None

    def hay_alguna_en_curso(self) -> bool:
        return self.ocupado


@pytest.fixture
def gestor(app) -> GestorFalso:
    falso = GestorFalso()
    app.extensions["tareas"] = falso
    return falso


@pytest.fixture
def paquete(tmp_path: Path) -> Path:
    """Una carpeta cualquiera: la ruta solo tiene que existir para pasar el filtro."""
    destino = tmp_path / "paquete"
    destino.mkdir()
    return destino


def _args(gestor: GestorFalso) -> list[str]:
    assert gestor.llamadas, "no se lanzó ninguna tarea"
    return gestor.llamadas[-1][2]


def _valor(args: list[str], opcion: str) -> str | None:
    return args[args.index(opcion) + 1] if opcion in args else None


# --- Panel --------------------------------------------------------------------
def test_el_panel_responde(cliente):
    r = cliente.get("/datos/")
    assert r.status_code == 200
    assert "Datos y modelos" in r.get_data(as_text=True)


def test_el_panel_resume_la_base_de_datos(cliente):
    html = cliente.get("/datos/").get_data(as_text=True)
    assert "La Liga" in html and "Ligue 1" in html
    assert "jugadores" in html and "equipos" in html


def test_el_panel_lista_los_modelos_con_su_fecha(cliente):
    html = cliente.get("/datos/").get_data(as_text=True)
    # La etiqueta sale de la normalizacion que sirva cada entidad, que cambia al
    # promover modelos: se deriva de la config en vez de fijar una redaccion.
    for entidad, (_, normalizacion) in config_app.MODELO_BASE.items():
        assert config_app.ETIQUETA_NORMALIZACION[normalizacion] in html
    assert VARIANTE_NOMBRE in html          # el reentrenado, con su nombre
    assert "Entrenar" in html


def test_el_panel_funciona_sin_base_de_datos(cliente_sin_bd):
    """La BD es opcional en toda la app; esta página no puede ser la excepción."""
    r = cliente_sin_bd.get("/datos/")
    assert r.status_code == 200
    assert "No hay base de datos" in r.get_data(as_text=True)


def test_hay_enlace_a_la_seccion_desde_cualquier_pagina(cliente):
    assert "/datos/" in cliente.get("/").get_data(as_text=True)


# --- Resúmenes ----------------------------------------------------------------
def test_resumen_bd_cuenta_lo_que_hay(bd: Path):
    resumen = resumen_bd(bd)
    assert resumen["partidos"] == 4
    assert resumen["jugadores"] == 7
    assert resumen["equipos"] == 3
    assert {liga["competicion"] for liga in resumen["ligas"]} == {"La Liga", "Ligue 1"}


def test_resumen_bd_sin_fichero_es_none(tmp_path: Path):
    assert resumen_bd(tmp_path / "no_existe.db") is None


def test_resumen_bd_de_un_fichero_que_no_es_sqlite(tmp_path: Path):
    ruta = tmp_path / "falsa.db"
    ruta.write_text("esto no es una base de datos", encoding="utf-8")
    assert resumen_bd(ruta) is None


def test_resumen_modelos_sin_nada_construido(tmp_path: Path):
    """El base sin artefactos no es un modelo a medias: no se ha construido."""
    from src.app.catalogo import Catalogo
    from src.app.conjuntos import CatalogoDatos
    from src.app.fuentes import Fuentes

    vacio = tmp_path / "vacio"
    assert resumen_modelos(
        Catalogo(vacio), CatalogoDatos(tmp_path / "no.db"),
        Fuentes(tmp_path / "no.db"), USUARIO_A,
    ) == []


def test_el_panel_sin_modelos_lo_dice(bd: Path, tmp_path: Path,
                                     fichero_usuarios: Path):
    app = crear_app(tmp_path / "vacio", db_path=bd,
                    ruta_usuarios=fichero_usuarios, testing=True)
    app.test_client_class = ClienteConCSRF
    cliente = app.test_client()
    entrar(cliente)
    assert "No hay modelos en" in cliente.get("/datos/").get_data(as_text=True)


def test_resumen_modelos_agrupa_por_modelo(app):
    """Uno por nombre, con sus dos artefactos (jugador y equipo) fechados."""
    modelos = resumen_modelos(
        app.extensions["catalogo"], app.extensions["datos"],
        app.extensions["fuentes"], USUARIO_A,
    )
    assert [m["variante"].slug for m in modelos] == ["base", VARIANTE_SLUG]
    for m in modelos:
        assert {a["clave"].entidad for a in m["artefactos"]} == {"jugador", "equipo"}
        assert all(a["fecha"] is not None for a in m["artefactos"])


# --- Validar / ingerir --------------------------------------------------------
def test_validar_lanza_solo_validar(cliente, gestor: GestorFalso, paquete: Path):
    r = cliente.post("/datos/validar", data={"paquete": str(paquete)})
    assert r.status_code == 302
    args = _args(gestor)
    assert gestor.llamadas[-1][0] == "validar"
    assert "--solo-validar" in args
    assert _valor(args, "--paquete") == str(paquete)


def test_ingerir_no_pide_solo_validar(cliente, gestor: GestorFalso, paquete: Path):
    cliente.post("/datos/ingerir",
                 data={"paquete": str(paquete), "nombre_datos": "Ampliado"})
    args = _args(gestor)
    assert "--solo-validar" not in args
    assert "--db" in args


def test_los_filtros_opcionales_viajan_al_cli(cliente, gestor: GestorFalso, paquete: Path):
    cliente.post("/datos/ingerir", data={
        "paquete": str(paquete),
        "nombre_datos": "Ampliado",
        "competicion": "1. Bundesliga",
        "temporada": "2015/2016",
        "partidos": "3890561,3890505",
    })
    args = _args(gestor)
    assert _valor(args, "--competicion") == "1. Bundesliga"
    assert _valor(args, "--temporada") == "2015/2016"
    assert _valor(args, "--partidos") == "3890561,3890505"


def test_los_filtros_vacios_no_se_mandan(cliente, gestor: GestorFalso, paquete: Path):
    cliente.post("/datos/ingerir", data={
        "paquete": str(paquete), "nombre_datos": "Ampliado",
        "competicion": "  ", "temporada": ""})
    args = _args(gestor)
    assert "--competicion" not in args and "--temporada" not in args


def test_la_casilla_de_omitir_invalidos_se_traduce(cliente, gestor: GestorFalso,
                                                   paquete: Path):
    cliente.post("/datos/ingerir", data={
        "paquete": str(paquete), "nombre_datos": "Ampliado",
        "omitir_invalidos": "on"})
    assert "--omitir-invalidos" in _args(gestor)


def test_una_carpeta_inexistente_no_lanza_nada(cliente, gestor: GestorFalso,
                                               tmp_path: Path):
    r = cliente.post("/datos/ingerir", data={"paquete": str(tmp_path / "fantasma")})
    assert r.status_code == 400
    assert "No existe la carpeta" in r.get_data(as_text=True)
    assert gestor.llamadas == []


def test_un_fichero_no_vale_como_paquete(cliente, gestor: GestorFalso, tmp_path: Path):
    fichero = tmp_path / "suelto.json"
    fichero.write_text("{}", encoding="utf-8")
    r = cliente.post("/datos/ingerir", data={"paquete": str(fichero)})
    assert r.status_code == 400
    assert gestor.llamadas == []


def test_sin_carpeta_se_pide(cliente, gestor: GestorFalso):
    r = cliente.post("/datos/ingerir", data={"paquete": "   "})
    assert r.status_code == 400
    assert "Indica la carpeta" in r.get_data(as_text=True)
    assert gestor.llamadas == []


def test_lo_escrito_sobrevive_a_la_redireccion(cliente, gestor: GestorFalso,
                                               paquete: Path):
    """Validar y después incorporar sin volver a teclear el filtro."""
    r = cliente.post("/datos/validar", data={
        "paquete": str(paquete), "competicion": "1. Bundesliga",
        "temporada": "2015/2016"})
    assert "competicion=1.+Bundesliga" in r.headers["Location"]

    html = cliente.get(r.headers["Location"]).get_data(as_text=True)
    assert 'value="1. Bundesliga"' in html
    assert 'value="2015/2016"' in html


def test_sin_nada_escrito_se_propone_el_paquete_por_defecto(cliente, gestor):

    html = cliente.get("/datos/").get_data(as_text=True)
    assert f'value="{config_app.PAQUETE_DEFECTO}"' in html


# --- Entrenar un modelo nuevo -------------------------------------------------
def test_reentrenar_apunta_a_lo_que_sirve_la_app(cliente, gestor: GestorFalso, app):
    """Debe entrenar sobre la BD que la app está sirviendo, partiendo del base."""
    cliente.post("/datos/reentrenar", data={"nombre": "Prueba"})
    args = _args(gestor)
    ajustes = app.config["APP_SIMILITUD"]
    assert _valor(args, "--db") == str(ajustes.db_path)
    assert _valor(args, "--modelos") == str(ajustes.model_dir)


def test_el_resultado_va_a_la_carpeta_del_modelo_nuevo(cliente, gestor: GestorFalso, app):
    """El origen no se pisa: se puede volver a él desde el buscador."""
    cliente.post("/datos/reentrenar", data={"nombre": "Con la Bundesliga"})
    args = _args(gestor)
    catalogo = app.extensions["catalogo"]
    assert _valor(args, "--out") == str(catalogo.dir_modelo("con-la-bundesliga"))
    assert _valor(args, "--out") != _valor(args, "--modelos")
    assert catalogo.variante("con-la-bundesliga", USUARIO_A).nombre == "Con la Bundesliga"


def test_siempre_las_dos_entidades_y_siempre_warm(cliente, gestor: GestorFalso):
    """No se pregunta ni la entidad ni cómo ajustar: `--servibles` y warm start."""
    cliente.post("/datos/reentrenar", data={"nombre": "Prueba"})
    args = _args(gestor)
    assert "--servibles" in args
    assert "--frio" not in args and "--entidad" not in args


def test_la_pagina_no_ofrece_ajuste_en_frio(cliente):
    html = cliente.get("/datos/").get_data(as_text=True)
    assert 'name="frio"' not in html and 'name="verificar"' not in html


def test_se_puede_partir_de_un_modelo_reentrenado(cliente, gestor: GestorFalso, app):
    cliente.post("/datos/reentrenar",
                 data={"nombre": "Encadenado", "origen": VARIANTE_SLUG})
    args = _args(gestor)
    catalogo = app.extensions["catalogo"]
    assert _valor(args, "--modelos") == str(catalogo.dir_modelo(VARIANTE_SLUG))


def test_el_desplegable_de_origen_solo_sale_si_hay_donde_elegir(cliente):
    """Con el base y el reentrenado del fixture hay dos: se pregunta."""
    html = cliente.get("/datos/").get_data(as_text=True)
    assert 'name="origen"' in html and VARIANTE_NOMBRE in html


def test_sin_nombre_no_se_lanza_nada(cliente, gestor: GestorFalso):
    r = cliente.post("/datos/reentrenar", data={"nombre": "   "})
    assert r.status_code == 400
    assert gestor.llamadas == []


def test_un_nombre_repetido_no_se_lanza(cliente, gestor: GestorFalso):
    r = cliente.post("/datos/reentrenar", data={"nombre": VARIANTE_NOMBRE})
    assert r.status_code == 400
    assert "Ya hay un modelo" in r.get_data(as_text=True)
    assert gestor.llamadas == []


def test_un_origen_inventado_se_rechaza(cliente, gestor: GestorFalso):
    """El origen acaba siendo una ruta en la línea de comandos."""
    r = cliente.post("/datos/reentrenar",
                     data={"nombre": "Prueba", "origen": "../../fuera"})
    assert r.status_code == 400
    assert gestor.llamadas == []


def test_con_una_tarea_en_curso_el_nombre_queda_libre(cliente, app):
    """Si no se llega a lanzar, no puede quedar un modelo fantasma bloqueándolo."""
    app.extensions["tareas"] = GestorFalso(ocupado=True)
    r = cliente.post("/datos/reentrenar", data={"nombre": "Reintentable"})
    assert r.status_code == 409
    assert not app.extensions["catalogo"].dir_modelo("reintentable").exists()


# --- Conjuntos de datos -------------------------------------------------------
def test_ingerir_crea_un_conjunto_nuevo_y_apunta_ahi(cliente, gestor: GestorFalso,
                                                     app, paquete: Path):
    """La ingesta escribe en la COPIA, nunca en el conjunto del que se parte."""
    cliente.post("/datos/ingerir",
                 data={"paquete": str(paquete), "nombre_datos": "Con la Bundesliga"})
    args = _args(gestor)
    catalogo_datos = app.extensions["datos"]
    nuevo = catalogo_datos.conjunto("con-la-bundesliga", USUARIO_A)
    assert _valor(args, "--db") == str(nuevo.ruta)
    assert _valor(args, "--db") != str(app.config["APP_SIMILITUD"].db_path)
    # La copia ES el estado anterior: la de seguridad de `ingesta` sobra.
    assert "--sin-copia" in args


def test_la_copia_llega_completa_antes_de_lanzar(cliente, gestor: GestorFalso, app,
                                                 paquete: Path):
    cliente.post("/datos/ingerir",
                 data={"paquete": str(paquete), "nombre_datos": "Ampliado"})
    assert resumen_bd(app.extensions["datos"].ruta("ampliado"))["partidos"] == 4


def test_se_puede_partir_de_un_conjunto_ya_creado(cliente, gestor: GestorFalso, app,
                                                  paquete: Path):
    app.extensions["datos"].crear("Primero", USUARIO_A)
    cliente.post("/datos/ingerir", data={
        "paquete": str(paquete), "nombre_datos": "Segundo", "datos_origen": "primero"})
    assert app.extensions["datos"].conjunto("segundo", USUARIO_A).origen == "primero"


def test_sin_nombre_no_se_crea_ningun_conjunto(cliente, gestor: GestorFalso, app,
                                               paquete: Path):
    r = cliente.post("/datos/ingerir",
                     data={"paquete": str(paquete), "nombre_datos": "  "})
    assert r.status_code == 400
    assert gestor.llamadas == []
    assert [c.slug for c in app.extensions["datos"].conjuntos(USUARIO_A)] == ["base"]


def test_un_nombre_de_conjunto_repetido_se_rechaza(cliente, gestor: GestorFalso, app,
                                                   paquete: Path):
    app.extensions["datos"].crear("Ampliado", USUARIO_A)
    r = cliente.post("/datos/ingerir",
                     data={"paquete": str(paquete), "nombre_datos": "ampliado"})
    assert r.status_code == 400
    assert "Ya hay un conjunto de datos" in r.get_data(as_text=True)
    assert gestor.llamadas == []


def test_un_conjunto_de_origen_inventado_se_rechaza(cliente, gestor: GestorFalso,
                                                    paquete: Path):
    """El origen acaba siendo una ruta en la línea de comandos."""
    r = cliente.post("/datos/ingerir", data={
        "paquete": str(paquete), "nombre_datos": "X", "datos_origen": "../../fuera"})
    assert r.status_code == 400
    assert gestor.llamadas == []


def test_validar_no_crea_ningun_conjunto(cliente, gestor: GestorFalso, app,
                                         paquete: Path):
    """«Solo validar» no escribe nada: tampoco un conjunto vacío."""
    r = cliente.post("/datos/validar", data={"paquete": str(paquete)})
    assert r.status_code == 302
    assert [c.slug for c in app.extensions["datos"].conjuntos(USUARIO_A)] == ["base"]


def test_el_panel_lista_los_conjuntos_con_sus_cifras(cliente, app):
    app.extensions["datos"].crear("Con la Bundesliga", USUARIO_A)
    html = cliente.get("/datos/").get_data(as_text=True)
    assert "Conjuntos de datos" in html
    assert "Con la Bundesliga" in html
    assert "<strong>4</strong> partidos" in html


def test_entrenar_usa_el_conjunto_elegido(cliente, gestor: GestorFalso, app):
    """El modelo se entrena sobre esa BD y lo deja apuntado para servirse luego."""
    nuevo = app.extensions["datos"].crear("Con la Bundesliga", USUARIO_A)
    cliente.post("/datos/reentrenar",
                 data={"nombre": "M2", "datos_origen": "con-la-bundesliga"})
    assert _valor(_args(gestor), "--db") == str(nuevo.ruta)
    assert app.extensions["catalogo"].variante("m2", USUARIO_A).datos == "con-la-bundesliga"


def test_entrenar_sobre_un_conjunto_inventado_se_rechaza(cliente, gestor: GestorFalso,
                                                         app):
    r = cliente.post("/datos/reentrenar",
                     data={"nombre": "M2", "datos_origen": "fantasma"})
    assert r.status_code == 400
    assert gestor.llamadas == []
    # Y no queda el modelo reservado ocupando el nombre.
    assert not app.extensions["catalogo"].dir_modelo("m2").exists()


def test_por_defecto_se_entrena_sobre_el_base(cliente, gestor: GestorFalso, app):
    cliente.post("/datos/reentrenar", data={"nombre": "M2"})
    ajustes = app.config["APP_SIMILITUD"]
    assert _valor(_args(gestor), "--db") == str(ajustes.db_path)


# --- Explorador de carpetas ---------------------------------------------------
def test_el_explorador_lista_solo_carpetas(cliente, tmp_path: Path):
    raiz = tmp_path / "arbol"
    (raiz / "una").mkdir(parents=True)
    (raiz / "otra").mkdir()
    (raiz / "suelto.json").write_text("{}", encoding="utf-8")
    datos = cliente.get(f"/datos/carpetas?ruta={raiz}").get_json()
    assert [c["nombre"] for c in datos["carpetas"]] == ["otra", "una"]
    assert datos["ruta"] == str(raiz.resolve())


def test_el_explorador_deja_subir_y_cambiar_de_unidad(cliente, tmp_path: Path):
    hija = tmp_path / "dentro"
    hija.mkdir()
    datos = cliente.get(f"/datos/carpetas?ruta={hija}").get_json()
    assert datos["padre"] == str(tmp_path.resolve())
    assert datos["unidades"]      # sin esto no se puede saltar de disco


def test_el_explorador_marca_un_paquete_statsbomb(cliente, tmp_path: Path):
    """Pista para el usuario; la validación de verdad la hace «Solo validar»."""
    raiz = tmp_path / "paquete"
    (raiz / "matches").mkdir(parents=True)
    (raiz / "events").mkdir()
    (raiz / "competitions.json").write_text("[]", encoding="utf-8")
    assert cliente.get(f"/datos/carpetas?ruta={raiz}").get_json()["es_paquete"]
    assert not cliente.get(
        f"/datos/carpetas?ruta={raiz / 'matches'}").get_json()["es_paquete"]


def test_el_explorador_con_una_ruta_que_no_existe(cliente, tmp_path: Path):
    r = cliente.get(f"/datos/carpetas?ruta={tmp_path / 'fantasma'}")
    assert r.status_code == 404
    assert "No existe la carpeta" in r.get_json()["error"]


def test_el_explorador_no_devuelve_ficheros(cliente, tmp_path: Path):
    """Solo navega directorios: nunca sirve el contenido de un fichero."""
    raiz = tmp_path / "arbol"
    raiz.mkdir()
    (raiz / "secreto.txt").write_text("nada", encoding="utf-8")
    datos = cliente.get(f"/datos/carpetas?ruta={raiz}").get_json()
    assert all("secreto" not in c["nombre"] for c in datos["carpetas"])


def test_el_boton_de_examinar_esta_en_la_pagina(cliente):
    html = cliente.get("/datos/").get_data(as_text=True)
    assert "abrir-explorador" in html and "explorador.js" in html


# --- Concurrencia -------------------------------------------------------------
def test_con_una_tarea_en_curso_se_avisa(cliente, app, paquete: Path):
    app.extensions["tareas"] = GestorFalso(ocupado=True)
    r = cliente.post("/datos/ingerir",
                     data={"paquete": str(paquete), "nombre_datos": "Ampliado"})
    assert r.status_code == 409
    assert "en curso" in r.get_data(as_text=True)
    # Y la copia que se acababa de hacer no queda ocupando el nombre.
    assert not app.extensions["datos"].ruta("ampliado").exists()


# --- API de seguimiento -------------------------------------------------------
def test_la_api_de_tarea_responde_sin_tareas(cliente, gestor: GestorFalso):
    assert cliente.get("/datos/tarea").get_json() == {"tarea": None, "ocupado": False}


class ConTarea(GestorFalso):
    """Gestor con una tarea ya terminada, para ver cómo se pinta."""

    def ultima(self, usuario: str | None = None):
        return Tarea(id="abc", tipo="ingerir", titulo="Incorporando",
                     comando=["python", "-m", "x", "--paquete", "sitio"],
                     lineas=["hola"], estado="terminada", codigo=0,
                     usuario=usuario, progreso=1.0, paso="partido 3 de 3")


class ConTareaFallida(GestorFalso):
    """Gestor con una tarea que ha fallado: la página tiene que decir por qué."""

    def ultima(self, usuario: str | None = None):
        return Tarea(id="abc", tipo="ingerir", titulo="Incorporando",
                     comando=["python", "-m", "x"], estado="fallida", codigo=2,
                     lineas=["Error: la carpeta no existe"], usuario=usuario)


def test_la_api_de_tarea_describe_la_ultima(cliente, app):
    app.extensions["tareas"] = ConTarea()
    datos = cliente.get("/datos/tarea").get_json()["tarea"]
    assert datos["id"] == "abc"
    assert datos["titulo"] == "Incorporando"
    assert datos["progreso"] == 1.0
    assert datos["paso"] == "partido 3 de 3"


def test_la_api_de_tarea_no_manda_ni_el_log_ni_el_comando(cliente, app):
    """La página no los pinta, así que mandarlos solo sería exponer el servidor."""
    app.extensions["tareas"] = ConTarea()
    datos = cliente.get("/datos/tarea").get_json()["tarea"]
    assert "lineas" not in datos and "comando" not in datos


def test_la_pagina_no_pinta_ni_la_consola_ni_el_comando(cliente, app):
    """Del subproceso se enseña el avance, no su salida ni cómo se invocó."""
    app.extensions["tareas"] = ConTarea()
    html = cliente.get("/datos/").get_data(as_text=True)
    assert "python -m x --paquete sitio" not in html
    assert "tarea-log" not in html
    assert "hola" not in html


def test_la_pagina_muestra_el_estado_y_el_avance(cliente, app):
    app.extensions["tareas"] = ConTarea()
    html = cliente.get("/datos/").get_data(as_text=True)
    assert "Incorporando" in html
    assert "estado-terminada" in html
    assert "partido 3 de 3" in html


def test_una_tarea_fallida_dice_por_que_en_la_pagina(cliente, app):
    """Es lo que sustituye al volcado del log: sin esto, «fallida» a secas."""
    app.extensions["tareas"] = ConTareaFallida()
    html = cliente.get("/datos/").get_data(as_text=True)
    assert "Error: la carpeta no existe" in html
    assert cliente.get("/datos/tarea").get_json()["tarea"]["error"] == (
        "Error: la carpeta no existe")

"""Registro persistente de un barrido: nombres estables y metricas acumuladas.

Sin este modulo, `barrido.py` nombraba las combinaciones `v01`, `v02`, ... por su
POSICION en el producto cartesiano de `HIPERPARAMETROS`. Como esa lista se EDITA
entre ejecuciones, el mismo nombre designaba configuraciones distintas en cada
una, y relanzar el barrido sobre la misma carpeta era destructivo en tres sitios
a la vez: `vNN/` se sobrescribia con otra configuracion,
`barrido_metricas.csv` se reescribia con SOLO las combinaciones de la ultima
ejecucion, y `resumen_barrido.md` con ella. Ampliar la rejilla obligaba a crear una
carpeta nueva a mano (`barrido_v1`, `barrido_v2`...) y las superficies 3D salian
con los puntos de una unica ejecucion, que es justo lo que impide ver donde esta el
optimo.

La carpeta del barrido pasa a ser ACUMULATIVA:

1. **Nombres por configuracion, no por posicion.** `combinaciones.json` guarda el
   mapa nombre -> valores EFECTIVOS de todos los ejes. Una configuracion ya vista
   recupera su nombre (y con el su carpeta y su cache de modelos); una nueva
   recibe el siguiente numero libre. Dos ejecuciones cualesquiera sobre la misma
   carpeta hablan del mismo `v07`.
2. **Metricas acumuladas.** `barrido_metricas.csv` se funde por
   `CLAVE_METRICA`: lo que la ejecucion actual recalcula sustituye a lo guardado,
   lo demas se conserva. Asi `figuras3d` dibuja la superficie con TODOS los
   puntos evaluados en la carpeta, vengan de una ejecucion o de cinco.
3. **Procedencia.** Cada entrada anota cuando se evaluo, con que BD y con que
   hash de codigo. Acumular numeros producidos con datos o nucleo numerico
   distintos los haria incomparables sin avisar, y un fichero acumulado invita
   precisamente a eso.

La identidad de una combinacion es su configuracion efectiva sobre TODOS los ejes
publicados (los barridos y los que se quedaron en su default), comparada por
`clave`. Por eso el llamador tiene que pasar siempre la union de los ejes vistos
hasta hoy: si un eje desaparece de `HIPERPARAMETROS`, la configuracion sigue
teniendo un valor para el (el default) y tiene que seguir contando para la
identidad, o la misma configuracion recibiria un nombre nuevo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .huella import SUFIJO as SUFIJO_HUELLA

ARCHIVO = "combinaciones.json"
ARCHIVO_METRICAS = "barrido_metricas.csv"
VERSION = 1

# Identidad de una fila de metricas dentro de la carpeta. Lo que la ejecucion
# actual vuelve a calcular sustituye a lo guardado; el resto se conserva. La
# entidad/formulacion/normalizacion no entran porque ya estan determinadas por
# `modelo`: incluirlas solo abriria la puerta a duplicados por discrepancia de
# formato.
CLAVE_METRICA = ("combinacion", "modelo", "fase", "metrica")

# Campo de la procedencia que guarda CUANDO se evaluo una combinacion. El campo
# se llamaba antes de otra manera (`_CAMPO_FECHA_ANTIGUO`); las carpetas escritas
# entonces se migran al leerlas, para no perder la fecha de lo ya evaluado por un
# cambio de nombre. Es la unica razon por la que el nombre viejo sigue escrito en
# el codigo, y puede borrarse cuando no queden carpetas de antes.
CAMPO_FECHA = "ejecucion"
_CAMPO_FECHA_ANTIGUO = "corrida"

_RE_NOMBRE = re.compile(r"^v(\d+)$")
_RE_ADORNO = re.compile(r"[`*]")


# --------------------------------------------------------------------------- #
# Identidad de una configuracion                                               #
# --------------------------------------------------------------------------- #

def _normalizar(valor: object) -> str:
    """Forma canonica de un valor de hiperparametro, para comparar identidades.

    Se compara por TEXTO y no por el objeto, porque la misma configuracion llega
    por caminos con tipos distintos: de `src.similitud.config` llega tipada, de
    `combinaciones.json` pasa por JSON (las tuplas vuelven como listas) y de la
    tabla markdown de un resumen antiguo llega como cadena. Los numeros se
    unifican a float para que `50` y `50.0` —el mismo punto de la rejilla escrito
    de dos maneras— no generen dos combinaciones.
    """
    if isinstance(valor, bool):
        return str(valor)
    if isinstance(valor, (int, float)):
        return repr(float(valor))
    if isinstance(valor, (list, tuple)):
        return repr([_normalizar(v) for v in valor])
    return str(valor)


def clave(config: dict[str, object], ejes_id: list[str] | None = None) -> str:
    """Clave de identidad de una configuracion efectiva.

    `ejes_id` fija sobre QUE ejes se compara. Es necesario cuando el juego de ejes
    cambia entre ejecuciones: comparando cada configuracion con sus propias claves,
    quitar un eje de `HIPERPARAMETROS` haria que la misma configuracion pareciera
    nueva. Con `ejes_id`, los dos lados se recortan al mismo conjunto.
    """
    claves = list(config) if ejes_id is None else list(ejes_id)
    return json.dumps({k: _normalizar(config.get(k)) for k in claves}, sort_keys=True)


def orden_nombre(nombre: str) -> tuple[int, str]:
    """Orden natural de los nombres: `v9` antes que `v10` (no alfabetico)."""
    m = _RE_NOMBRE.match(str(nombre))
    return (int(m.group(1)), "") if m else (10**9, str(nombre))


def _siguiente_nombre(usados: set[str]) -> str:
    numeros = [int(m.group(1)) for m in map(_RE_NOMBRE.match, usados) if m]
    return f"v{(max(numeros) + 1 if numeros else 1):02d}"


# --------------------------------------------------------------------------- #
# Lectura de la tabla markdown (migracion de barridos anteriores)              #
# --------------------------------------------------------------------------- #

def _limpiar(celda: str) -> str:
    """Quita el marcado de la celda markdown (backticks y negritas)."""
    return _RE_ADORNO.sub("", celda).strip()


def _convertir(txt: str) -> object:
    """`"512"` -> 512, `"0.5"` -> 0.5, `"True"` -> True, el resto se queda en texto.

    Los valores llegan como texto desde el markdown; hay que recuperar el tipo
    para poder ORDENAR los ejes numericamente (si no, 1024 iria antes que 256).
    """
    if txt in ("True", "False"):
        return txt == "True"
    for conv in (int, float):
        try:
            return conv(txt)
        except ValueError:
            pass
    return txt


# Columnas de la tabla de combinaciones que NO son ejes de hiperparametro.
_COLUMNAS_NO_EJE = frozenset({"evaluada"})


def leer_markdown(barrido_dir: Path) -> dict[str, dict[str, object]] | None:
    """{combinacion -> {eje: valor}} leido de la tabla de `resumen_barrido.md`.

    Es la unica fuente que tienen los barridos anteriores a este modulo: su
    resumen publica la configuracion efectiva de cada `vNN`. Se usa para migrar
    esas carpetas al registro sin perder la correspondencia (y, en `figuras3d`,
    como respaldo si el registro no esta).
    """
    ruta_md = Path(barrido_dir) / "resumen_barrido.md"
    if not ruta_md.is_file():
        return None

    lineas = ruta_md.read_text(encoding="utf-8").splitlines()
    inicio = next(
        (i for i, ln in enumerate(lineas) if ln.strip().startswith("| combinacion |")),
        None,
    )
    if inicio is None:
        return None

    cabecera = [_limpiar(c) for c in lineas[inicio].strip().strip("|").split("|")]
    ejes = cabecera[1:]
    combos: dict[str, dict[str, object]] = {}
    for ln in lineas[inicio + 2:]:          # +2: salta la linea de separacion
        if not ln.strip().startswith("|"):
            break
        celdas = [_limpiar(c) for c in ln.strip().strip("|").split("|")]
        if len(celdas) != len(cabecera):
            break
        combos[celdas[0]] = {e: _convertir(v) for e, v in zip(ejes, celdas[1:])
                             # La columna de procedencia no es un hiperparametro.
                             if e not in _COLUMNAS_NO_EJE}
    return combos or None


# --------------------------------------------------------------------------- #
# Carga y guardado del registro                                                #
# --------------------------------------------------------------------------- #

def ruta(out_dir: Path) -> Path:
    return Path(out_dir) / ARCHIVO


def vacio() -> dict:
    return {"version": VERSION, "combinaciones": {}}


def _migrar_campo_fecha(registro: dict) -> None:
    """Renombra el campo de la fecha en las entradas escritas con el nombre viejo.

    Se hace al leer para que una carpeta anterior al cambio conserve la fecha en
    que se evaluo cada combinacion (el resumen la publica en la columna
    `evaluada`) y para que el nombre viejo desaparezca del fichero en cuanto se
    vuelva a guardar. No toca nada mas: es un renombrado, no una reevaluacion.
    """
    for entrada in registro.get("combinaciones", {}).values():
        if _CAMPO_FECHA_ANTIGUO in entrada:
            entrada.setdefault(CAMPO_FECHA, entrada[_CAMPO_FECHA_ANTIGUO])
            del entrada[_CAMPO_FECHA_ANTIGUO]


def cargar(out_dir: Path) -> dict:
    """Registro de la carpeta; lo migra desde `resumen_barrido.md` si no existe.

    Un registro ilegible o de otra version aborta en vez de empezar de cero: si
    se ignorase, la numeracion volveria a arrancar en `v01` y las carpetas ya
    existentes quedarian asignadas a configuraciones distintas — exactamente el
    fallo que este modulo evita.
    """
    p = ruta(out_dir)
    if p.is_file():
        try:
            reg = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise SystemExit(f"No puedo leer {p}: {e}. Es el indice de la carpeta "
                             "del barrido; arreglalo o mueve la carpeta antes de "
                             "seguir (renombrarla equivale a empezar un barrido "
                             "nuevo).")
        if reg.get("version") != VERSION:
            raise SystemExit(f"{p} es de la version {reg.get('version')} y este "
                             f"codigo escribe la {VERSION}.")
        reg.setdefault("combinaciones", {})
        _migrar_campo_fecha(reg)
        return reg

    combos = leer_markdown(out_dir)
    if combos:
        print(f"  [migracion] {ruta(out_dir).name} no existe: se reconstruye desde "
              f"resumen_barrido.md ({len(combos)} combinaciones ya conocidas).")
        return {"version": VERSION,
                "combinaciones": {n: {"hiperparametros": c} for n, c in combos.items()}}
    return vacio()


def guardar(out_dir: Path, registro: dict) -> Path:
    p = ruta(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    ordenado = {
        "version": VERSION,
        "combinaciones": {n: registro["combinaciones"][n]
                          for n in sorted(registro["combinaciones"], key=orden_nombre)},
    }
    p.write_text(json.dumps(ordenado, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def hiperparametros(registro: dict) -> dict[str, dict[str, object]]:
    """{nombre -> configuracion efectiva} de todas las combinaciones conocidas."""
    return {n: e.get("hiperparametros", {})
            for n, e in sorted(registro["combinaciones"].items(), key=lambda kv: orden_nombre(kv[0]))}


def _clave_orden(valores: list[object]) -> list[object]:
    """Orden numerico si todos los valores lo son; alfabetico en otro caso."""
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valores):
        return sorted(valores)
    return sorted(valores, key=str)


def ejes_variables(configs: dict[str, dict[str, object]]) -> dict[str, list[object]]:
    """{eje -> valores ordenados} de los hiperparametros que REALMENTE se movieron.

    Un eje con un solo valor (el que se quedo en su default porque no afectaba a
    la rejilla) no es un eje de la rejilla: aportaria una unica fila y haria creer
    que se probo algo que no se probo. Lo usan tanto las superficies 3D como el
    resumen, y tiene que dar lo mismo en los dos.
    """
    todos: dict[str, list[object]] = {}
    for valores in configs.values():
        for eje, v in valores.items():
            todos.setdefault(eje, [])
            if v not in todos[eje]:
                todos[eje].append(v)
    return {eje: _clave_orden(vs) for eje, vs in todos.items() if len(vs) > 1}


def ejes(registro: dict, preferidos: list[str] | None = None) -> list[str]:
    """Union de los ejes de todas las entradas, con `preferidos` primero.

    El orden importa poco salvo para leer las tablas, pero la UNION es lo que
    sostiene la identidad: un eje que hoy no se barre sigue formando parte de la
    configuracion (con su default) y tiene que seguir publicandose.
    """
    vistos = list(preferidos or [])
    for config in hiperparametros(registro).values():
        for eje in config:
            if eje not in vistos:
                vistos.append(eje)
    return vistos


# --------------------------------------------------------------------------- #
# Asignacion de nombres                                                        #
# --------------------------------------------------------------------------- #

def nombrar(
    registro: dict,
    configuraciones: list[dict[str, object]],
    ejes_id: list[str] | None = None,
) -> dict[str, dict]:
    """{nombre -> configuracion} de esta ejecucion, creando las entradas que falten.

    Muta `registro`. Una configuracion ya registrada recupera su nombre, con lo
    que la ejecucion reutiliza su carpeta y su cache de modelos y reescribe SUS
    resultados; una nueva se lleva el siguiente numero libre, sin tocar a las
    demas.
    """
    por_clave = {clave(c, ejes_id): n for n, c in hiperparametros(registro).items()}
    seleccion: dict[str, dict] = {}
    for config in configuraciones:
        k = clave(config, ejes_id)
        nombre = por_clave.get(k)
        if nombre is None:
            nombre = _siguiente_nombre(set(registro["combinaciones"]))
            registro["combinaciones"][nombre] = {}
            por_clave[k] = nombre
        # Se reescribe la configuracion con los valores TIPADOS de esta ejecucion:
        # una entrada migrada desde markdown guarda lo que se pudo reconstruir del
        # texto, y esta es la unica ocasion de mejorarla sin cambiar su identidad.
        registro["combinaciones"][nombre]["hiperparametros"] = dict(config)
        seleccion[nombre] = dict(config)
    return dict(sorted(seleccion.items(), key=lambda kv: orden_nombre(kv[0])))


def anotar(registro: dict, nombres: list[str], procedencia: dict) -> None:
    """Marca las combinaciones evaluadas ahora (fecha, datos, codigo)."""
    for nombre in nombres:
        registro["combinaciones"].setdefault(nombre, {}).update(procedencia)


def procedencias(registro: dict) -> dict[str, dict]:
    """{nombre -> {ejecucion, datos, codigo}} de las entradas que lo declaran."""
    return {n: {k: v for k, v in e.items() if k != "hiperparametros"}
            for n, e in registro["combinaciones"].items()}


def homogeneo(registro: dict) -> bool:
    """True si todo lo acumulado se evaluo con los mismos datos y el mismo codigo.

    Cuando es False las metricas de la carpeta vienen de mundos distintos (se
    reextrajo la BD, se toco el nucleo numerico) y compararlas entre si no
    significa lo que parece. No se borra nada por ello: se avisa, porque la
    decision de rehacerlas es del usuario.
    """
    marcas = {json.dumps([p.get("datos"), p.get("codigo")], sort_keys=True)
              for p in procedencias(registro).values() if p}
    return len(marcas) <= 1


# --------------------------------------------------------------------------- #
# Ejes que faltan en entradas antiguas                                         #
# --------------------------------------------------------------------------- #

def _valor_en_huellas(out_dir: Path, nombre: str, eje: str) -> object | None:
    """Valor de `eje` segun las huellas de los modelos de esa combinacion.

    Es la fuente autoritativa: la huella de cada artefacto registra con que
    hiperparametros se construyo. Solo sirve si todas las que lo mencionan
    coinciden (el alcance por celda hace que muchas ni lo incluyan).
    """
    valores = []
    for p in sorted(Path(out_dir).glob(f"{nombre}/modelo/*{SUFIJO_HUELLA}")):
        try:
            h = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if eje in h.get("hiperparametros", {}):
            valores.append(h["hiperparametros"][eje])
    if not valores:
        return None
    primero = valores[0]
    return primero if all(v == primero for v in valores) else None


def completar_ejes(
    out_dir: Path, registro: dict, ejes_pedidos: list[str], defectos: dict[str, object]
) -> list[str]:
    """Rellena los ejes que una entrada antigua no registro. Devuelve los avisos.

    Pasa cuando se AÑADE un eje a `HIPERPARAMETROS` que antes no existia: las
    combinaciones ya registradas no lo declaran, asi que su clave de identidad no
    podria compararse con la de una configuracion nueva y se duplicarian. El valor
    se busca primero en las huellas de sus modelos (que si lo registran) y solo si
    ahi no aparece se asume el default vigente, avisando: si ese default cambio
    desde entonces, la etiqueta seria incorrecta.
    """
    avisos: list[str] = []
    for nombre, entrada in registro["combinaciones"].items():
        config = entrada.setdefault("hiperparametros", {})
        for eje in ejes_pedidos:
            if eje in config:
                continue
            valor = _valor_en_huellas(out_dir, nombre, eje)
            if valor is None:
                valor = defectos.get(eje)
                avisos.append(f"{nombre}.{eje} = {valor!r} (asumido del default "
                              "actual: no consta ni en el registro ni en las huellas)")
            config[eje] = valor
    return avisos


# --------------------------------------------------------------------------- #
# Metricas acumuladas                                                          #
# --------------------------------------------------------------------------- #

def leer_metricas(out_dir: Path) -> pd.DataFrame:
    """`barrido_metricas.csv` acumulado en la carpeta (vacio si no hay)."""
    p = Path(out_dir) / ARCHIVO_METRICAS
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return _tipos(df)


def _tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Homogeneiza los tipos de las columnas clave.

    `formulacion` y `fase` viajan como texto ("2", "5", "0") pero `read_csv` los
    devuelve como enteros: sin esto, las filas leidas de disco y las recien
    calculadas no se reconocerian como la misma celda al fundirlas.
    """
    if df.empty:
        return df
    df = df.copy()
    for col in ("combinacion", "modelo", "formulacion", "entidad",
                "normalizacion", "fase", "metrica"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df


def acumular(previa: pd.DataFrame, nueva: pd.DataFrame) -> pd.DataFrame:
    """Funde lo guardado con lo recien calculado: gana lo nuevo, se conserva el resto.

    La sustitucion es por `CLAVE_METRICA`, no por combinacion entera: una ejecucion
    acotada con `--formulaciones 5` reescribe las filas de sus modelos F5 y deja
    intactas las F2 que esa misma combinacion tenia de antes.
    """
    previa, nueva = _tipos(previa), _tipos(nueva)
    if previa.empty:
        return nueva
    if nueva.empty:
        return previa
    juntas = pd.concat([previa, nueva], ignore_index=True)
    juntas = juntas.drop_duplicates(subset=list(CLAVE_METRICA), keep="last")
    juntas = juntas.sort_values(
        by=["combinacion", "modelo", "fase", "metrica"],
        key=lambda s: s.map(orden_nombre) if s.name == "combinacion" else s,
    )
    return juntas.reset_index(drop=True)


def escribir_metricas(out_dir: Path, df: pd.DataFrame) -> Path:
    p = Path(out_dir) / ARCHIVO_METRICAS
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p

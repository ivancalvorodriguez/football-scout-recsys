"""Marca como `euclidea` todo lo que se construyo antes de que la distancia fuera un eje.

    python -m src.evaluacion.migrar_distancia [--raiz outputs] [--aplicar]

Hasta ahora habia UNA distancia y no se declaraba en ninguna parte: los
artefactos de `outputs/modelo/`, los CSV de los barridos y las huellas de sus
~1.100 combinaciones no dicen con que geometria se midieron porque solo habia
una. Al convertirla en el cuarto eje del modelo
(`src.similitud.distancias`), ese silencio pasa a ser ambiguo: una fila sin
`distancia` no se distingue de una fila cuya distancia se olvido de anotar.

Este script cierra ese hueco. **No recalcula nada** —no toca ni una matriz S ni
una metrica— y es puramente ADITIVO: escribe `euclidea` donde antes no habia
campo, que es exactamente lo que esos ficheros midieron.

## Que toca

| Fichero | Que le hace |
|---|---|
| `<modelo>.json` (meta del artefacto) | `meta.distancia = "euclidea"` |
| `<modelo>.huella.json` | `celda.distancia = "euclidea"` |
| `barrido_metricas.csv` | columna `distancia` + sufijo en `modelo` |
| `<vNN>/fase*.csv` | columna `distancia` + sufijo en `modelo` |
| `figuras3d/rejilla_3d.csv`, `score.csv` | sufijo en `modelo` |
| `produccion.json` | `distancia` en el linaje de cada artefacto promovido |

## Que NO toca, y por que

- **Los nombres de los ficheros.** La euclidea no lleva sufijo en disco
  (`modelo.stem_artefacto`): renombrar miles de `.npz`/`.warm.npz`/`.huella.json`
  para no cambiar ni un numero seria mucho riesgo a cambio de nada, y la app
  apunta a esos nombres. Los CSV y las tablas SI la declaran siempre, que es
  donde importa que ningun resultado quede sin decir como se midio.
- **Los estados warm (`.warm.npz`).** `EstadoWarm.cargar` ya relee como
  `euclidea` un estado que no declara distancia, asi que reescribir binarios no
  aporta nada.
- **`resumen_barrido.md`.** Es una SALIDA, no un dato: parchear el markdown seria
  arreglar la vista dejando su fuente igual. Lo que hace falta es REGENERARLA, y
  eso es `--regenerar-resumen`: reescribe las tablas desde el CSV ya migrado (con
  la columna `distancia` que hoy publica `barrido._COLUMNAS_ID`) sin construir ni
  evaluar nada. Va aparte de la migracion porque reescribe una salida entera con
  el codigo de hoy, no solo el campo que faltaba.

Correr esto no es obligatorio: `registro._tipos` rellena la distancia al leer una
tabla antigua y `huella._celda` al comparar una huella antigua, asi que una
carpeta sin migrar sigue funcionando. Lo que da la migracion es que el dato este
declarado EN DISCO en vez de asumido al leerlo.

Por defecto SIMULA (dice que haria y no escribe). Con `--aplicar` escribe.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.evaluacion import registro
from src.similitud.distancias import DISTANCIAS_VALIDAS, POR_DEFECTO

# La distancia con la que se construyo todo lo anterior a este eje.
HISTORICA = POR_DEFECTO

# Columnas de un CSV de resultados que identifican al modelo.
_COL_MODELO = "modelo"
_COL_DISTANCIA = "distancia"


@dataclass
class Recuento:
    """Que se ha cambiado, para poder informarlo y para las pruebas."""

    metas: list[Path] = field(default_factory=list)
    huellas: list[Path] = field(default_factory=list)
    csv: list[Path] = field(default_factory=list)
    produccion: list[Path] = field(default_factory=list)
    resumenes: list[Path] = field(default_factory=list)
    intactos: int = 0

    @property
    def total(self) -> int:
        return (len(self.metas) + len(self.huellas) + len(self.csv)
                + len(self.produccion) + len(self.resumenes))


def _leer_json(ruta: Path) -> dict | None:
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return datos if isinstance(datos, dict) else None


def _escribir_json(ruta: Path, datos: dict) -> None:
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def etiquetar(modelo: str, distancia: str = HISTORICA) -> str:
    """`F5_jugador_por_liga` -> `F5_jugador_por_liga_euclidea` (idempotente).

    Si la etiqueta ya termina en una distancia conocida se deja como esta: la
    migracion se puede volver a lanzar sobre una carpeta ya migrada sin
    encadenar sufijos.
    """
    texto = str(modelo)
    if texto.rsplit("_", 1)[-1] in DISTANCIAS_VALIDAS:
        return texto
    return f"{texto}_{distancia}"


# --------------------------------------------------------------------------- #
# Artefactos: meta y huella                                                     #
# --------------------------------------------------------------------------- #

def _es_meta_de_artefacto(datos: dict) -> bool:
    """¿Es el `.json` que acompaña a un `.npz` de modelo?

    Se reconoce por su forma (`formulacion` + `entidad` + `meta`), no por el
    nombre: en la misma carpeta hay otros JSON (huellas, `variante.json`,
    `produccion.json`) y cada uno se migra distinto.
    """
    return (
        "formulacion" in datos and "entidad" in datos
        and isinstance(datos.get("meta"), dict)
    )


def migrar_meta(ruta: Path, aplicar: bool) -> bool:
    """Añade `meta.distancia` al artefacto. True si habia algo que cambiar."""
    datos = _leer_json(ruta)
    if datos is None or not _es_meta_de_artefacto(datos):
        return False
    if datos["meta"].get(_COL_DISTANCIA):
        return False
    datos["meta"][_COL_DISTANCIA] = HISTORICA
    if aplicar:
        _escribir_json(ruta, datos)
    return True


def migrar_huella(ruta: Path, aplicar: bool) -> bool:
    """Añade `celda.distancia` a una huella. True si habia algo que cambiar."""
    datos = _leer_json(ruta)
    if datos is None or not isinstance(datos.get("celda"), dict):
        return False
    if datos["celda"].get(_COL_DISTANCIA):
        return False
    datos["celda"][_COL_DISTANCIA] = HISTORICA
    if aplicar:
        _escribir_json(ruta, datos)
    return True


def migrar_produccion(ruta: Path, aplicar: bool) -> bool:
    """Declara la distancia en el linaje de los artefactos promovidos.

    `produccion.json` no tiene un esquema fijo (es el linaje escrito a mano de
    los modelos que sirve la app), asi que se anota la distancia en la raiz y en
    cualquier diccionario anidado que describa un artefacto —los que declaran
    `formulacion`—, sin tocar el resto.
    """
    datos = _leer_json(ruta)
    if datos is None:
        return False
    tocado = False

    def anotar(nodo):
        nonlocal tocado
        if isinstance(nodo, dict):
            if "formulacion" in nodo and not nodo.get(_COL_DISTANCIA):
                nodo[_COL_DISTANCIA] = HISTORICA
                tocado = True
            for valor in nodo.values():
                anotar(valor)
        elif isinstance(nodo, list):
            for valor in nodo:
                anotar(valor)

    anotar(datos)
    if not datos.get(_COL_DISTANCIA):
        datos[_COL_DISTANCIA] = HISTORICA
        tocado = True
    if tocado and aplicar:
        _escribir_json(ruta, datos)
    return tocado


# --------------------------------------------------------------------------- #
# CSV de resultados                                                             #
# --------------------------------------------------------------------------- #

def migrar_csv(ruta: Path, aplicar: bool) -> bool:
    """Añade la columna `distancia` y el sufijo a `modelo`. True si cambio algo.

    La columna se coloca justo detras de `normalizacion` cuando esa existe, para
    que el CSV se lea con los cuatro ejes del modelo juntos y en el mismo orden
    que las tablas del resumen.
    """
    try:
        df = pd.read_csv(ruta)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return False
    if df.empty or _COL_MODELO not in df.columns:
        return False
    if _COL_DISTANCIA in df.columns and df[_COL_DISTANCIA].notna().all():
        return False

    df[_COL_MODELO] = df[_COL_MODELO].map(etiquetar)
    if _COL_DISTANCIA in df.columns:
        df[_COL_DISTANCIA] = df[_COL_DISTANCIA].fillna(HISTORICA)
    else:
        columnas = list(df.columns)
        df[_COL_DISTANCIA] = HISTORICA
        if "normalizacion" in columnas:
            destino = columnas.index("normalizacion") + 1
            columnas.insert(destino, _COL_DISTANCIA)
        else:
            columnas.append(_COL_DISTANCIA)
        df = df[columnas]
    if aplicar:
        df.to_csv(ruta, index=False)
    return True


# --------------------------------------------------------------------------- #
# Resumen del barrido (vista regenerable)                                       #
# --------------------------------------------------------------------------- #

def regenerar_resumen(barrido_dir: Path, aplicar: bool) -> bool:
    """Reescribe `resumen_barrido.md` desde el CSV acumulado de esa carpeta.

    No es migracion del markdown: es volver a generar la vista con el codigo de
    hoy, que declara la distancia en las tablas (`barrido._COLUMNAS_ID`). Un
    resumen escrito antes de que la geometria fuera un eje identifica cada modelo
    por tres columnas y no dice con que se midio; el CSV de al lado ya lo dice
    (`migrar_csv`), asi que la informacion no hay que inventarla, solo publicarla.

    No construye ni evalua: solo lee `barrido_metricas.csv` + `combinaciones.json`.
    Los ejes de la tabla de combinaciones salen del REGISTRO de la carpeta, no de
    la lista `HIPERPARAMETROS` de hoy: regenerar no puede cambiar la identidad de
    lo que ya esta medido.

    Devuelve True si la carpeta tiene metricas que regenerar.
    """
    # Import perezoso: `barrido` arrastra el motor de evaluacion entero y esta
    # migracion se usa tambien sin querer regenerar nada.
    from src.evaluacion import barrido

    barrido_dir = Path(barrido_dir)
    df = registro.leer_metricas(barrido_dir)
    if df.empty:
        return False
    if not aplicar:
        return True

    reg = registro.cargar(barrido_dir)
    # Las tablas se indexan por el registro de la carpeta: una combinacion medida
    # en el CSV pero sin entrada ahi no tiene columna, y callarlo dejaria un
    # resumen mas pobre que el anterior sin decir por que.
    huerfanas = sorted(set(df["combinacion"]) - set(registro.hiperparametros(reg)))
    if huerfanas:
        print(f"  [aviso] {barrido_dir}: {len(huerfanas)} combinaciones estan en "
              f"{registro.ARCHIVO_METRICAS} pero no en {registro.ARCHIVO} "
              f"({', '.join(huerfanas[:5])}...): quedan fuera de las tablas.")

    barrido.escribir_resumen_barrido(df, reg, barrido_dir, {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "fases": [],
        "bootstrap": 0,
        "ejes": registro.ejes(reg),
        "evaluadas": [],
        "regenerado": True,
    })
    return True


# --------------------------------------------------------------------------- #
# Recorrido                                                                     #
# --------------------------------------------------------------------------- #

def carpetas_de_barrido(raiz: Path) -> list[Path]:
    """Carpetas bajo `raiz` que son la salida de un barrido (tienen su CSV)."""
    return sorted({p.parent for p in Path(raiz).rglob(registro.ARCHIVO_METRICAS)})


def migrar(raiz: Path, aplicar: bool = False,
           resumenes: bool = False) -> Recuento:
    """Recorre `raiz` y migra todo lo que encuentre. Devuelve el recuento.

    `resumenes` regenera ademas los `resumen_barrido.md` que encuentre, DESPUES
    de migrar los CSV: la vista se escribe a partir del dato ya migrado.
    """
    raiz = Path(raiz)
    r = Recuento()

    for ruta in sorted(raiz.rglob("*.json")):
        nombre = ruta.name
        if nombre.endswith(".huella.json"):
            destino, migrador = r.huellas, migrar_huella
        elif nombre == "produccion.json":
            destino, migrador = r.produccion, migrar_produccion
        else:
            destino, migrador = r.metas, migrar_meta
        if migrador(ruta, aplicar):
            destino.append(ruta)
        else:
            r.intactos += 1

    for ruta in sorted(raiz.rglob("*.csv")):
        if migrar_csv(ruta, aplicar):
            r.csv.append(ruta)
        else:
            r.intactos += 1

    if resumenes:
        for carpeta in carpetas_de_barrido(raiz):
            if regenerar_resumen(carpeta, aplicar):
                r.resumenes.append(carpeta / "resumen_barrido.md")

    return r


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Declara `euclidea` en todo lo construido antes de que la "
                    "distancia fuera un eje del modelo. No recalcula nada.")
    p.add_argument("--raiz", type=Path, default=Path("outputs"),
                   help="Carpeta que se recorre (por defecto, outputs/).")
    p.add_argument("--aplicar", action="store_true",
                   help="Escribe los cambios. Sin esto solo simula y los lista.")
    p.add_argument("--regenerar-resumen", action="store_true",
                   help="Reescribe ademas cada resumen_barrido.md desde su CSV "
                        "acumulado, para que las tablas declaren la distancia. "
                        "No construye ni evalua nada.")
    args = p.parse_args(argv)

    if not args.raiz.exists():
        print(f"No existe {args.raiz}: nada que migrar.")
        return 0

    r = migrar(args.raiz, aplicar=args.aplicar, resumenes=args.regenerar_resumen)
    modo = "APLICADO" if args.aplicar else "SIMULACION (usa --aplicar para escribir)"
    print(f"[{modo}] {args.raiz}")
    print(f"  metas de artefacto : {len(r.metas)}")
    print(f"  huellas            : {len(r.huellas)}")
    print(f"  CSV de resultados  : {len(r.csv)}")
    print(f"  linaje produccion  : {len(r.produccion)}")
    if args.regenerar_resumen:
        print(f"  resumenes a rehacer: {len(r.resumenes)}")
        for ruta in r.resumenes:
            print(f"    {ruta}")
    print(f"  ya correctos o ajenos: {r.intactos}")
    print(f"  TOTAL a cambiar    : {r.total}")
    if not args.aplicar and r.total:
        print("\nNada escrito. Repite con --aplicar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

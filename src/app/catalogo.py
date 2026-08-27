"""Modelos servibles: descubrimiento, creación de nuevos y cacheo en memoria.

Un **modelo** de la app es una pareja de artefactos con un nombre: el de jugador
y el de equipo. Cuál es cuál no lo elige el usuario, lo fija
`src.similitud.config.MODELOS_SERVIBLES` (las dos entidades con la Formulación 5
y z-score `por_liga`) y `DISTANCIA_SERVIBLE` (la geometría, hoy `mahalanobis`, que
es la que pone el sufijo del nombre de fichero). El resto de
combinaciones que sabe construir `src.similitud.build` siguen existiendo en disco
y sirven para la comparación experimental del TFG, pero la app no las sirve: aquí
se elige entre MODELOS, no entre formulaciones.

Hay dos clases de modelo, y las dos viven en `outputs/modelo/`:

- el **base**, el de fábrica, en la raíz del directorio (lo deja `build`);
- los **reentrenados desde la interfaz**, cada uno en
  `outputs/modelo/variantes/<slug>/` con su `variante.json` (el nombre que
  escribió el usuario, cuándo se creó y de qué modelo salió).

El layout es el único sitio que conoce esa distinción: cada carpeta contiene
artefactos con los MISMOS nombres de fichero, así que `cargar_modelo` y el
reentrenamiento (`--modelos <origen> --out <destino>`) funcionan sobre cualquiera
sin saber si es el base o uno con nombre.

La carga es perezosa y cacheada: la matriz S de jugadores es de 2640x2640 y
releerla en cada petición sería absurdo. Junto al modelo se cachea su
`MatrizFases` (el perfil por fases de TODAS las entidades), que se calcula aquí y
no en la vista porque los percentiles necesitan el universo entero: es una
propiedad del artefacto, no de la consulta, y comparte con él la invalidación.

La caché se invalida por fecha de modificación del `.npz`, para que reentrenar
con el servidor levantado se refleje sin reiniciarlo (un `stat` por petición).
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.similitud import config as config_similitud
from src.similitud.modelo import ModeloSimilitud, cargar_modelo, stem_artefacto

from . import config
from .fases import MatrizFases, construir_matriz
# Re-exportados: dar de alta un modelo y dar de alta un conjunto de datos siguen
# las mismas reglas de nombre (ver `nombres`).
from .nombres import NombreInvalido, preparar, slug  # noqa: F401


class ModeloNoDisponible(LookupError):
    """No hay ningún artefacto para el modelo pedido."""


class CuotaSuperada(RuntimeError):
    """La cuenta ya tiene tantos modelos como se le permiten."""


# --- Identidad de un modelo ---------------------------------------------------

@dataclass(frozen=True)
class Variante:
    """Un modelo con nombre: el base o uno reentrenado desde la interfaz.

    `entidades` son los tipos de entidad para los que YA tiene artefacto
    completo (`.npz` + `.json`). Puede estar vacío mientras el reentrenamiento
    corre: la carpeta y su `variante.json` se crean antes de lanzar la tarea, así
    que la página de datos puede enseñar el modelo «en construcción» sin que el
    buscador llegue a ofrecerlo.
    """

    slug: str
    nombre: str
    entidades: tuple[str, ...] = ()
    creado: str | None = None
    origen: str | None = None
    # Conjunto de datos con el que se entrenó (slug de `conjuntos`). El de
    # fábrica no lo lleva apuntado: sale del conjunto base, que es de donde
    # `build` lee por defecto.
    datos: str = config.VARIANTE_BASE
    # Cuenta que lo entrenó. `None` es COMPARTIDO: lo es el base, y lo son los
    # modelos que quedaron en disco antes de que hubiera cuentas. Un compartido
    # lo ve todo el mundo, no lo borra nadie y no le cuenta cuota a nadie.
    usuario: str | None = None

    @property
    def es_base(self) -> bool:
        return self.slug == config.VARIANTE_BASE

    @property
    def compartido(self) -> bool:
        return self.usuario is None

    def visible_para(self, usuario: str | None) -> bool:
        return self.compartido or self.usuario == usuario

    def borrable_por(self, usuario: str | None) -> bool:
        """Solo su dueño, y nunca el base ni lo compartido.

        Que lo compartido no se pueda borrar es la otra mitad de que se vea desde
        todas las cuentas: si cualquiera pudiera borrarlo, «compartido» sería
        «de todos para destruirlo».
        """
        return not self.es_base and not self.compartido and self.usuario == usuario

    @property
    def completo(self) -> bool:
        """True si sirve a las dos entidades (lo normal tras un reentrenamiento)."""
        return set(self.entidades) == set(config.ENTIDADES)

    @property
    def fecha(self) -> datetime | None:
        """`creado` como fecha, o None si no lo lleva (el base no lo lleva)."""
        if not self.creado:
            return None
        try:
            return datetime.fromisoformat(self.creado)
        except ValueError:
            return None


@dataclass(frozen=True, order=True)
class ClaveModelo:
    """Identifica un artefacto servible: qué entidad, de qué modelo.

    La formulación y la normalización NO son parte de la elección: se deducen de
    la entidad (`config.MODELO_BASE`). Se exponen como propiedades porque la
    interfaz sí las rotula —el usuario tiene derecho a saber con qué se le está
    respondiendo— y porque son las que forman el nombre del artefacto.
    """

    entidad: str
    variante: str = config.VARIANTE_BASE

    @property
    def formulacion(self) -> str:
        return config.MODELO_BASE.get(self.entidad, ("", ""))[0]

    @property
    def normalizacion(self) -> str:
        return config.MODELO_BASE.get(self.entidad, ("", ""))[1]

    @property
    def es_base(self) -> bool:
        return self.variante == config.VARIANTE_BASE

    @property
    def distancia(self) -> str:
        """Geometría con la que se sirve. No es una elección por entidad."""
        return config_similitud.DISTANCIA_SERVIBLE

    @property
    def stem(self) -> str:
        """Nombre (sin extensión) del artefacto dentro de la carpeta del modelo.

        La regla la fija `src.similitud.modelo.stem_artefacto` y no se reescribe
        aquí: es la que decide que la euclídea vaya SIN sufijo y las demás con él
        (`..._mahalanobis`). Componerlo a mano funcionaba mientras se servía la
        euclídea y dejó de funcionar en cuanto dejó de servirse.
        """
        return stem_artefacto(
            self.formulacion, self.entidad, self.normalizacion, self.distancia)

    @property
    def etiqueta(self) -> str:
        return (
            f"{config.ETIQUETA_ENTIDAD.get(self.entidad, self.entidad)} · "
            f"F{self.formulacion} · "
            f"{config.ETIQUETA_NORMALIZACION.get(self.normalizacion, self.normalizacion)}"
        )


@dataclass(frozen=True)
class Cargado:
    """Un artefacto ya en memoria: el modelo y su perfil por fases."""

    modelo: ModeloSimilitud
    fases: MatrizFases


class Catalogo:
    """Modelos disponibles en un directorio, con carga perezosa y cacheada."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        # clave -> (huella del npz, cargado). El lock protege la caché: el
        # servidor de desarrollo de Flask es multihilo y cargar el mismo modelo
        # dos veces a la vez desperdicia memoria y tiempo.
        self._cache: dict[ClaveModelo, tuple[tuple[float, int], Cargado]] = {}
        self._lock = threading.Lock()

    # --- Layout --------------------------------------------------------------

    @property
    def dir_variantes(self) -> Path:
        """Carpeta que agrupa los modelos con nombre."""
        return self.model_dir / config.SUBDIR_VARIANTES

    def dir_modelo(self, variante: str) -> Path:
        """Carpeta de un modelo: la raíz si es el base, su subcarpeta si no."""
        if variante == config.VARIANTE_BASE:
            return self.model_dir
        return self.dir_variantes / variante

    # --- Descubrimiento ------------------------------------------------------

    def _entidades_en(self, carpeta: Path) -> tuple[str, ...]:
        """Entidades con artefacto completo en esa carpeta.

        Un `.npz` sin su `.json` no cuenta: el JSON trae los nombres de las
        entidades, sin los cuales el modelo no se puede consultar por nombre.
        """
        presentes = []
        for entidad in config.ENTIDADES:
            npz = carpeta / f"{ClaveModelo(entidad).stem}.npz"
            if npz.exists() and npz.with_suffix(".json").exists():
                presentes.append(entidad)
        return tuple(presentes)

    def _leer_metadatos(self, carpeta: Path) -> dict:
        ruta = carpeta / config.FICHERO_VARIANTE
        try:
            # `utf-8-sig` y no `utf-8`: el fichero lo escribe `crear_variante` sin
            # BOM, pero es texto que un usuario puede reescribir a mano (para
            # renombrar el modelo), y en Windows casi cualquier editor le mete uno.
            datos = json.loads(ruta.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        return datos if isinstance(datos, dict) else {}

    def todas_las_variantes(self) -> list[Variante]:
        """TODO lo que hay en disco, sin filtrar por dueño.

        De uso interno (contar cuota, comprobar que un nombre está libre, saber
        si un slug existe para NO confirmárselo a quien no es su dueño). Ninguna
        vista debe llamar a esto: para eso está `variantes(usuario)`.
        """
        salida = [Variante(
            slug=config.VARIANTE_BASE,
            nombre=config.NOMBRE_BASE,
            entidades=self._entidades_en(self.model_dir),
        )]
        if self.dir_variantes.is_dir():
            nombradas = []
            for carpeta in self.dir_variantes.iterdir():
                if not carpeta.is_dir():
                    continue
                meta = self._leer_metadatos(carpeta)
                dueno = meta.get("usuario")
                nombradas.append(Variante(
                    slug=carpeta.name,
                    nombre=str(meta.get("nombre") or carpeta.name),
                    entidades=self._entidades_en(carpeta),
                    creado=meta.get("creado"),
                    origen=meta.get("origen"),
                    datos=str(meta.get("datos") or config.VARIANTE_BASE),
                    # Sin campo `usuario` (modelos anteriores a las cuentas) el
                    # modelo es compartido de solo lectura: adjudicárselo a
                    # alguien seria inventarse un dueño.
                    usuario=str(dueno) if dueno else None,
                ))
            # Por fecha y, sin ella (metadatos ilegibles), por slug: el orden
            # tiene que ser estable entre peticiones o los desplegables bailan.
            nombradas.sort(key=lambda v: (v.creado or "", v.slug))
            salida += nombradas
        return salida

    def variantes(self, usuario: str | None) -> list[Variante]:
        """Modelos que esa cuenta puede ver: los suyos y los compartidos.

        `usuario` es obligatorio y no tiene valor por defecto **a propósito**: un
        default convierte «se me olvidó filtrar» en «se ve todo», que es
        exactamente el fallo que esto viene a cerrar. Sin argumento, `TypeError`.
        """
        return [v for v in self.todas_las_variantes() if v.visible_para(usuario)]

    def variante(self, slug_pedido: str, usuario: str | None) -> Variante:
        """El modelo de ese slug, si esa cuenta puede verlo.

        Un modelo de otro usuario da el MISMO error que uno inexistente: decir
        «existe pero no es tuyo» ya confirma que existe y quién más usa la app.
        """
        for v in self.variantes(usuario):
            if v.slug == slug_pedido:
                return v
        raise ModeloNoDisponible(f"No existe ningún modelo llamado {slug_pedido!r}.")

    def disponibles(self, usuario: str | None) -> list[ClaveModelo]:
        """Artefactos presentes en disco (entidad x modelo), en orden estable."""
        return [
            ClaveModelo(entidad=entidad, variante=v.slug)
            for v in self.variantes(usuario)
            for entidad in config.ENTIDADES
            if entidad in v.entidades
        ]

    def entidades_disponibles(self, usuario: str | None) -> list[str]:
        """Tipos de entidad con al menos un modelo, en el orden de `config`."""
        con_modelo = {c.entidad for c in self.disponibles(usuario)}
        return [e for e in config.ENTIDADES if e in con_modelo]

    def opciones(self, entidad: str, usuario: str | None) -> list[ClaveModelo]:
        """Modelos servibles para un tipo de entidad, el base primero."""
        return [c for c in self.disponibles(usuario) if c.entidad == entidad]

    def opciones_de_variante(self, variante: str,
                             usuario: str | None) -> list[ClaveModelo]:
        """Artefactos servibles de un modelo concreto (0, 1 o 2)."""
        return [c for c in self.disponibles(usuario) if c.variante == variante]

    def variantes_de(self, entidad: str, usuario: str | None) -> list[Variante]:
        """Modelos que pueden responder por esa entidad (para el desplegable)."""
        return [v for v in self.variantes(usuario) if entidad in v.entidades]

    # --- Cuota ---------------------------------------------------------------

    def propios(self, usuario: str | None) -> list[Variante]:
        """Los que ha entrenado esa cuenta (los que le cuentan cuota)."""
        return [v for v in self.todas_las_variantes() if v.usuario == usuario]

    def cuota(self, usuario: str | None) -> tuple[int, int]:
        """(cuántos tiene, cuántos puede tener) esa cuenta."""
        return len(self.propios(usuario)), config.MAX_MODELOS_POR_USUARIO

    def hay_hueco(self, usuario: str | None) -> bool:
        usados, tope = self.cuota(usuario)
        return usados < tope

    # --- Resolución ----------------------------------------------------------

    def resolver(self, entidad: str, usuario: str | None,
                 variante: str | None = None) -> ClaveModelo:
        """Clave concreta a partir de una petición posiblemente incompleta.

        Lo único obligatorio es la entidad (es lo que pide el usuario: jugador o
        equipo). Sin modelo explícito se sirve el base, y si el base no cubre esa
        entidad, el primero que la cubra: la app tiene que seguir sirviendo
        aunque solo se haya construido una parte de los artefactos.
        """
        candidatos = self.opciones(entidad, usuario)
        if not candidatos:
            raise ModeloNoDisponible(
                f"No hay ningún modelo de '{entidad}' disponible. "
                f"Constrúyelos con: python -m src.similitud.build"
            )
        if variante is not None:
            for clave in candidatos:
                if clave.variante == variante:
                    return clave
            raise ModeloNoDisponible(
                f"El modelo {variante!r} no tiene artefacto de {entidad}."
            )
        return candidatos[0]

    # --- Carga ---------------------------------------------------------------

    def cargado(self, clave: ClaveModelo) -> Cargado:
        """Modelo de `clave` y su perfil por fases, desde la caché si sigue vigente.

        La vigencia se mide con (mtime, tamaño) del `.npz`: es barato y detecta
        cualquier reconstrucción normal. No pretende ser un hash — dos
        reconstrucciones del mismo tamaño dentro de la misma marca de tiempo del
        sistema de ficheros pasarían por idénticas, algo irrelevante fuera de un
        test.
        """
        npz = self.ruta_artefacto(clave)
        if not npz.exists():
            raise ModeloNoDisponible(f"Falta el artefacto {npz}.")
        estado = npz.stat()
        huella = (estado.st_mtime, estado.st_size)
        with self._lock:
            guardado = self._cache.get(clave)
            if guardado is not None and guardado[0] == huella:
                return guardado[1]
        # Fuera del lock: cargar es lento (I/O + descompresión) y no queremos
        # bloquear al resto de peticiones mientras tanto.
        modelo = cargar_modelo(
            self.dir_modelo(clave.variante),
            clave.formulacion,
            clave.entidad,
            normalizacion=clave.normalizacion,
            distancia=clave.distancia,
        )
        entrada = Cargado(
            modelo=modelo,
            fases=construir_matriz(
                modelo.entidad, modelo.feat_names, modelo.feat_display
            ),
        )
        with self._lock:
            self._cache[clave] = (huella, entrada)
        return entrada

    def obtener(self, clave: ClaveModelo) -> ModeloSimilitud:
        """Solo el modelo de `clave` (atajo de `cargado`)."""
        return self.cargado(clave).modelo

    def ruta_artefacto(self, clave: ClaveModelo) -> Path:
        """Ruta del `.npz` de `clave`.

        Pública porque hay dos cosas que se preguntan del artefacto SIN cargarlo
        (su fecha en la página de datos, sus `entity_ids` para medir la
        cobertura), y componer el nombre a mano en cada sitio duplicaría la regla
        de `ClaveModelo.stem`.
        """
        return self.dir_modelo(clave.variante) / f"{clave.stem}.npz"

    # --- Alta de un modelo nuevo ---------------------------------------------

    def crear_variante(self, nombre: str, usuario: str | None,
                       origen: str = config.VARIANTE_BASE,
                       datos: str = config.VARIANTE_BASE) -> Variante:
        """Reserva la carpeta de un modelo nuevo y escribe su `variante.json`.

        Se llama ANTES de lanzar el reentrenamiento, por dos motivos: el comando
        necesita un `--out` que exista y el nombre solo lo conoce la app (el CLI
        trabaja con rutas). Si el reentrenamiento falla, queda una carpeta sin
        artefactos, que la página marca como incompleta y el buscador ignora.

        `datos` es el conjunto de datos sobre el que se entrena. Se apunta aquí
        porque después hace falta para SERVIRLO: el equipo de cada jugador, sus
        minutos por posición y los valores reales de las métricas salen de esa
        misma BD, y leerlos de otra dejaría sin contexto justo a las entidades
        que aporta el conjunto nuevo.

        `usuario` queda apuntado como dueño. La cuota se comprueba ANTES de crear
        la carpeta, para no dejar un directorio a medias por un límite que ya se
        sabía. El nombre se contrasta contra TODOS los slugs del disco, no solo
        contra los visibles: dos carpetas no pueden llamarse igual aunque sean de
        cuentas distintas, y dejar que colisionen sería peor que decir que el
        nombre está cogido.
        """
        usados, tope = self.cuota(usuario)
        if usados >= tope:
            raise CuotaSuperada(
                f"Has llegado al máximo de {tope} modelos. Borra alguno de los "
                f"tuyos antes de entrenar otro."
            )
        limpio, destino = preparar(
            nombre, (v.slug for v in self.todas_las_variantes()), "modelo")
        carpeta = self.dir_variantes / destino
        carpeta.mkdir(parents=True)
        variante = Variante(
            slug=destino,
            nombre=limpio,
            creado=datetime.now().isoformat(timespec="seconds"),
            origen=origen,
            datos=datos,
            usuario=usuario,
        )
        (carpeta / config.FICHERO_VARIANTE).write_text(
            json.dumps(
                {"nombre": variante.nombre, "slug": variante.slug,
                 "creado": variante.creado, "origen": variante.origen,
                 "datos": variante.datos, "usuario": variante.usuario},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        return variante

    def descartar_variante(self, slug_pedido: str) -> None:
        """Borra la carpeta de un modelo recién creado (alta que no se lanzó)."""
        if slug_pedido == config.VARIANTE_BASE:
            raise ValueError("el modelo base no se descarta")
        shutil.rmtree(self.dir_variantes / slug_pedido, ignore_errors=True)

    def borrar_variante(self, slug_pedido: str, usuario: str | None) -> Variante:
        """Borra un modelo del usuario, con sus artefactos. Devuelve el borrado.

        Sin borrado no hay cuota que valga: un tope de 3 sin forma de liberar
        sitio es un tope de 3 modelos por cuenta y para siempre.

        Un modelo que no sea suyo (o el base, o uno compartido) da
        `ModeloNoDisponible`, el mismo error que uno inexistente: quien no puede
        verlo tampoco tiene por qué enterarse de que está ahí.
        """
        objetivo = next(
            (v for v in self.todas_las_variantes() if v.slug == slug_pedido), None)
        if objetivo is None or not objetivo.borrable_por(usuario):
            raise ModeloNoDisponible(
                f"No existe ningún modelo llamado {slug_pedido!r}.")
        shutil.rmtree(self.dir_variantes / objetivo.slug)
        with self._lock:
            # La caché va por (entidad, modelo): si no se limpia, el modelo
            # borrado se seguiría sirviendo desde memoria hasta reiniciar.
            for clave in [c for c in self._cache if c.variante == objetivo.slug]:
                del self._cache[clave]
        return objetivo

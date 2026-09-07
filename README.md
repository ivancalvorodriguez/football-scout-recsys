# Recomendador de scouting por similitud

Sistema de recomendación que, dado un jugador o un equipo, devuelve los **más
parecidos por cómo juegan**, no por sus resultados. Todo el recorrido está en el
repositorio: desde los eventos crudos de StatsBomb hasta una app web que sirve el
top-k explicado, pasando por la construcción de los modelos, su evaluación sin
*ground truth* y la incorporación incremental de partidos nuevos.

El motor de similitud es **SLIM** (*Sparse Linear Methods*), en dos formulaciones
que se comparan entre sí:

- **Formulación 2** — SLIM instancia-instancia: aprende una matriz de pesos `W`
  reconstruyendo cada observación *(entidad, partido)* como combinación dispersa
  y no negativa de las demás y agrega bloques de `W` para obtener la similitud
  entidad-entidad. La agregación cae sobre los pesos aprendidos, nunca sobre las
  features de entrada.
- **Formulación 5** — distribucional + EASE: cada entidad es la **nube** de sus
  partidos. Se mide la similitud entre nubes (MMD con *Random Fourier Features*,
  o transporte óptimo entrópico / Sinkhorn) y después EASE (SLIM de forma
  cerrada) re-rankea esa matriz. Es la que se sirve en producción.

---

## Qué incluye

| Pieza | Paquete | Qué hace |
|---|---|---|
| Extracción | `src.extraccion` | Convierte los JSON de eventos de StatsBomb en métricas por *(jugador, partido)* y *(equipo, partido)* sobre SQLite. |
| Similitud | `src.similitud` | Capa de features (per-90, ratios, diferencias, z-score) y ajuste de las dos formulaciones SLIM. |
| Evaluación | `src.evaluacion` | Protocolo de evaluación **sin ground truth** por fases, barrido de hiperparámetros y superficies 3D del espacio de búsqueda. |
| Incremental | `src.incremental` | Añade partidos nuevos a la BD y reentrena reaprovechando el ajuste anterior (*warm start*). |
| App web | `src.app` | Flask + waitress: búsqueda, top-k explicado, fichas, radar por fases, glosario, API JSON y una sección para ingerir datos y entrenar desde el navegador. |

## Los datos

**StatsBomb Open Data**, cinco grandes ligas europeas de la temporada 2015/2016:

- Premier League (Inglaterra), La Liga (España), Serie A (Italia), Ligue 1
  (Francia) y 1. Bundesliga (Alemania).
- **1.823 partidos**, **2.640 jugadores**, **98 equipos**,
  50.579 filas *(jugador, partido)* y 3.646 *(equipo, partido)*.

Todas las coordenadas siguen el sistema StatsBomb (campo de 120×80; 1 unidad ≈
0,9144 m). Las métricas replican definiciones públicas del sector: xT sobre
rejilla, xA, SCA/GCA, *deep completions* (Wyscout), *high turnovers* (Opta), PPDA,
*field tilt*, acciones defensivas ajustadas por posesión, y presiones y
contrapresiones con ventana de recuperación de 5 s.

## Requisitos e instalación

Python **3.11**. Dependencias mínimas: numpy, pandas y matplotlib (más flask,
waitress y redis para la app web).

```bash
git clone <este-repo> && cd football-scout-recsys
python -m venv .venv && .venv\Scripts\activate    # Windows
# source .venv/bin/activate                        # Linux/macOS

pip install -r requirements-dev.txt    # entorno de desarrollo completo
# pip install -r requirements.txt      # solo el pipeline
# pip install -r requirements-app.txt  # solo lo que necesita la app web
```

El dataset **no** viaja en el repositorio (son ~20 GB). Para reconstruir la base
de datos desde cero hay que clonarlo dentro del proyecto:

```bash
git clone https://github.com/statsbomb/open-data.git open-data
```

Para **usar** la app no hace falta: la base de datos ya extraída
(`outputs/db/scouting.db`) y los dos modelos servidos (`outputs/modelo/`) sí están
versionados.

## Arranque rápido

```bash
python -m src.app                 # http://127.0.0.1:5000
```

Consultar un modelo desde la línea de comandos:

```bash
python -m src.similitud.probar --formulacion 5 --entidad jugador \
    --normalizacion por_liga --distancia manhattan --nombre "Messi" --k 10

python -m src.similitud.probar --formulacion 5 --entidad equipo \
    --normalizacion global --distancia manhattan --nombre "Sevilla"
```

Los cuatro selectores son obligatorios en la práctica porque los defaults del CLI
(`--formulacion 2`, `--distancia euclidea`) apuntan a artefactos que este
repositorio no incluye: los únicos presentes son los dos servidos, y son
formulación 5 con distancia manhattan (ver la tabla más abajo). Además de los
similares, `probar` imprime **en qué métricas coinciden**, con el z-score de la
referencia frente al del candidato.

## El pipeline completo

Cada etapa es un punto de entrada independiente, todas aceptan `--help`. Los pasos
3 y 4 recorren la rejilla completa de variantes, así que dan por hecho un `build`
previo que las haya construido: el repositorio solo trae los dos artefactos
servidos.

**1. Extracción** — de los JSON a SQLite. Reprocesar un partido lo actualiza
(UPSERT), no lo duplica.

```bash
python -m src.extraccion.extract                                  # interactivo
python -m src.extraccion.extract --competition "Premier League" \
    --season "2015/2016" --pct 100 --db outputs/db/scouting.db
```

**2. Construcción de modelos** — features + ajuste. No toca la BD (solo lectura).

```bash
python -m src.similitud.build --formulacion ambas --entidad ambas \
    --normalizacion ambas --distancia manhattan
```

Cada artefacto se guarda como `<nombre>.npz` + `.json` (metadatos) +
`.huella.json` (identidad reproducible del ajuste) + `.warm.npz` (el estado
interno que permite reentrenar después sin partir de cero).

**3. Comparación entre variantes** — cruza los top-k de dos normalizaciones y
genera CSV y figuras.

```bash
python -m src.similitud.comparar --entidad jugador --formulacion 5
```

**4. Evaluación** — sobre los artefactos ya construidos.

```bash
python -m src.evaluacion.evaluar --fases 0,1,2,3,5 --bootstrap 200
```

| Fase | Qué mide |
|---|---|
| 0 | *Sanity checks* del artefacto y de la matriz de similitud. |
| 1 | **Auto-similitud**: se parte cada entidad en mitades (jornadas pares/impares) y se comprueba si el modelo reconoce a una en la otra. Es el pilar. |
| 2 | **Estabilidad**: remuestreos bootstrap + RBO sobre los rankings. |
| 3 | **Triangulación**: test de Mantel contra medidas independientes. |
| 4 | Aporte del *denoising* de EASE (se calcula dentro de la fase 1). |
| 5 | *Downstream* y métricas *beyond-accuracy* (cobertura, diversidad). |
| 6 | Validación humana con scouts — **no automatizable**; queda documentada como paso pendiente. |
| 7 | **Generalización**: *hold-out* por liga. Se ajusta sin una liga entera, se proyectan sus entidades con *fold-in* y se mide dentro y fuera de muestra. |

**5. Barrido de hiperparámetros** — construye *y* evalúa una celda por
combinación, en paralelo, con deduplicación por huella y reutilización de las
métricas ya calculadas. La carpeta de salida es acumulativa: ampliar la rejilla es
relanzar el barrido con más valores.

```bash
python -m src.evaluacion.barrido --entidades jugador --distancias manhattan \
    --holdout india --trabajos auto --out outputs/evaluacion/jugador_v3
python -m src.evaluacion.figuras3d --barrido outputs/evaluacion/jugador_v3
```

Las superficies 3D no dicen qué combinación gana (eso lo dice el resumen del
barrido), sino qué **forma** tiene el óptimo: meseta, cresta estrecha o máximo en
el borde de la rejilla, la señal de que el rango se quedó corto.

**6. Incorporación incremental** — dos comandos encadenados. Un *paquete* de
partidos es un directorio con la misma estructura que `open-data/data`, de modo
que los partidos nuevos pasan por exactamente el mismo motor de métricas y son
comparables por construcción.

```bash
python -m src.incremental.ingesta --paquete ruta/al/paquete --solo-validar
python -m src.incremental.ingesta --paquete ruta/al/paquete
python -m src.incremental.reentrenar --servibles          # warm start
python -m src.incremental.reentrenar --frio --verificar   # de cero, y compara
```

## El modelo que se sirve

`build` genera las cuatro combinaciones por entidad porque comparar formulaciones
y normalizaciones es el eje experimental del proyecto, pero la app sirve **una por
entidad**. El
linaje completo (huellas, SHA-256, origen y cautelas declaradas) está en
`outputs/modelo/produccion.json`.

| Entidad | Formulación | Normalización | Distancia | Hiperparámetros |
|---|---|---|---|---|
| Jugador | 5 (MMD + EASE) | z-score por liga | manhattan | `RFF_DIM 1536`, `EASE_LAMBDA 0.0` |
| Equipo | 5 (MMD + EASE) | z-score global | manhattan | `RFF_DIM 1216`, `EASE_LAMBDA 0.03` |

**La posición del jugador es una feature** (25 columnas con la fracción de
minutos disputada en cada una), pero el bloque se escala por `1/sqrt(25)` para que pese como *una* feature: la posición informa sin gobernar el ranking. Como
contrapartida, las métricas que usan la posición como etiqueta (purezaposicional, k-NN por posición) quedan **circulares** y no valen comevidencia.

## La app web

```bash
python -m src.app --debug                    # desarrollo (servidor de Flask)
python -m src.app --produccion --hilos 8     # waitress, modo endurecido
```

En `--produccion` se exige `SCOUTING_SECRET_KEY`, la cookie de sesión se marca
`Secure` (hace falta TLS delante) y la interfaz deja de mostrar rutas absolutas
del servidor. Los dos modos son excluyentes con `--debug`: el depurador de
Werkzeug ejecuta código arbitrario desde el navegador.

**Rutas públicas** — no necesitan cuenta: `/` (buscador), `/similares`
(resultados, o lista de candidatos si el nombre es ambiguo), `/jugador/<id>` y
`/equipo/<id>` (fichas con radar por fases de juego y valores crudos),
`/glosario`, y la API JSON `/api/sugerencias`, `/api/similares` y
`/api/ficha/<entidad>/<id>`.

**Sección `/datos`** — requiere cuenta (el alta es abierta desde `/registro`).
Permite validar e ingerir un paquete de partidos y entrenar un modelo con nombre,
con barra de progreso y cancelación. Cada cuenta ve y usa solo lo suyo, más el
modelo y el conjunto de fábrica, que son compartidos e imborrables. Todos los POST
exigen token CSRF.

Cuando se pregunta por una entidad que está en la base de datos pero no en el
modelo, la app no falla ni reentrena: la **proyecta** (*fold-in*) sobre la
geometría ya aprendida y rotula en la respuesta si esa proyección es `FIEL` (F5
con MMD: es la definición, no una aproximación) o `APROXIMADA`.

Administración de cuentas al margen del formulario:

```bash
python -m src.app.usuarios --listar
python -m src.app.usuarios --alta scout --nombre "Nombre visible"
python -m src.app.usuarios --contrasena scout
python -m src.app.usuarios --baja scout
```

## Despliegue con Docker

```bash
docker compose up -d --build                  # http://localhost:8000
docker compose --profile https up -d --build  # https://localhost (Caddy delante)
```

Cuatro servicios: **app** (sirve páginas y API; no ejecuta nada largo, encola),
**worker** (misma imagen, otro *entrypoint*: ejecuta la ingesta y el
reentrenamiento), **redis** (la cola y el estado de las tareas) y **proxy**
(Caddy, termina TLS; solo en el perfil `https`, que es el que hace falta con
`SCOUTING_MODO=produccion`).

`outputs/` y `open-data/` entran como *bind mount*, no van en la imagen: así lo
entrenado sobrevive a un `docker compose down` y `open-data/` puede montarse en
solo lectura. El worker **no se escala**: las dos tareas escriben en sitios
compartidos y lo que garantiza el «una tarea a la vez» es que ese bucle sea
secuencial y único.

Variables principales: `SCOUTING_MODO` (`local` | `produccion`),
`SCOUTING_SECRET_KEY`, `SCOUTING_COLA` (URL de redis; vacía significa que la app
ejecuta las tareas en su propio proceso), `SCOUTING_MODELO`, `SCOUTING_BD`,
`SCOUTING_HILOS`, `SCOUTING_LOG`, `SCOUTING_PUERTO` y `SCOUTING_DOMINIO`.

Los módulos del pipeline (extracción, similitud, evaluación) **no** están
containerizados a propósito: son comandos de un solo uso cuyos tiempos y consumo
de memoria piden la máquina entera.

## Estructura

```
src/
  extraccion/    JSON de StatsBomb -> métricas -> SQLite
  similitud/     features, formulaciones 2 y 5, fold-in, warm start, CLIs
  evaluacion/    fases 0-7, barrido paralelo, figuras 3D, huellas
  incremental/   ingesta de paquetes y reentrenamiento
  app/           Flask: dominio (servicio) + vistas + cola de tareas
  exploracion/   scripts puntuales, al margen del pipeline
docker/          entrypoints, healthcheck y Caddyfile
outputs/         BD, modelos y salida de evaluación
open-data/       dataset StatsBomb (se clona aparte)
```

La configuración vive centralizada, no repartida por las vistas ni por los
scripts: `src/extraccion/config.py` (dominio y geometría del campo),
`src/similitud/config.py` (features e hiperparámetros), `src/evaluacion/config.py`
y `src/app/config.py`.

## Limitaciones conocidas

- El explorador de carpetas de `/datos` sirve el árbol de directorios del
  servidor. Hoy exige autenticación, pero la exposición está **mitigada, no
  eliminada**: en un servidor compartido habría que acotarlo a una raíz permitida.
- El auto-registro debilita el «hace falta cuenta para entrenar»: lo que acota el
  daño son las cuotas por cuenta y el aislamiento entre usuarios. Un despliegue
  expuesto a internet necesitaría invitación o moderación.
- El score compuesto que ordena el barrido es una **heurística de ordenación**, no
  un criterio validado; las métricas por separado no siempre dan un ganador limpio.
- El estado de las tareas y los catálogos viven en memoria del proceso: la app se
  sirve con un proceso y varios hilos, nunca con varios procesos.
- La fase 6 (validación con scouts) no está hecha, y sin ella no hay *ground
  truth* humano contra el que contrastar los rankings.

## Créditos

Datos: [StatsBomb Open Data](https://github.com/statsbomb/open-data), bajo sus
propios términos y condiciones. Cualquier análisis publicado a partir de ellos
debe citar a **StatsBomb** como fuente.

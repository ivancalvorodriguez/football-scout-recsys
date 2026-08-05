"""Exporta las superficies del barrido a un HTML interactivo (rotable) autocontenido.

El PNG deja la camara fija, y en una superficie 3D el angulo decide lo que se ve:
una cresta estrecha puede parecer una meseta segun desde donde se mire. Este modulo
escribe **un solo fichero HTML** con los datos de todas las superficies embebidos y
un visor que se arrastra con el raton para girarlas.

Se dibuja con un renderer 3D escrito a mano sobre `<canvas>` 2D (proyeccion
ortografica + algoritmo del pintor), por la misma razon por la que el nucleo
numerico del proyecto no usa scipy: **no añade ninguna dependencia**. El fichero se
abre con doble clic, sin servidor y sin conexion, y se puede mover a otra maquina o
adjuntar a la memoria tal cual.

Toda la geometria viene resuelta desde Python (`figuras3d._geometria_json`): el
visor recibe las coordenadas de cada nodo ya normalizadas a [-1, 1] **conservando
el reparto real** de los valores barridos, y la superficie ya estimada sobre la
malla fina (`Zf`) junto a los valores medidos (`Z`). Asi el HTML y los PNG enseñan
exactamente la misma superficie, y el visor no tiene que reimplementar ni la escala
de los ejes ni la interpolacion.

La malla ronda los 30x30 cuadrilateros por superficie: ordenarlos por profundidad
en cada fotograma sigue siendo barato y no hace falta z-buffer.
"""

from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# Plantilla                                                                     #
# --------------------------------------------------------------------------- #
# Sin f-strings: el JS esta lleno de llaves. Los dos huecos se rellenan con
# `str.replace`.

_PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITULO__</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 system-ui, "Segoe UI", sans-serif;
         background: #f6f7f9; color: #1b1d21; }
  @media (prefers-color-scheme: dark) {
    body { background: #14161a; color: #e6e8ec; }
    .panel, .lienzo-caja { background: #1d2027 !important; border-color: #2c313a !important; }
    select, button { background: #262a33; color: #e6e8ec; border-color: #3a404b; }
  }
  header { padding: 14px 18px 6px; }
  h1 { font-size: 17px; margin: 0 0 2px; font-weight: 600; }
  .sub { font-size: 12px; opacity: .7; }
  .cuerpo { display: flex; gap: 14px; padding: 10px 18px 18px; align-items: flex-start;
            flex-wrap: wrap; }
  .lienzo-caja { position: relative; flex: 1 1 560px; min-width: 320px; background: #fff;
                 border: 1px solid #dde0e6; border-radius: 10px; overflow: hidden; }
  canvas { display: block; width: 100%; height: 66vh; min-height: 380px; cursor: grab; }
  canvas.girando { cursor: grabbing; }
  .panel { flex: 0 0 264px; background: #fff; border: 1px solid #dde0e6;
           border-radius: 10px; padding: 12px 14px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
              opacity: .6; margin: 14px 0 6px; }
  .panel h2:first-child { margin-top: 0; }
  select, button { width: 100%; padding: 6px 8px; border-radius: 7px;
                   border: 1px solid #ccd1d9; background: #fff; font: inherit; }
  button { cursor: pointer; margin-top: 8px; }
  label.chk { display: flex; gap: 7px; align-items: center; margin: 5px 0; font-size: 13px; }
  label.chk input { width: auto; }
  .dato { font-variant-numeric: tabular-nums; }
  .dato b { font-size: 19px; }
  .leyenda { margin-top: 8px; font-size: 12.5px; }
  .leyenda div { display: flex; gap: 7px; align-items: center; margin: 3px 0; }
  .tinta { width: 13px; height: 13px; border-radius: 3px; flex: 0 0 auto;
           border: 1px solid rgba(0,0,0,.25); }
  .pie { padding: 0 18px 22px; font-size: 12px; opacity: .72; max-width: 1000px; }
  .aviso { color: #b4451f; font-weight: 600; }
  @media (prefers-color-scheme: dark) { .aviso { color: #ff9f7a; } }
</style>
</head>
<body>
<header>
  <h1>__TITULO__</h1>
  <div class="sub">Arrastra para girar · rueda para acercar · doble clic para reencuadrar</div>
</header>

<div class="cuerpo">
  <div class="lienzo-caja"><canvas id="lienzo"></canvas></div>
  <div class="panel">
    <h2>Metrica</h2>
    <select id="sel-metrica"></select>
    <h2>Modelo</h2>
    <select id="sel-modelo"></select>
    <h2>Ejes</h2>
    <select id="sel-ejes"></select>
    <h2>Optimo</h2>
    <div class="dato" id="optimo"></div>
    <h2>Ver</h2>
    <label class="chk"><input type="checkbox" id="chk-puntos" checked> Puntos evaluados</label>
    <label class="chk"><input type="checkbox" id="chk-malla" checked> Rejilla medida</label>
    <label class="chk"><input type="checkbox" id="chk-suelo" checked> Proyeccion en el suelo</label>
    <button id="btn-reset">Vista inicial</button>
    <div class="leyenda" id="leyenda"></div>
  </div>
</div>
<div class="pie" id="pie"></div>

<script>
const DATOS = __DATOS__;

// --- Paleta --------------------------------------------------------------- //
const VIRIDIS = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
                 [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
const SERIES = ["#3b7dd8","#e0803a","#5aa469","#a45ac0","#c0455a","#3ba7a0"];

function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VIRIDIS.length - 1), i = Math.min(Math.floor(x), VIRIDIS.length - 2);
  const f = x - i, a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [0,1,2].map(k => Math.round(a[k] + f * (b[k] - a[k])));
}
const rgb = (c, alfa) => `rgba(${c[0]},${c[1]},${c[2]},${alfa})`;
function aRGB(hex) {
  return [1,3,5].map(i => parseInt(hex.slice(i, i + 2), 16));
}

// --- Estado --------------------------------------------------------------- //
const AZ0 = -0.62, EL0 = 0.46;
let az = AZ0, el = EL0, zoom = 1, vista = DATOS.vistas[0];

const lienzo = document.getElementById("lienzo");
const ctx = lienzo.getContext("2d");

// --- Proyeccion ----------------------------------------------------------- //
// Ortografica: giro sobre el eje vertical (az) + inclinacion de camara (el). La
// profundidad `d` sale del mismo giro y es lo unico que ordena el dibujado: no hay
// z-buffer, se pinta de lejos a cerca (algoritmo del pintor).
function proyectar(p, cx, cy, esc) {
  const ca = Math.cos(az), sa = Math.sin(az);
  const x1 = p[0] * ca - p[1] * sa;
  const y1 = p[0] * sa + p[1] * ca;
  const ce = Math.cos(el), se = Math.sin(el);
  const y2 = y1 * ce - p[2] * se;      // profundidad (mayor = mas lejos)
  const z2 = y1 * se + p[2] * ce;      // altura en pantalla
  return [cx + esc * x1, cy - esc * z2, y2];
}

// --- Normalizacion de una vista ------------------------------------------- //
// Las coordenadas de los ejes llegan ya calculadas desde Python, en [-1, 1] y con
// el reparto REAL de los valores barridos (escala lineal, logaritmica u ordinal
// segun el eje; `escalaX`/`escalaY` dicen cual). La z se normaliza aqui, al rango
// de los valores MEDIDOS de la vista: la superficie estimada no se sale de el.
const SUELO = -0.8, TECHO = 0.8;

function rango(v) {
  let lo = Infinity, hi = -Infinity;
  for (const s of v.superficies) for (const fila of s.Z) for (const z of fila) {
    if (z === null) continue;
    if (z < lo) lo = z;
    if (z > hi) hi = z;
  }
  return [lo, hi];
}

function altura(z, lo, hi) {
  return hi > lo ? SUELO + ((z - lo) / (hi - lo)) * (TECHO - SUELO) : 0;
}

// Nodo de la malla fina (superficie estimada) y nodo medido (punto evaluado).
function nodoF(v, i, j, z, lo, hi) { return [v.coordXf[j], v.coordYf[i], altura(z, lo, hi)]; }
function nodoM(v, i, j, z, lo, hi) { return [v.coordX[j], v.coordY[i], altura(z, lo, hi)]; }

// --- Dibujo --------------------------------------------------------------- //
function dibujar() {
  const dpr = window.devicePixelRatio || 1;
  const w = lienzo.clientWidth, h = lienzo.clientHeight;
  lienzo.width = w * dpr; lienzo.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2, cy = h / 2 + h * 0.04;
  const esc = Math.min(w, h) * 0.30 * zoom;
  const P = p => proyectar(p, cx, cy, esc);
  const v = vista, [lo, hi] = rango(v);
  const multi = v.superficies.length > 1;
  const oscuro = window.matchMedia &&
                 window.matchMedia("(prefers-color-scheme: dark)").matches;
  const tinta = oscuro ? "#e6e8ec" : "#1b1d21";

  // Caja: solo las aristas del suelo. Van debajo de todo; un cubo completo
  // estorba mas que ayuda con una rejilla de tan pocos valores barridos.
  ctx.strokeStyle = "rgba(128,138,152,.55)"; ctx.lineWidth = 1;
  const base = SUELO - 0.12;
  const esquinas = [[-1,-1,base],[1,-1,base],[1,1,base],[-1,1,base]];
  ctx.beginPath();
  esquinas.forEach((e, i) => {
    const [x, y] = P(e);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.closePath(); ctx.stroke();

  // Cuadrilateros de la malla fina de todas las superficies, mezclados y ordenados
  // por profundidad. `aristas` marca los lados que caen sobre un valor REALMENTE
  // barrido: son la rejilla medida, y se dibujan con cada cara para que las tape
  // correctamente lo que este delante.
  const nxf = v.coordXf.length, nyf = v.coordYf.length;
  const caras = [];
  v.superficies.forEach((s, idx) => {
    const base_col = multi ? aRGB(SERIES[idx % SERIES.length]) : null;
    for (let i = 0; i < nyf - 1; i++) {
      for (let j = 0; j < nxf - 1; j++) {
        const zs = [s.Zf[i][j], s.Zf[i][j+1], s.Zf[i+1][j+1], s.Zf[i+1][j]];
        if (zs.some(z => z === null)) continue;   // sin datos alrededor: no se estima
        const idxs = [[i,j],[i,j+1],[i+1,j+1],[i+1,j]];
        const pts = idxs.map(([a, b], k) => P(nodoF(v, a, b, zs[k], lo, hi)));
        const media = zs.reduce((a, b) => a + b, 0) / 4;
        const col = base_col || viridis(hi > lo ? (media - lo) / (hi - lo) : 0.5);
        const aristas = [];
        if (v.medidaYf[i]) aristas.push([0, 1]);
        if (v.medidaXf[j]) aristas.push([0, 3]);
        if (i === nyf - 2 && v.medidaYf[i+1]) aristas.push([3, 2]);
        if (j === nxf - 2 && v.medidaXf[j+1]) aristas.push([1, 2]);
        caras.push({ pts, col, aristas, d: pts.reduce((a, p) => a + p[2], 0) / 4 });
      }
    }
  });
  caras.sort((a, b) => b.d - a.d);           // de lejos a cerca
  const malla = document.getElementById("chk-malla").checked;
  for (const c of caras) {
    ctx.beginPath();
    c.pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = rgb(c.col, multi ? 0.62 : 0.94); ctx.fill();
    // Relleno y borde del mismo color: sin el, el antialias del canvas deja
    // costuras claras entre cuadrilateros contiguos de la malla fina.
    ctx.strokeStyle = rgb(c.col, multi ? 0.62 : 0.94); ctx.lineWidth = 1; ctx.stroke();
    if (malla && c.aristas.length) {
      ctx.strokeStyle = oscuro ? "rgba(235,238,244,.40)" : "rgba(20,22,26,.45)";
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      for (const [a, b] of c.aristas) {
        ctx.moveTo(c.pts[a][0], c.pts[a][1]);
        ctx.lineTo(c.pts[b][0], c.pts[b][1]);
      }
      ctx.stroke();
    }
  }

  // Proyeccion de nivel en el suelo: recupera las diferencias que el escorzo se come.
  // Misma casilla, aplastada contra la base (la z se sustituye por `base`).
  if (document.getElementById("chk-suelo").checked && !multi) {
    const s = v.superficies[0];
    for (let i = 0; i < nyf - 1; i++) {
      for (let j = 0; j < nxf - 1; j++) {
        const zs = [s.Zf[i][j], s.Zf[i][j+1], s.Zf[i+1][j+1], s.Zf[i+1][j]];
        if (zs.some(z => z === null)) continue;
        const idxs = [[i,j],[i,j+1],[i+1,j+1],[i+1,j]];
        const pts = idxs.map(([a, b]) => P([v.coordXf[b], v.coordYf[a], base]));
        const media = zs.reduce((a, b) => a + b, 0) / 4;
        ctx.beginPath();
        pts.forEach((p, k) => k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
        ctx.closePath();
        const col = rgb(viridis(hi > lo ? (media - lo) / (hi - lo) : 0.5), 0.38);
        ctx.fillStyle = col; ctx.fill();
        ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.stroke();
      }
    }
  }

  // Puntos realmente evaluados (lo demas es la estimacion entre ellos).
  if (document.getElementById("chk-puntos").checked) {
    ctx.fillStyle = "rgba(25,28,33,.85)";
    for (const s of v.superficies) {
      for (let i = 0; i < v.coordY.length; i++) {
        for (let j = 0; j < v.coordX.length; j++) {
          const z = s.Z[i][j];
          if (z === null) continue;
          const [x, y] = P(nodoM(v, i, j, z, lo, hi));
          ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 6.2832); ctx.fill();
        }
      }
    }
  }

  // Optimo (siempre un punto MEDIDO: la estimacion no crea maximos nuevos).
  const op = optimo(v);
  if (op) {
    const [x, y] = P(nodoM(v, op.i, op.j, op.z, lo, hi));
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 6.2832);
    ctx.fillStyle = "#d92b4b"; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.6; ctx.stroke();
  }

  // Etiquetas de los ejes, con el valor real de cada casilla.
  ctx.fillStyle = tinta; ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  v.valoresX.forEach((etq, j) => {
    const [x, y] = P([v.coordX[j], -1.22, base]);
    ctx.fillText(etq, x, y);
  });
  v.valoresY.forEach((etq, i) => {
    const [x, y] = P([1.24, v.coordY[i], base]);
    ctx.fillText(etq, x, y);
  });
  ctx.font = "600 12px system-ui, sans-serif";
  const nombreEje = (n, esc) => esc === "lineal" ? n : `${n} (${esc})`;
  const [ex, ey] = P([0, -1.62, base]);
  ctx.fillText(nombreEje(v.ejeX, v.escalaX), ex, ey);
  const [fx, fy] = P([1.66, 0, base]);
  ctx.fillText(nombreEje(v.ejeY, v.escalaY), fx, fy);

  // Escala de la z: el rango de la vista, arriba a la izquierda.
  ctx.textAlign = "left"; ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = "rgba(128,138,152,1)";
  ctx.fillText(`${v.metrica}: ${fmt(lo)} … ${fmt(hi)}`, 12, 16);
}

function fmt(x) { return Math.abs(x) < 0.001 && x !== 0 ? x.toExponential(2) : x.toFixed(3); }

function optimo(v) {
  if (v.superficies.length !== 1) return null;
  const Z = v.superficies[0].Z;
  let mejor = null;
  for (let i = 0; i < Z.length; i++) for (let j = 0; j < Z[i].length; j++) {
    const z = Z[i][j];
    if (z === null) continue;
    if (!mejor || (v.mejor === "max" ? z > mejor.z : z < mejor.z)) mejor = { i, j, z };
  }
  return mejor;
}

// --- Interaccion ---------------------------------------------------------- //
let arrastrando = false, px = 0, py = 0;
lienzo.addEventListener("pointerdown", e => {
  arrastrando = true; px = e.clientX; py = e.clientY;
  lienzo.classList.add("girando"); lienzo.setPointerCapture(e.pointerId);
});
lienzo.addEventListener("pointermove", e => {
  if (!arrastrando) return;
  az += (e.clientX - px) * 0.011;
  el = Math.max(-0.15, Math.min(1.45, el + (e.clientY - py) * 0.008));
  px = e.clientX; py = e.clientY;
  dibujar();
});
["pointerup", "pointercancel"].forEach(ev => lienzo.addEventListener(ev, e => {
  arrastrando = false; lienzo.classList.remove("girando");
}));
lienzo.addEventListener("wheel", e => {
  e.preventDefault();
  zoom = Math.max(0.45, Math.min(3.2, zoom * (e.deltaY < 0 ? 1.09 : 1 / 1.09)));
  dibujar();
}, { passive: false });
lienzo.addEventListener("dblclick", () => { az = AZ0; el = EL0; zoom = 1; dibujar(); });
document.getElementById("btn-reset").onclick =
  () => { az = AZ0; el = EL0; zoom = 1; dibujar(); };
["chk-puntos", "chk-malla", "chk-suelo"].forEach(
  id => document.getElementById(id).onchange = dibujar);
window.addEventListener("resize", dibujar);

// --- Selectores ----------------------------------------------------------- //
const selM = document.getElementById("sel-metrica");
const selMod = document.getElementById("sel-modelo");
const selEjes = document.getElementById("sel-ejes");

function opciones(sel, valores, actual) {
  sel.innerHTML = "";
  for (const v of valores) {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    if (v === actual) o.selected = true;
    sel.appendChild(o);
  }
  sel.disabled = valores.length < 2;
}
const unicos = a => [...new Set(a)];

function refrescarSelectores() {
  const metricas = unicos(DATOS.vistas.map(v => v.metrica));
  opciones(selM, metricas, vista.metrica);
  const modelos = unicos(DATOS.vistas.filter(v => v.metrica === vista.metrica)
                                     .map(v => v.modelo));
  opciones(selMod, modelos, vista.modelo);
  const ejes = unicos(DATOS.vistas.filter(v => v.metrica === vista.metrica &&
                                               v.modelo === vista.modelo)
                                  .map(v => v.ejes));
  opciones(selEjes, ejes, vista.ejes);
}

function elegir(metrica, modelo, ejes) {
  const cand = DATOS.vistas.filter(v => v.metrica === metrica);
  vista = cand.find(v => v.modelo === modelo && v.ejes === ejes)
       || cand.find(v => v.modelo === modelo)
       || cand[0];
  refrescarSelectores(); pintarPanel(); dibujar();
}
selM.onchange = () => elegir(selM.value, vista.modelo, vista.ejes);
selMod.onchange = () => elegir(vista.metrica, selMod.value, vista.ejes);
selEjes.onchange = () => elegir(vista.metrica, vista.modelo, selEjes.value);

function pintarPanel() {
  const v = vista, op = optimo(v);
  const caja = document.getElementById("optimo");
  if (op) {
    const borde = (v.valoresY.length > 2 && (op.i === 0 || op.i === v.valoresY.length - 1))
               || (v.valoresX.length > 2 && (op.j === 0 || op.j === v.valoresX.length - 1));
    caja.innerHTML = `<b>${fmt(op.z)}</b><br>${v.ejeX} = ${v.valoresX[op.j]}<br>` +
      `${v.ejeY} = ${v.valoresY[op.i]}<br>` +
      `<span style="opacity:.7">${v.mejor === "max" ? "mayor es mejor" : "menor es mejor"}</span>` +
      (borde ? `<br><span class="aviso">optimo en el borde de la rejilla</span>` : "");
  } else {
    caja.innerHTML = '<span style="opacity:.7">Vista comparativa: una superficie ' +
                     'por modelo, mismo score.</span>';
  }
  const ley = document.getElementById("leyenda");
  ley.innerHTML = "";
  if (v.superficies.length > 1) {
    v.superficies.forEach((s, i) => {
      const d = document.createElement("div");
      d.innerHTML = `<span class="tinta" style="background:${SERIES[i % SERIES.length]}">` +
                    `</span>${s.modelo}`;
      ley.appendChild(d);
    });
  }
  document.getElementById("pie").textContent = v.pie;
}

// Un fallo en el visor dejaria un lienzo en blanco sin explicacion: se muestra en
// la propia pagina, que es donde se va a ver (aqui no hay consola que mire nadie).
try {
  refrescarSelectores(); pintarPanel(); dibujar();
} catch (err) {
  document.querySelector(".lienzo-caja").innerHTML =
    '<div style="padding:22px;font:13px/1.6 monospace;color:#b4451f">' +
    'El visor no ha podido dibujar: ' + (err && err.message ? err.message : err) +
    '<br><br>Los mismos datos estan en los PNG y en rejilla_3d.csv, junto a este ' +
    'fichero.</div>';
}
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# API                                                                           #
# --------------------------------------------------------------------------- #

def escribir(vistas: list[dict], destino: Path, titulo: str) -> Path:
    """Escribe el visor con `vistas` embebidas. Devuelve la ruta del HTML.

    Cada vista es un dict con `metrica`, `modelo`, `ejes` (etiqueta de la pareja),
    `mejor`, `pie`, la geometria que produce `figuras3d._geometria_json`
    (`ejeX`/`ejeY`, `valoresX`/`valoresY` con las etiquetas ya formateadas,
    `escalaX`/`escalaY`, `coordX`/`coordY` de los puntos medidos,
    `coordXf`/`coordYf` de la malla fina y `medidaXf`/`medidaYf`) y
    `superficies` = [{`modelo`, `Z`, `Zf`}]. `Z` son los valores medidos y `Zf` la
    superficie estimada sobre la malla fina, las dos como listas de listas con
    `None` donde no hay valor (JSON no tiene NaN).
    """
    destino = Path(destino)
    datos = {"titulo": titulo, "vistas": vistas}
    html = (_PLANTILLA
            .replace("__TITULO__", titulo)
            .replace("__DATOS__", json.dumps(datos, ensure_ascii=False)))
    destino.write_text(html, encoding="utf-8")
    return destino

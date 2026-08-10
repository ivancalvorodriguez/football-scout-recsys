/* Sigue en vivo la tarea en curso (ingesta o reentrenamiento).
 *
 * La pagina ya llega renderizada con el ultimo estado conocido; esto solo la va
 * refrescando mientras haya algo corriendo, y para de consultar en cuanto la
 * tarea termina. Sin dependencias, como el resto de la app. */
(function () {
  "use strict";

  var panel = document.getElementById("tarea");
  if (!panel) return;

  var url = panel.dataset.url;
  var INTERVALO = 1500;
  var temporizador = null;

  function texto(selector, valor) {
    var nodo = panel.querySelector(selector);
    if (nodo) nodo.textContent = valor;
  }

  function pintar(tarea) {
    if (!tarea) return;
    texto(".tarea-titulo", tarea.titulo);
    texto(".tarea-paso", tarea.paso || "");
    texto(".tarea-tiempo", tarea.segundos + " s");
    texto(".tarea-log", (tarea.lineas || []).join("\n"));
    texto(".tarea-comando", tarea.comando);

    var estado = panel.querySelector(".tarea-estado");
    if (estado) {
      estado.textContent = tarea.estado;
      estado.className = "tarea-estado estado-" + tarea.estado;
    }
    var relleno = panel.querySelector(".barra-relleno");
    if (relleno) {
      var pct = tarea.progreso === null ? 0 : Math.round(tarea.progreso * 1000) / 10;
      relleno.style.width = pct + "%";
    }
    var log = panel.querySelector(".tarea-log");
    if (log) log.scrollTop = log.scrollHeight;
  }

  function consultar() {
    fetch(url, { headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (datos) {
        var tarea = datos && datos.tarea;
        pintar(tarea);
        if (!tarea || tarea.estado !== "en_curso") {
          clearInterval(temporizador);
          temporizador = null;
          // Al terminar, el resto de la pagina (contadores de la BD, fechas de
          // los modelos) se ha quedado vieja: se recarga una sola vez.
          if (tarea && tarea.estado === "terminada") window.location.reload();
        }
      })
      .catch(function () { /* un fallo puntual de red no debe parar el seguimiento */ });
  }

  if (panel.dataset.activa === "1") {
    temporizador = setInterval(consultar, INTERVALO);
    consultar();
  }
})();

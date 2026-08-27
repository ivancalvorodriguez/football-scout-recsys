/* Sigue en vivo la tarea en curso (ingesta o reentrenamiento).
 *
 * La pagina ya llega renderizada con el ultimo estado conocido; esto solo la va
 * refrescando mientras haya algo corriendo, y para de consultar en cuanto la
 * tarea termina. Sin dependencias, como el resto de la app.
 *
 * Dos relojes distintos a proposito:
 *
 * - El SONDEO al servidor va cada 1,5 s. Es lo que trae progreso, paso y estado,
 *   y no tiene sentido pedirlo mas a menudo: el hijo no reporta mas fino que eso
 *   y una peticion cada decima serian 600 al minuto por pestaña abierta.
 * - El CRONOMETRO corre en el navegador cada decima. El tiempo transcurrido no
 *   hace falta preguntarlo: se sabe sumando lo que ha pasado desde el ultimo
 *   dato del servidor, que ademas es lo que evita que el numero salte de 1,5 en
 *   1,5 s. Cada respuesta reajusta la base, asi que no se va acumulando desvio.
 *
 * Lo que NO se pinta: la salida del subproceso ni su linea de comandos. La
 * pagina enseña el estado de la tarea, no una consola; si falla, el servidor
 * manda la razon en una linea (`error`). */
(function () {
  "use strict";

  var panel = document.getElementById("tarea");
  if (!panel) return;

  var url = panel.dataset.url;
  var INTERVALO = 1500;     // sondeo al servidor
  var LATIDO = 100;         // refresco del cronometro
  var temporizador = null;
  var cronometro = null;

  // Base del cronometro: segundos que llevaba la tarea segun el servidor, y el
  // instante local en que se supo. `performance.now()` es monotono, asi que un
  // cambio de hora del sistema no hace saltar el contador hacia atras.
  var segundosBase = parseFloat(panel.dataset.segundos || "0") || 0;
  var marcaBase = performance.now();

  function texto(selector, valor) {
    var nodo = panel.querySelector(selector);
    if (nodo) nodo.textContent = valor;
  }

  function pintarTiempo(segundos) {
    texto(".tarea-tiempo", segundos.toFixed(1) + " s");
  }

  function latir() {
    pintarTiempo(segundosBase + (performance.now() - marcaBase) / 1000);
  }

  function pararCronometro() {
    if (cronometro !== null) {
      clearInterval(cronometro);
      cronometro = null;
    }
  }

  function pintar(tarea) {
    if (!tarea) return;
    texto(".tarea-titulo", tarea.titulo);
    texto(".tarea-paso", tarea.paso || "");

    // El servidor manda la verdad del tiempo; el cronometro solo interpola entre
    // dos respuestas. Si la tarea ya termino, ese valor es el definitivo.
    segundosBase = tarea.segundos;
    marcaBase = performance.now();
    pintarTiempo(segundosBase);

    var error = panel.querySelector(".tarea-error");
    if (error) {
      error.textContent = tarea.error || "";
      error.hidden = !tarea.error;
    }
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
          pararCronometro();
          // Al terminar, el resto de la pagina (contadores de la BD, fechas de
          // los modelos) se ha quedado vieja: se recarga una sola vez.
          if (tarea && tarea.estado === "terminada") window.location.reload();
        }
      })
      .catch(function () { /* un fallo puntual de red no debe parar el seguimiento */ });
  }

  if (panel.dataset.activa === "1") {
    temporizador = setInterval(consultar, INTERVALO);
    cronometro = setInterval(latir, LATIDO);
    consultar();
  }
})();

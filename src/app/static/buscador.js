/*
 * Autocompletado del buscador y recarga de opciones al cambiar de entidad.
 *
 * Vanilla JS a propósito: el proyecto no tiene cadena de build de frontend ni
 * dependencias de CDN (la app debe funcionar sin red). Sin JS el formulario
 * sigue siendo usable: se envía el nombre escrito y el servidor lo resuelve.
 */
(function () {
  "use strict";

  var formulario = document.querySelector(".buscador");
  if (!formulario) return;

  var entrada = formulario.querySelector("#nombre");
  var oculto = formulario.querySelector("#id-entidad");
  var lista = formulario.querySelector("#sugerencias");
  var selectorEntidad = formulario.querySelector("#entidad");
  var selectorModelo = formulario.querySelector("#modelo");
  var apiSugerencias = formulario.dataset.apiSugerencias;

  // Al cambiar jugador/equipo cambian los modelos disponibles y el universo de
  // nombres: se recarga el formulario en vez de mantener una selección inválida.
  // El modelo elegido se arrastra (cada modelo trae las dos entidades); si el
  // reentrenamiento aún no ha dejado la otra, el servidor lo dirá.
  if (selectorEntidad && selectorEntidad.dataset.recarga) {
    selectorEntidad.addEventListener("change", function () {
      var destino = selectorEntidad.dataset.recarga +
        "?entidad=" + encodeURIComponent(selectorEntidad.value);
      if (selectorModelo && selectorModelo.value) {
        destino += "&modelo=" + encodeURIComponent(selectorModelo.value);
      }
      window.location = destino;
    });
  }

  if (!entrada || !lista || !apiSugerencias) return;

  var temporizador = null;
  var peticion = 0;

  function ocultar() {
    lista.hidden = true;
    lista.innerHTML = "";
  }

  function parametros(texto) {
    var p = new URLSearchParams();
    p.set("entidad", selectorEntidad ? selectorEntidad.value : "");
    p.set("q", texto);
    if (selectorModelo && selectorModelo.value) p.set("modelo", selectorModelo.value);
    return p;
  }

  function pintar(sugerencias) {
    lista.innerHTML = "";
    if (!sugerencias.length) {
      ocultar();
      return;
    }
    sugerencias.forEach(function (s) {
      var li = document.createElement("li");
      var boton = document.createElement("button");
      boton.type = "button";
      boton.className = "sugerencia";
      var nombre = document.createElement("span");
      nombre.className = "sugerencia-nombre";
      nombre.textContent = s.nombre;
      boton.appendChild(nombre);
      // Equipo y liga desambiguan homónimos (hay varios "Garcia" en el dataset).
      // `contexto` es la trayectoria ya compuesta ("Barcelona (La Liga
      // 2015/2016)"); solo viene en jugadores y solo si hay base de datos, así
      // que sin ella se cae a las ligas del artefacto.
      var nombresLiga = s.ligas_nombre && s.ligas_nombre.length ? s.ligas_nombre : s.ligas;
      var contexto = s.contexto || (nombresLiga ? nombresLiga.join(", ") : "");
      if (contexto) {
        var ligas = document.createElement("span");
        ligas.className = "sugerencia-ligas";
        ligas.textContent = contexto;
        boton.appendChild(ligas);
      }
      // El id fija la identidad: dos entidades pueden compartir nombre.
      boton.addEventListener("click", function () {
        entrada.value = s.nombre;
        oculto.value = s.id;
        ocultar();
        formulario.submit();
      });
      li.appendChild(boton);
      lista.appendChild(li);
    });
    lista.hidden = false;
  }

  function consultar(texto) {
    var mia = ++peticion;
    fetch(apiSugerencias + "?" + parametros(texto).toString(), {
      headers: { "Accept": "application/json" }
    })
      .then(function (r) { return r.ok ? r.json() : { sugerencias: [] }; })
      .then(function (datos) {
        // Descarta respuestas que llegan tarde y pisarían a una más reciente.
        if (mia !== peticion) return;
        pintar(datos.sugerencias || []);
      })
      .catch(ocultar);
  }

  entrada.addEventListener("input", function () {
    // Escribir invalida el id elegido antes: manda lo que se ve en el campo.
    oculto.value = "";
    var texto = entrada.value.trim();
    window.clearTimeout(temporizador);
    if (texto.length < 2) {
      ocultar();
      return;
    }
    temporizador = window.setTimeout(function () { consultar(texto); }, 150);
  });

  entrada.addEventListener("keydown", function (e) {
    if (e.key === "Escape") ocultar();
  });

  document.addEventListener("click", function (e) {
    if (!formulario.contains(e.target)) ocultar();
  });
})();

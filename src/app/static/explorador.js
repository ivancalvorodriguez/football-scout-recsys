/* Explorador de carpetas del servidor, para el campo «Carpeta del paquete».
 *
 * El navegador no puede dar la ruta absoluta de una carpeta local (un
 * <input type="file" webkitdirectory> entrega los ficheros, no dónde están), y
 * es el servidor quien tiene que abrir esa ruta después. Así que el árbol lo
 * sirve él (`/datos/carpetas`) y esto solo lo pinta.
 *
 * Sin dependencias, como el resto de la app. Sin JS el campo de texto sigue
 * siendo editable a mano: esto es un atajo, no el único camino. */
(function () {
  "use strict";

  var boton = document.getElementById("abrir-explorador");
  var panel = document.getElementById("explorador");
  if (!boton || !panel) return;

  var url = boton.dataset.url;
  var destino = document.getElementById(boton.dataset.destino);
  var caja = panel.querySelector(".explorador-caja");
  var rotuloRuta = panel.querySelector(".explorador-ruta");
  var unidades = panel.querySelector(".explorador-unidades");
  var lista = panel.querySelector(".explorador-lista");
  var error = panel.querySelector(".explorador-error");
  var actual = "";

  function fila(texto, ruta, clase) {
    var li = document.createElement("li");
    var b = document.createElement("button");
    b.type = "button";
    b.className = clase || "explorador-item";
    b.textContent = texto;
    b.addEventListener("click", function () { cargar(ruta); });
    li.appendChild(b);
    return li;
  }

  function pintar(datos) {
    actual = datos.ruta || "";
    rotuloRuta.textContent = actual;
    error.hidden = !datos.error;
    error.textContent = datos.error || "";

    unidades.innerHTML = "";
    (datos.unidades || []).forEach(function (u) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "explorador-unidad";
      b.textContent = u;
      b.addEventListener("click", function () { cargar(u); });
      unidades.appendChild(b);
    });

    lista.innerHTML = "";
    if (datos.padre) lista.appendChild(fila("⬆ " + datos.padre, datos.padre));
    (datos.carpetas || []).forEach(function (c) {
      lista.appendChild(fila("📁 " + c.nombre, c.ruta));
    });
    if (datos.truncado) {
      var li = document.createElement("li");
      li.className = "apunte";
      li.textContent = "…hay más carpetas de las que caben en la lista.";
      lista.appendChild(li);
    }
    // Marca la pista de que estamos justo donde hay que estar. La validación
    // de verdad la hace «Solo validar», que sí sabe qué exige cada fichero.
    if (datos.es_paquete) {
      var ok = document.createElement("li");
      ok.className = "explorador-ok";
      ok.textContent = "✓ Esta carpeta tiene formato de paquete StatsBomb.";
      lista.insertBefore(ok, lista.firstChild);
    }
  }

  function cargar(ruta) {
    fetch(url + "?ruta=" + encodeURIComponent(ruta || ""),
          { headers: { Accept: "application/json" } })
      .then(function (r) { return r.json(); })
      .then(pintar)
      .catch(function () {
        error.hidden = false;
        error.textContent = "No se pudo leer la carpeta.";
      });
  }

  function abrir() {
    panel.hidden = false;
    cargar(destino && destino.value ? destino.value : "");
  }

  function cerrar() {
    panel.hidden = true;
  }

  boton.addEventListener("click", abrir);
  panel.addEventListener("click", function (e) {
    var accion = e.target.dataset ? e.target.dataset.accion : null;
    if (accion === "cancelar" || !caja.contains(e.target)) {
      cerrar();
    } else if (accion === "elegir") {
      if (destino && actual) destino.value = actual;
      cerrar();
    }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !panel.hidden) cerrar();
  });
})();

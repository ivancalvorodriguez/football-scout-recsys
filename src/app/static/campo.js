/*
 * Globo informativo de las marcas del campo de posiciones.
 *
 * Vanilla JS, como `buscador.js`: el proyecto no tiene cadena de build de
 * frontend ni dependencias de CDN (la app debe funcionar sin red).
 *
 * Es una MEJORA sobre algo que ya funciona: cada marca lleva un <title> SVG con
 * el mismo texto (`contexto.MarcaCampo.rotulo`), que el navegador enseña como
 * tooltip nativo. Sin JavaScript esa sigue siendo la lectura. Con JavaScript se
 * cambia por este globo, que sale al instante (el nativo tarda ~1 s), respeta la
 * tipografia de la app y puede colorear cada porcentaje con el color de su
 * jugador — que es justo lo que en la figura distingue a uno de otro.
 *
 * Al sustituirlo, el texto del <title> pasa a `aria-label` del grupo: el globo es
 * decorativo (`aria-hidden`) y sin eso la marca se quedaria sin nombre accesible.
 */
(function () {
  "use strict";

  var figuras = document.querySelectorAll(".campo-juego");
  if (!figuras.length) return;

  // Separacion entre el globo y el borde de la marca, y margen minimo con el
  // borde de la figura al empujarlo para que no se salga.
  var HUECO = 8;
  var MARGEN = 4;

  function fila(nombre, valor, clase) {
    var li = document.createElement("li");
    var punto = document.createElement("span");
    // `marca-pct-a`/`-b` ya definen el color de cada jugador (los mismos que el
    // radar y la leyenda): el punto solo lo hereda con `currentColor`.
    punto.className = "campo-globo-punto " + clase;
    var texto = document.createElement("span");
    texto.className = "campo-globo-nombre";
    texto.textContent = nombre;
    var cifra = document.createElement("span");
    cifra.className = "campo-globo-cifra " + clase;
    cifra.textContent = valor;
    li.appendChild(punto);
    li.appendChild(texto);
    li.appendChild(cifra);
    return li;
  }

  function contenido(globo, grupo, nombreA, nombreB) {
    globo.innerHTML = "";
    var titulo = document.createElement("p");
    titulo.className = "campo-globo-titulo";
    titulo.textContent = grupo.dataset.posicion || "";
    globo.appendChild(titulo);

    var pctB = grupo.dataset.pctB;
    if (pctB === undefined) {
      // Un solo jugador: su nombre ya encabeza la pagina y no hay otro color con
      // el que confundirlo, asi que la cifra se explica sola y va sin punto ni
      // nombre («92 % de sus minutos»), igual que en el <title>.
      var solo = document.createElement("p");
      solo.className = "campo-globo-solo";
      var cifra = document.createElement("span");
      cifra.className = "campo-globo-cifra marca-pct-a";
      cifra.textContent = grupo.dataset.pctA;
      solo.appendChild(cifra);
      solo.appendChild(document.createTextNode(" de sus minutos"));
      globo.appendChild(solo);
      return;
    }
    var lista = document.createElement("ul");
    lista.className = "campo-globo-filas";
    lista.appendChild(fila(nombreA, grupo.dataset.pctA, "marca-pct-a"));
    lista.appendChild(fila(nombreB, pctB, "marca-pct-b"));
    globo.appendChild(lista);
  }

  function colocar(figura, globo, grupo) {
    var marca = grupo.querySelector("circle");
    if (!marca) return;
    var caja = marca.getBoundingClientRect();
    var marco = figura.getBoundingClientRect();
    var ancho = globo.offsetWidth;
    var alto = globo.offsetHeight;

    // Centrado sobre la marca, y empujado hacia dentro si se sale por un lado:
    // las marcas de banda (x=13 y x=87) quedan pegadas al borde de la figura.
    var centro = caja.left + caja.width / 2 - marco.left;
    var izquierda = Math.min(
      Math.max(centro - ancho / 2, MARGEN),
      Math.max(marco.width - ancho - MARGEN, MARGEN)
    );

    // Por defecto encima de la marca; debajo si ahi no cabe (las posiciones de
    // ataque estan a y=12, contra el borde superior del campo).
    var arriba = caja.top - marco.top - alto - HUECO;
    var abajo = arriba < 0;
    if (abajo) arriba = caja.bottom - marco.top + HUECO;

    globo.classList.toggle("campo-globo-abajo", abajo);
    globo.style.left = izquierda + "px";
    globo.style.top = arriba + "px";
    // La flecha apunta a la marca aunque el globo se haya empujado de lado.
    globo.style.setProperty("--flecha", (centro - izquierda) + "px");
  }

  function preparar(figura) {
    var globo = figura.querySelector(".campo-globo");
    var grupos = figura.querySelectorAll(".marca-grupo");
    if (!globo || !grupos.length) return;
    globo.setAttribute("aria-hidden", "true");
    var nombreA = figura.dataset.nombreA || "";
    var nombreB = figura.dataset.nombreB || "";

    // Marca cuyo globo esta abierto, para no repintarlo al pasar del circulo al
    // texto de dentro (son hermanos: `mouseover` salta en cada uno).
    var abierta = null;

    function mostrar(grupo) {
      if (abierta === grupo) return;
      abierta = grupo;
      contenido(globo, grupo, nombreA, nombreB);
      globo.hidden = false;
      colocar(figura, globo, grupo);
    }

    function ocultar() {
      abierta = null;
      globo.hidden = true;
    }

    Array.prototype.forEach.call(grupos, function (grupo) {
      var titulo = grupo.querySelector("title");
      if (titulo) {
        // Con el <title> puesto saldrian los dos tooltips, el nativo encima del
        // globo. Se quita, pero su texto es el nombre accesible de la marca.
        grupo.setAttribute("role", "img");
        grupo.setAttribute("aria-label", titulo.textContent.trim());
        titulo.parentNode.removeChild(titulo);
      }
      // `mouseover` y no `mouseenter`: el segundo no llega en todos los casos
      // sobre elementos SVG (se comprobo con el navegador). `mouseover` burbujea
      // desde el circulo y los textos, de ahi el guardia de `abierta`; para
      // cerrar sirve `mouseleave`, que a diferencia de `mouseout` no salta al
      // moverse entre los hijos del grupo.
      grupo.addEventListener("mouseover", function () { mostrar(grupo); });
      grupo.addEventListener("focus", function () { mostrar(grupo); });
      grupo.addEventListener("mouseleave", ocultar);
      grupo.addEventListener("blur", ocultar);
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") ocultar();
    });
    // Al hacer scroll o cambiar el ancho, la posicion guardada deja de valer.
    window.addEventListener("resize", ocultar);
  }

  Array.prototype.forEach.call(figuras, preparar);
})();

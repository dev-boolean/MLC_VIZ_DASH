/* Movimiento: respeta la preferencia del sistema, salvo que la persona lo active en la barra lateral o con ?movimiento=1. */
(function () {
  "use strict";
  var KEY = "li-movimiento", v = null, q = new URLSearchParams(location.search).get("movimiento");
  try { if (q !== null) localStorage.setItem(KEY, q === "0" ? "0" : "1"); v = localStorage.getItem(KEY); } catch (e) { v = q; }
  var sys = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.LIReduced = v === "1" ? false : v === "0" ? true : sys;
  document.documentElement.classList.toggle("li-still", window.LIReduced);
  function label() { var b = document.getElementById("li-motion"); if (b) { b.textContent = "Animaciones: " + (window.LIReduced ? "no" : "sí"); b.setAttribute("aria-pressed", String(!window.LIReduced)); } else setTimeout(label, 300); }
  document.addEventListener("click", function (e) {
    if (!e.target.closest("#li-motion")) return;
    try { localStorage.setItem(KEY, window.LIReduced ? "1" : "0"); } catch (x) { /* sin almacenamiento */ }
    location.reload();
  });
  label();
})();

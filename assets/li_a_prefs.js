/* Movimiento: las animaciones están siempre activas, sin importar la preferencia «reducir movimiento» del sistema. */
(function () {
  "use strict";
  window.LIReduced = false;
  try { localStorage.removeItem("li-movimiento"); } catch (e) { /* sin almacenamiento */ }
})();

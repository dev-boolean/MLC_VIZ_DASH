/* Introducción animada: de los datos a la plataforma.
   Dibuja la tasa de desempleo observada de Colombia (enero 2001 a diciembre 2025, la misma serie del EDA, incrustada en
   la página por app.py) mes a mes en un gráfico 2D, se detiene en el último dato y lo lleva hasta Colombia en el globo
   de la portada. Se muestra una vez por pestaña; «Ver la introducción» la repite. Si algo falla, la plataforma abre igual. */
(function () {
  "use strict";
  var KEY = "li-intro-visto", NS = "http://www.w3.org/2000/svg";
  var MON = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"];
  var MONTH = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"];
  // Tiempos (ms) desde el inicio: título, marco del gráfico, dibujo, cierre, mensaje y paso al globo.
  var T = { frame: 700, draw: 1300, drawMs: 6500, msg: 8700, trans: 9600, safety: 20000 };

  var D = null;
  try {
    var raw = JSON.parse(document.getElementById("li-intro-data").textContent);
    if (raw.fechas.length === raw.valores.length && raw.valores.length > 1) D = raw;
  } catch (e) { D = null; }

  var st = { run: false };

  function num(v, dec) { return v.toFixed(dec).replace(".", ","); }
  function ym(s) { var p = s.split("-"); return { y: +p[0], m: +p[1] - 1 }; }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function svg(tag, attrs, parent) { var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
  function later(fn, ms) { st.timers.push(setTimeout(function () { if (st.run) safe(fn); }, ms)); }
  function safe(fn) { try { fn(); } catch (e) { if (window.console) console.warn("Introducción:", e); finish(true); } }
  function ease(p) { return .5 * p + .25 * (1 - Math.cos(Math.PI * p)); }  // casi lineal, suave en los extremos

  /* ---------------------------------------------------------------- estructura */
  function build() {
    var n = D.valores.length, a = ym(D.fechas[0]), b = ym(D.fechas[n - 1]);
    var o = el("div", "li-intro");
    o.id = "li-intro";
    o.setAttribute("role", "dialog");
    o.setAttribute("aria-modal", "true");
    o.setAttribute("aria-label", "Introducción animada: tasa de desempleo de Colombia, " + a.y + " a " + b.y);
    o.innerHTML =
      '<button type="button" class="lii-skip">Saltar introducción <span aria-hidden="true">→</span></button>' +
      '<header class="lii-head">' +
      '  <div class="lii-brand"><h1 class="lii-title">LABOR INSIGHTS</h1><p class="lii-sub">Global Labor Market Intelligence</p></div>' +
      '  <div class="lii-topic"><p class="lii-kicker">El mercado laboral de Colombia</p>' +
      '  <p class="lii-range">' + MONTH[a.m].toUpperCase() + " " + a.y + " — " + MONTH[b.m].toUpperCase() + " " + b.y + "</p></div>" +
      '  <div class="lii-final" aria-live="polite"><p class="lii-f1"></p><p class="lii-f2"></p><p class="lii-f3">De la evidencia histórica a la inteligencia predictiva.</p></div>' +
      "</header>" +
      '<div class="lii-stage">' +
      '  <div class="lii-chart">' +
      '    <div class="lii-read" aria-hidden="true"><span class="chip obs">Observado</span><b class="lii-date"></b><span class="lii-val"></span><small class="lii-count"></small></div>' +
      "  </div>" +
      '  <p class="lii-src">Tasa de desempleo nacional (%), mensual. Fuente: ' + D.fuente + ". " +
      (D.interpolados ? D.interpolados + " meses interpolados en la fuente del proyecto." : "Datos observados; ningún mes interpolado.") + "</p>" +
      "</div>" +
      '<p class="lii-sr"></p>';
    return o;
  }

  /* Geometría del gráfico; se recalcula al cambiar el tamaño de la ventana. */
  function layout() {
    var box = st.chart, W = box.clientWidth, H = box.clientHeight, n = D.valores.length, small = W < 640;
    var ml = small ? 36 : 52, mr = small ? 14 : 28, mt = small ? 92 : 28, mb = small ? 30 : 36;
    var max = Math.max.apply(null, D.valores), top = Math.ceil(max / 5) * 5, step = 5;
    if (st.svg) st.svg.remove();
    var s = svg("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, "aria-hidden": "true", class: "lii-svg" });
    var X = [], Y = [], x = function (i) { return ml + i / (n - 1) * (W - ml - mr); }, y = function (v) { return mt + (1 - v / top) * (H - mt - mb); };
    // Cuadrícula y eje vertical (escala desde 0: no se exageran los cambios).
    var g = svg("g", { class: "lii-grid" }, s);
    for (var v = 0; v <= top; v += step) {
      svg("line", { x1: ml, x2: W - mr, y1: y(v), y2: y(v) }, g);
      var t = svg("text", { x: ml - 8, y: y(v) + 4, "text-anchor": "end" }, g); t.textContent = v + (small ? "" : " %");
    }
    // Años: el primero, el último y algunos intermedios, según el ancho.
    var first = ym(D.fechas[0]).y, last = ym(D.fechas[n - 1]).y, every = small ? 8 : W < 1000 ? 4 : 3;
    for (var yr = first; yr <= last; yr += every) {
      var i = D.fechas.indexOf(yr + "-01");
      if (i < 0) continue;
      svg("line", { x1: x(i), x2: x(i), y1: H - mb, y2: H - mb + 5 }, g);
      var tx = svg("text", { x: x(i), y: H - mb + 20, "text-anchor": "middle" }, g); tx.textContent = yr;
    }
    // Línea, área y marcador. La línea se revela con stroke-dashoffset hasta el vértice k (sin puntos intermedios).
    var d = "", cum = [0], L = 0;
    for (i = 0; i < n; i++) {
      X.push(x(i)); Y.push(y(D.valores[i]));
      d += (i ? "L" : "M") + X[i].toFixed(2) + " " + Y[i].toFixed(2);
      if (i) { L += Math.hypot(X[i] - X[i - 1], Y[i] - Y[i - 1]); cum.push(L); }
    }
    var defs = svg("defs", {}, s), cp = svg("clipPath", { id: "lii-clip" }, defs);
    st.clip = svg("rect", { x: ml, y: 0, width: 0, height: H }, cp);
    svg("path", { d: d + "L" + X[n - 1].toFixed(2) + " " + y(0) + "L" + X[0].toFixed(2) + " " + y(0) + "Z", class: "lii-area", "clip-path": "url(#lii-clip)" }, s);
    st.line = svg("path", { d: d, class: "lii-line", "stroke-dasharray": L + " " + L, "stroke-dashoffset": L }, s);
    // Hito: el máximo de la serie (dato observado, sin atribuir causas).
    var pk = D.valores.indexOf(max), pd = ym(D.fechas[pk]);
    var right = X[pk] + 250 < W - mr;
    st.peak = svg("g", { class: "lii-peak" }, s);
    svg("circle", { cx: X[pk], cy: Y[pk], r: 3.5 }, st.peak);
    var pt = svg("text", { x: X[pk] + (right ? 12 : -12), y: Y[pk] + 4, "text-anchor": right ? "start" : "end" }, st.peak);
    pt.textContent = "Máximo de la serie: " + num(max, 2) + " % · " + MON[pd.m].toLowerCase() + " " + pd.y;
    st.halo = svg("circle", { r: 9, class: "lii-halo" }, s);
    st.dot = svg("circle", { r: 4.5, class: "lii-dot" }, s);
    box.insertBefore(s, box.firstChild);
    st.svg = s; st.X = X; st.Y = Y; st.cum = cum; st.L = L; st.pk = pk;
    var k = st.k; st.k = -1; if (k >= 0) draw(k);
  }

  function draw(k) {
    if (k === st.k) return;
    st.k = k;
    st.line.setAttribute("stroke-dashoffset", st.L - st.cum[k]);
    st.clip.setAttribute("width", Math.max(0, st.X[k] - st.X[0]));
    st.dot.setAttribute("cx", st.X[k]); st.dot.setAttribute("cy", st.Y[k]);
    st.halo.setAttribute("cx", st.X[k]); st.halo.setAttribute("cy", st.Y[k]);
    var f = ym(D.fechas[k]);
    st.date.textContent = MON[f.m] + " " + f.y;  // siempre el mes del último dato revelado
    st.val.textContent = num(D.valores[k], 2) + " %";
    st.count.textContent = "Observación " + (k + 1) + " de " + D.valores.length;
    st.peak.classList.toggle("on", k >= st.pk);
  }

  /* ---------------------------------------------------------------- secuencia */
  function play() {
    if (st.run || !D) return;
    var html = document.documentElement;
    st = { run: true, k: -1, timers: [], anims: [], raf: 0, t0: 0 };
    html.classList.remove("li-intro-rev");
    html.classList.add("li-intro-on");
    var o = build();
    document.body.appendChild(o);
    st.o = o; st.chart = o.querySelector(".lii-chart"); st.stage = o.querySelector(".lii-stage");
    st.date = o.querySelector(".lii-date"); st.val = o.querySelector(".lii-val"); st.count = o.querySelector(".lii-count");
    var n = D.valores.length, a = ym(D.fechas[0]), b = ym(D.fechas[n - 1]);
    o.querySelector(".lii-sr").textContent = "Serie mensual observada de la tasa de desempleo de Colombia: " + num(D.valores[0], 2) + " % en " +
      MONTH[a.m] + " de " + a.y + " y " + num(D.valores[n - 1], 2) + " % en " + MONTH[b.m] + " de " + b.y + ".";
    o.querySelector(".lii-skip").addEventListener("click", function () { finish(true); });
    st.key = function (e) { if (e.key === "Escape") finish(true); };
    // Se redibuja si cambia el tamaño del gráfico (ventana, o el encabezado que crece con el mensaje final).
    st.size = [0, 0];
    st.resize = function () {
      var w = st.chart.clientWidth, h = st.chart.clientHeight;
      if (w === st.size[0] && h === st.size[1]) return;
      st.size = [w, h]; clearTimeout(st.rz);
      st.rz = setTimeout(function () { if (st.run && !st.leaving) safe(layout); }, 60);
    };
    document.addEventListener("keydown", st.key);
    window.addEventListener("resize", st.resize);
    if (window.ResizeObserver) { st.ro = new ResizeObserver(st.resize); st.ro.observe(st.chart); }
    window.scrollTo(0, 0);
    layout();
    st.size = [st.chart.clientWidth, st.chart.clientHeight];
    draw(0);
    o.querySelector(".lii-skip").focus({ preventScroll: true });
    requestAnimationFrame(function () { o.classList.add("s-head"); });
    st.timers.push(setTimeout(function () { finish(true); }, T.safety));  // nunca bloquea la plataforma

    if (window.LIReduced) {  // sin movimiento: gráfico completo y paso directo a la portada
      o.classList.add("s-chart", "s-final", "s-msg");
      draw(n - 1); finalText();
      later(function () { finish(true); }, 2600);
      return;
    }
    later(function () { o.classList.add("s-chart"); }, T.frame);
    later(function () { st.t0 = performance.now(); st.raf = requestAnimationFrame(frame); }, T.draw);
  }

  function frame(now) {
    if (!st.run) return;
    safe(function () {
      var p = Math.min(1, (now - st.t0) / T.drawMs);
      draw(Math.round(ease(p) * (D.valores.length - 1)));
      if (p < 1) { st.raf = requestAnimationFrame(frame); return; }
      st.o.classList.add("s-final"); finalText();
      later(function () { st.o.classList.add("s-msg"); }, T.msg - T.draw - T.drawMs);
      later(transition, T.trans - T.draw - T.drawMs);
    });
  }

  function finalText() {
    var n = D.valores.length, b = ym(D.fechas[n - 1]), years = Math.round(n / 12);
    st.o.querySelector(".lii-f1").textContent = MONTH[b.m].toUpperCase() + " " + b.y + " · " + num(D.valores[n - 1], 2) + " %";
    st.o.querySelector(".lii-f2").textContent = years + " años de dinámica del mercado laboral.";
  }

  /* Del gráfico al globo: el gráfico se recoge sobre su último punto y ese punto viaja hasta Colombia en el globo. */
  function transition(tries) {
    var globe = document.querySelector(".hero-globe");
    if (!globe && (tries || 0) < 30) { later(function () { transition((tries || 0) + 1); }, 100); return; }  // la portada aún carga
    st.leaving = true;
    var o = st.o, dot = st.dot.getBoundingClientRect(), from = { x: dot.left + dot.width / 2, y: dot.top + dot.height / 2 };
    var to = null;
    if (globe) {
      try { to = window.LIGlobe && LIGlobe.anchor("colombia"); } catch (e) { to = null; }
      if (!to) { var r = globe.getBoundingClientRect(); to = { x: r.left + r.width / 2, y: r.top + r.height / 2 }; }
    }
    if (!to || !o.animate) { finish(true); return; }
    var dx = to.x - from.x, dy = to.y - from.y, sr = st.stage.getBoundingClientRect(), curve = "cubic-bezier(.65,0,.25,1)";
    st.stage.style.transformOrigin = (from.x - sr.left) + "px " + (from.y - sr.top) + "px";
    st.anims.push(st.stage.animate([{ transform: "none", opacity: 1 }, { opacity: 1, offset: .35 }, { transform: "translate(" + dx + "px," + dy + "px) scale(.12)", opacity: 0 }],
      { duration: 1500, easing: curve, fill: "forwards" }));
    st.anims.push(o.querySelector(".lii-head").animate([{ opacity: 1 }, { opacity: 0, transform: "translateY(-10px)" }], { duration: 450, fill: "forwards" }));
    st.anims.push(o.animate([{ backgroundColor: "rgba(246,247,245,1)" }, { backgroundColor: "rgba(246,247,245,0)" }], { duration: 1100, delay: 350, easing: "ease-out", fill: "forwards" }));
    var fly = el("div", "lii-fly");
    fly.style.left = from.x + "px"; fly.style.top = from.y + "px";
    o.appendChild(fly);
    var move = fly.animate([{ transform: "translate(-50%,-50%)" }, { transform: "translate(calc(-50% + " + dx + "px), calc(-50% + " + dy + "px))" }],
      { duration: 1500, easing: curve, fill: "forwards" });
    st.anims.push(move);
    later(reveal, 300);
    move.onfinish = function () {
      if (!st.run) return;
      st.anims.push(fly.animate([{ transform: "translate(calc(-50% + " + dx + "px), calc(-50% + " + dy + "px)) scale(1)", opacity: 1 },
        { transform: "translate(calc(-50% + " + dx + "px), calc(-50% + " + dy + "px)) scale(3.2)", opacity: 0 }], { duration: 650, easing: "ease-out", fill: "forwards" }));
      later(function () { cleanup(); }, 700);
    };
  }

  /* La portada aparece: el globo y el titular vuelven con una transición CSS (labor.css, html.li-intro-rev). */
  function reveal() {
    var html = document.documentElement;
    if (!html.classList.contains("li-intro-on")) return;
    html.classList.add("li-intro-rev");
    void html.offsetWidth;
    html.classList.remove("li-intro-on");
  }

  /* Saltar (o cualquier fallo): corta todo y abre la portada de inmediato. */
  function finish(fast) {
    if (!st.run) return;
    st.leaving = true;
    cancelAnimationFrame(st.raf);
    st.timers.forEach(clearTimeout); st.timers = [];
    // Las animaciones en curso saltan a su estado final (no vuelven al gráfico completo).
    st.anims.forEach(function (a) { try { a.onfinish = null; a.finish(); } catch (e) { /* ya terminó */ } }); st.anims = [];
    reveal();
    var o = st.o, cur = st, done = function () { if (st === cur) cleanup(); };
    if (fast && o && o.animate) {
      o.animate([{ opacity: getComputedStyle(o).opacity }, { opacity: 0 }], { duration: 350, easing: "ease-out", fill: "forwards" }).onfinish = done;
      setTimeout(done, 600);
    } else cleanup();
  }

  function cleanup() {
    if (!st.run) return;
    st.run = false;
    cancelAnimationFrame(st.raf);
    st.timers.forEach(clearTimeout);
    clearTimeout(st.rz);
    document.removeEventListener("keydown", st.key);
    window.removeEventListener("resize", st.resize);
    if (st.ro) st.ro.disconnect();
    if (st.o) st.o.remove();
    var html = document.documentElement;
    html.classList.remove("li-intro-on");
    setTimeout(function () { if (!st.run) html.classList.remove("li-intro-rev"); }, 1800);
    try { sessionStorage.setItem(KEY, "1"); } catch (e) { /* sin almacenamiento */ }
    st = { run: false };
  }

  /* Repetir desde cualquier sección: vuelve a la portada (sin recargar) y empieza de nuevo. */
  function replay() {
    if (st.run || !D) return;
    document.body.classList.remove("menu");
    if (location.pathname !== "/") {
      window.history.pushState({}, "", "/");
      window.dispatchEvent(new CustomEvent("_dashprivate_pushstate"));
    }
    safe(play);
  }

  document.addEventListener("click", function (e) { if (e.target.closest && e.target.closest("#li-intro-replay")) { e.preventDefault(); replay(); } });
  window.LIIntro = { play: function () { safe(play); }, replay: replay, skip: function () { finish(true); } };

  var seen = false;
  try { seen = sessionStorage.getItem(KEY) === "1"; } catch (e) { seen = false; }
  if (D && !seen && location.pathname === "/") {
    if (document.body) safe(play); else document.addEventListener("DOMContentLoaded", function () { safe(play); });
  }
})();

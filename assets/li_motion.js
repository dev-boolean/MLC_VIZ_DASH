/* Movimiento sutil (GSAP): entradas al hacer scroll, contadores y dibujo progresivo del pronóstico.
   Siempre activo; sin GSAP (sin internet), todo queda visible de inmediato. */
(function () {
  "use strict";
  var reduced = !!window.LIReduced;
  var ok = function () { return window.gsap && !reduced; };
  var timer = null, lastPath = null;
  var PRED = "#258477";

  function fmt(v, dec, sign) {
    var s = Math.abs(v).toFixed(dec).replace(".", ",");
    var neg = v < 0 && parseFloat(s.replace(",", ".")) !== 0;
    return (neg ? "−" : sign && v > 0 ? "+" : "") + s;
  }

  function reveal(el) {
    el.dataset.rv = "1";
    if (!ok()) return;
    gsap.fromTo(el, { y: 12, autoAlpha: 0 }, {
      y: 0, autoAlpha: 1, duration: .45, ease: "power2.out", clearProps: "transform",
      scrollTrigger: { trigger: el, start: "top 96%", once: true }
    });
  }

  function title(el) {
    el.dataset.rv = "1";
    if (!ok()) return;
    gsap.fromTo(el.children, { y: 14, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: .5, stagger: .08, ease: "power2.out", clearProps: "transform" });
  }

  function count(el) {
    el.dataset.rv = "1";
    var v = parseFloat(el.dataset.count), dec = +el.dataset.dec, sg = el.dataset.sign === "1", o = { x: v * 0.9 };
    if (!ok() || isNaN(v)) return;
    var final = fmt(v, dec, sg);
    gsap.to(o, {
      x: v, duration: .6, ease: "power1.out", scrollTrigger: { trigger: el, start: "top 98%", once: true },
      onUpdate: function () { el.textContent = fmt(o.x, dec, sg); }, onComplete: function () { el.textContent = final; }
    });
  }

  /* Historia → último dato → segmento de un mes → intervalo y valor. Menos de 1,5 s y no bloquea. */
  function forecast(gd) {
    if (!ok() || !gd || !gd.querySelector) return;
    var tr = gd.querySelectorAll(".scatterlayer .trace"), data = gd.data || [];
    if (tr.length < 3) return;
    var hist = tr[0].querySelector(".js-line"), seg = null, rest = [];
    for (var i = 1; i < tr.length; i++) {
      var d = data[i] || {}, line = tr[i].querySelector(".js-line");
      if (d.line && d.line.color === PRED && d.mode === "lines") seg = line; else rest.push(tr[i]);
    }
    if (!hist || !seg) return;
    var cut = gd.querySelectorAll(".shapelayer path"), ann = gd.querySelectorAll(".annotation"), L = hist.getTotalLength(), l = seg.getTotalLength();
    gd._li && gd._li.kill();
    var done = function () {
      hist.style.strokeDasharray = ""; hist.style.strokeDashoffset = ""; seg.style.strokeDasharray = ""; seg.style.strokeDashoffset = "";
      gsap.set(rest, { clearProps: "opacity" }); gsap.set(cut, { clearProps: "opacity" }); gsap.set(ann, { clearProps: "opacity" });
    };
    var tl = gd._li = gsap.timeline({ onComplete: done });
    tl.set(rest, { opacity: 0 }).set(cut, { opacity: 0 }).set(ann, { opacity: 0 })
      .set(hist, { strokeDasharray: L, strokeDashoffset: L }).set(seg, { strokeDasharray: l, strokeDashoffset: l })
      .to(hist, { strokeDashoffset: 0, duration: .8, ease: "power1.inOut" })
      .to(cut, { opacity: 1, duration: .15 }, "-=.05")
      .to(seg, { strokeDashoffset: 0, duration: .3, ease: "power2.out" })
      .to(rest, { opacity: 1, duration: .25, stagger: .03 }, "-=.1")
      .to(ann, { opacity: 1, duration: .2 }, "-=.15");
  }

  function hookForecast(box) {
    var gd = box.querySelector(".js-plotly-plot");
    if (!gd || !gd.querySelector(".scatterlayer .trace .js-line")) return;
    var key = gd.data && gd.data[0] && gd.data[0].y ? gd.data[0].y.length + ":" + JSON.stringify((gd.data[4] || {}).y || "") + (gd.data[2] && gd.data[2].y) : "";
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    forecast(gd);
  }

  function country() {
    var el = document.querySelector("#pais .dash-dropdown-value, #pais");
    return el ? (el.textContent || "").trim() : "";
  }

  function scan() {
    timer = null;
    if (window.gsap && window.ScrollTrigger && !scan.reg) { gsap.registerPlugin(ScrollTrigger); scan.reg = 1; }
    if (location.pathname !== lastPath) {
      lastPath = location.pathname; if (!location.hash) window.scrollTo(0, 0); document.body.classList.remove("menu");
      if (window.ScrollTrigger) ScrollTrigger.getAll().forEach(function (s) { if (!document.body.contains(s.trigger)) s.kill(); });
    }
    document.querySelectorAll(".reveal:not([data-rv])").forEach(reveal);
    document.querySelectorAll(".split:not([data-rv])").forEach(title);
    document.querySelectorAll("[data-count]:not([data-rv])").forEach(count);
    document.querySelectorAll(".forecast-anim").forEach(hookForecast);
    window.LIGlobe && LIGlobe.scan();
    if (window.LIGlobe && fetchMap.ids) { var id = fetchMap.ids[country()]; id && LIGlobe.select(id); }
    if (window.ScrollTrigger) { clearTimeout(scan.rf); scan.rf = setTimeout(function () { ScrollTrigger.refresh(); }, 400); }
  }
  function fetchMap() { fetch("/api/globo").then(function (r) { return r.json(); }).then(function (d) { fetchMap.ids = {}; d.forEach(function (x) { fetchMap.ids[x.name] = x.id; }); queue(); }); }
  function queue() { if (!timer) timer = setTimeout(scan, 60); }

  function progress() {
    var b = document.getElementById("li-progress-bar"), h = document.documentElement;
    if (b) b.style.transform = "scaleX(" + (h.scrollHeight > h.clientHeight ? h.scrollTop / (h.scrollHeight - h.clientHeight) : 0) + ")";
  }

  function start() {
    new MutationObserver(queue).observe(document.body, { childList: true, subtree: true, characterData: true });
    addEventListener("scroll", progress, { passive: true });
    document.addEventListener("click", function (e) {
      if (e.target.closest("#li-burger")) { document.body.classList.toggle("menu"); return; }
      if (document.body.classList.contains("menu") && !e.target.closest(".li-side")) document.body.classList.remove("menu");
    });
    fetchMap(); queue();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();

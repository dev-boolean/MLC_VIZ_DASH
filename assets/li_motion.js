/* Narrativa de scroll (GSAP), animación del pronóstico y escena conceptual del modelo. Sin GSAP todo queda visible. */
(function () {
  "use strict";
  var reduced = !!window.LIReduced;
  var ok = function () { return window.gsap && !reduced; };
  var flows = [], timer = null, lastPath = null;

  function fmt(v, dec, sign) {
    var s = Math.abs(v).toFixed(dec).replace(".", ",");
    var neg = v < 0 && parseFloat(s.replace(",", ".")) !== 0;
    return (neg ? "−" : sign && v > 0 ? "+" : "") + s;
  }

  function reveal(el) {
    el.dataset.rv = "1";
    if (!ok()) return;
    gsap.fromTo(el, { y: 26, autoAlpha: 0 }, {
      y: 0, autoAlpha: 1, duration: .6, ease: "power2.out", clearProps: "transform",
      scrollTrigger: { trigger: el, start: "top 94%", once: true }
    });
  }

  function title(el) {
    el.dataset.rv = "1";
    if (!ok()) return;
    gsap.fromTo(el.children, { y: 34, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: .75, stagger: .1, ease: "power3.out", clearProps: "transform" });
  }

  function count(el) {
    el.dataset.rv = "1";
    var v = parseFloat(el.dataset.count), dec = +el.dataset.dec, sg = el.dataset.sign === "1", o = { x: 0 };
    if (!ok() || isNaN(v)) return;
    var final = fmt(v, dec, sg);
    gsap.to(o, {
      x: v, duration: .9, ease: "power2.out", scrollTrigger: { trigger: el, start: "top 98%", once: true },
      onUpdate: function () { el.textContent = fmt(o.x, dec, sg); }, onComplete: function () { el.textContent = final; }
    });
  }

  /* Historia → último dato → corte → segmento predictivo → valor. Dura menos de dos segundos y no bloquea. */
  function forecast(gd) {
    if (!ok() || !gd || !gd.querySelector) return;
    var tr = gd.querySelectorAll(".scatterlayer .trace"), names = (gd.data || []).map(function (d) { return d; });
    if (tr.length < 3) return;
    var hist = tr[0].querySelector(".js-line"), seg = null, rest = [];
    for (var i = 1; i < tr.length; i++) {
      var d = names[i] || {}, line = tr[i].querySelector(".js-line");
      if (d.line && d.line.color === "#28D8F0" && d.mode === "lines") seg = line; else rest.push(tr[i]);
    }
    if (!hist || !seg) return;
    var cut = gd.querySelectorAll(".shapelayer path"), ann = gd.querySelectorAll(".annotation"), L = hist.getTotalLength(), l = seg.getTotalLength();
    gd._li && gd._li.kill();
    var tl = gd._li = gsap.timeline({ onComplete: function () { hist.style.strokeDasharray = ""; hist.style.strokeDashoffset = ""; seg.style.strokeDasharray = ""; seg.style.strokeDashoffset = ""; gsap.set(rest, { clearProps: "opacity" }); gsap.set(cut, { clearProps: "opacity,strokeWidth" }); gsap.set(ann, { clearProps: "opacity" }); } });
    tl.set(rest, { opacity: 0 }).set(cut, { opacity: 0 }).set(ann, { opacity: 0 })
      .set(hist, { strokeDasharray: L, strokeDashoffset: L }).set(seg, { strokeDasharray: l, strokeDashoffset: l })
      .to(hist, { strokeDashoffset: 0, duration: .95, ease: "power1.inOut" })
      .to(cut, { opacity: 1, duration: .2 }, "-=.05")
      .to(seg, { strokeDashoffset: 0, duration: .35, ease: "power2.out" })
      .to(rest, { opacity: 1, duration: .3, stagger: .04 }, "-=.1")
      .to(ann, { opacity: 1, duration: .25 }, "-=.2");
  }

  function hookForecast(box) {
    var gd = box.querySelector(".js-plotly-plot");
    if (!gd || !gd.querySelector(".scatterlayer .trace .js-line")) return;
    var key = gd.data && gd.data[0] && gd.data[0].y ? gd.data[0].y.length + ":" + JSON.stringify((gd.data[4] || {}).y || "") + (gd.data[2] && gd.data[2].y) : "";
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    forecast(gd);
  }

  /* Escena conceptual: datos que convergen en tres núcleos y salen como una sola línea. Ilustrativa. */
  function Flow(el) {
    this.el = el; this.dead = false; var me = this;
    try { this.r = new THREE.WebGLRenderer({ alpha: true, antialias: false, powerPreference: "low-power" }); } catch (e) { return; }
    this.r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5)); el.appendChild(this.r.domElement);
    this.s = new THREE.Scene(); this.c = new THREE.OrthographicCamera(-1, 1, 1, -1, -5, 5);
    var n = 1500, p = new Float32Array(n * 3);
    for (var i = 0; i < n; i++) { p[i * 3] = Math.random(); p[i * 3 + 1] = Math.random() * 2 - 1; p[i * 3 + 2] = Math.floor(Math.random() * 3); }
    var g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10);
    this.m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { t: { value: 0 }, px: { value: this.r.getPixelRatio() } },
      vertexShader: "uniform float t,px;varying vec3 c;varying float a;void main(){float k=fract(position.x+t*.045);float e=position.z-1.;float x=mix(-1.05,1.05,k);float y0=position.y*1.05;float yc=e*.3+position.y*.03;float s=smoothstep(.12,.5,k);float y=mix(y0,yc,s);float q=smoothstep(.56,.72,k);float line=.1*sin(x*9.)+.04*sin(x*23.);y=mix(y,line+position.y*.006,q);vec3 b=vec3(.22,.52,1.),v=vec3(.58,.4,1.),cy=vec3(.16,.85,.94);c=mix(mix(b,v,smoothstep(.3,.5,k)),cy,q);a=smoothstep(0.,.1,k)*(1.-smoothstep(.93,1.,k))*(.6+.4*q);gl_Position=vec4(x,y*.72,0.,1.);gl_PointSize=(2.2+1.1*q+.7*s)*px;}",
      fragmentShader: "varying vec3 c;varying float a;void main(){float d=length(gl_PointCoord-.5);if(d>.5)discard;gl_FragColor=vec4(c,a*(1.-d*1.6));}"
    });
    var pts = new THREE.Points(g, this.m); pts.frustumCulled = false; this.s.add(pts);
    this.ro = new ResizeObserver(function () { me.r.setSize(el.clientWidth, el.clientHeight); if (reduced) me.draw(6000); }); this.ro.observe(el);
    this.vis = true; this.io = new IntersectionObserver(function (e) { me.vis = e[0].isIntersecting; }); this.io.observe(el);
    if (reduced) { this.draw(6000); return; }
    (function loop(t) { if (me.dead) return; requestAnimationFrame(loop); if (me.vis && !document.hidden) me.draw(t || 0); })(0);
  }
  Flow.prototype.draw = function (t) { this.m.uniforms.t.value = t * .001; this.r.render(this.s, this.c); };
  Flow.prototype.destroy = function () { this.dead = true; this.ro && this.ro.disconnect(); this.io && this.io.disconnect(); if (this.r) { this.m.dispose(); this.r.dispose(); this.r.forceContextLoss && this.r.forceContextLoss(); } };

  function country() {
    var el = document.querySelector("#pais .dash-dropdown-value, #pais");
    return el ? (el.textContent || "").trim() : "";
  }

  function scan() {
    timer = null;
    if (window.gsap && window.ScrollTrigger && !scan.reg) { gsap.registerPlugin(ScrollTrigger); scan.reg = 1; }
    if (location.pathname !== lastPath) {
      lastPath = location.pathname; window.scrollTo(0, 0); document.body.classList.remove("menu");
      if (window.ScrollTrigger) ScrollTrigger.getAll().forEach(function (s) { if (!document.body.contains(s.trigger)) s.kill(); });
    }
    document.querySelectorAll(".reveal:not([data-rv])").forEach(reveal);
    document.querySelectorAll(".split:not([data-rv])").forEach(title);
    document.querySelectorAll("[data-count]:not([data-rv])").forEach(count);
    document.querySelectorAll(".forecast-anim").forEach(hookForecast);
    window.LIGlobe && LIGlobe.scan();
    if (window.THREE) {
      flows = flows.filter(function (f) { if (!document.body.contains(f.el)) { f.destroy(); return false; } return true; });
      document.querySelectorAll(".li-flow:not([data-on])").forEach(function (el) { el.dataset.on = "1"; flows.push(new Flow(el)); });
    }
    if (window.LIGlobe && fetchMap.ids) { var id = fetchMap.ids[country()]; id && LIGlobe.select(id); }
    if (window.ScrollTrigger) { clearTimeout(scan.rf); scan.rf = setTimeout(function () { ScrollTrigger.refresh(); }, 400); }
    var ld = document.getElementById("li-loader");
    if (ld && document.querySelector("#li-nav a") && document.querySelector("#li-main > *")) setTimeout(function () { ld.classList.add("gone"); setTimeout(function () { ld.style.display = "none"; }, 650); }, 250);
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
      var rp = e.target.closest("#fc-replay");
      if (rp) { var gd = document.querySelector("#fc-fig .js-plotly-plot, #fc-fig.js-plotly-plot"); gd && forecast(gd); }
      var nd = e.target.closest(".a-node");
      if (nd) { var i = Array.prototype.indexOf.call(nd.parentNode.querySelectorAll(".a-node"), nd), rb = document.querySelectorAll("#mi-stage input")[i]; rb && rb.click(); }
    });
    document.addEventListener("keydown", function (e) {
      if ((e.key === "Enter" || e.key === " ") && e.target.classList && e.target.classList.contains("a-node")) { e.preventDefault(); e.target.click(); }
    });
    fetchMap(); queue();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();

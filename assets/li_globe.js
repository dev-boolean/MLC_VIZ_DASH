/* Globo 3D de la portada (Three.js r128), en estilo claro y sin efectos permanentes.
   Es ilustrativo: ubica los nueve países, no mide nada. Se destruye al salir de la página. */
(function () {
  "use strict";
  var ATLAS = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json";
  var R = 1, RAD = Math.PI / 180;
  var COL = { sphere: 0xF7F9F8, grid: 0xDCE3E6, land: 0xB9C6D0, active: 0x47799D, sel: 0x183B56 };
  var reduced = !!window.LIReduced;
  var dataP = null, mapP = null, selected = "colombia", live = [];

  function countries() { return dataP || (dataP = fetch("/api/globo").then(function (r) { return r.json(); })); }

  function vec(lat, lon, r) {
    var p = (90 - lat) * RAD, t = (lon + 180) * RAD;
    return new THREE.Vector3(-r * Math.sin(p) * Math.cos(t), r * Math.cos(p), r * Math.sin(p) * Math.sin(t));
  }

  /* Decodifica el TopoJSON y lo pinta en un lienzo equirrectangular: cada color es un país. */
  function worldMap() {
    if (mapP) return mapP;
    mapP = fetch(ATLAS).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); }).then(function (topo) {
      var s = topo.transform.scale, t = topo.transform.translate;
      var arcs = topo.arcs.map(function (a) {
        var x = 0, y = 0;
        return a.map(function (p) { x += p[0]; y += p[1]; return [x * s[0] + t[0], y * s[1] + t[1]]; });
      });
      function ring(ids) {
        var out = [];
        ids.forEach(function (i) { var a = i >= 0 ? arcs[i] : arcs[~i].slice().reverse(); out = out.concat(out.length ? a.slice(1) : a); });
        return out;
      }
      var W = 1440, H = 720, cv = document.createElement("canvas");
      cv.width = W; cv.height = H;
      var g = cv.getContext("2d", { willReadFrequently: true }), borders = {};
      g.imageSmoothingEnabled = false;
      topo.objects.countries.geometries.forEach(function (geo) {
        var iso = parseInt(geo.id, 10) || 0, polys = geo.type === "Polygon" ? [geo.arcs] : geo.type === "MultiPolygon" ? geo.arcs : [];
        g.fillStyle = "rgb(" + (iso >> 8) + "," + (iso & 255) + ",255)";
        polys.forEach(function (poly) {
          g.beginPath();
          poly.forEach(function (rg) {
            var pts = ring(rg);
            pts.forEach(function (p, i) { g[i ? "lineTo" : "moveTo"]((p[0] + 180) / 360 * W, (90 - p[1]) / 180 * H); });
            g.closePath();
            (borders[iso] = borders[iso] || []).push(pts);
          });
          g.fill("evenodd");
        });
      });
      var px = g.getImageData(0, 0, W, H).data;
      return {
        borders: borders,
        at: function (lat, lon) {
          var x = Math.min(W - 1, Math.max(0, Math.floor((lon + 180) / 360 * W))), y = Math.min(H - 1, Math.max(0, Math.floor((90 - lat) / 180 * H)));
          var i = (y * W + x) * 4;
          return px[i + 3] < 200 ? -1 : px[i + 2] === 255 ? (px[i] << 8) + px[i + 1] : 0;
        }
      };
    }).catch(function () { mapP = null; return null; });
    return mapP;
  }

  function Globe(el) {
    this.el = el; this.dead = false; this.visible = true; this.last = 0;
    var w = el.clientWidth || 600, h = el.clientHeight || 500;
    try { this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" }); } catch (e) { el.dataset.nogl = "1"; return; }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
    this.renderer.setClearColor(0, 0);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(36, w / h, 0.1, 50);
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.labels = document.createElement("div"); this.labels.className = "globe-labels"; el.appendChild(this.labels);
    this.rot = { x: 0.1, y: 0 }; this.target = null; this.spin = 0.0004; this.drag = null;
    this.base(); this.resize(); this.bind();
    var me = this;
    countries().then(function (d) { if (me.dead) return; me.data = d; me.markers(); me.focus(selected, true); });
    worldMap().then(function (m) { if (me.dead || !m) return; me.map = m; me.land(); me.paint(); });
    this.loop = this.loop.bind(this); requestAnimationFrame(this.loop);
  }
  var P = Globe.prototype;

  P.base = function () {
    this.root.add(new THREE.Mesh(new THREE.SphereGeometry(R * 0.994, 64, 48), new THREE.MeshBasicMaterial({ color: COL.sphere })));
    var pos = [];
    for (var lat = -60; lat <= 60; lat += 30) for (var lon = -180; lon < 180; lon += 4) pos.push(vec(lat, lon, R * 0.997), vec(lat, lon + 4, R * 0.997));
    for (lon = -180; lon < 180; lon += 30) for (lat = -80; lat < 80; lat += 4) pos.push(vec(lat, lon, R * 0.997), vec(lat + 4, lon, R * 0.997));
    this.root.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pos), new THREE.LineBasicMaterial({ color: COL.grid, transparent: true, opacity: 0.9 })));
  };

  P.land = function () {
    var N = 22000, pos = [], iso = [], ga = Math.PI * (3 - Math.sqrt(5));
    for (var i = 0; i < N; i++) {
      var y = 1 - 2 * (i + .5) / N, lat = Math.asin(y) / RAD, lon = ((i * ga) / RAD) % 360 - 180;
      var id = this.map.at(lat, lon);
      if (id >= 0) { var v = vec(lat, lon, R); pos.push(v.x, v.y, v.z); iso.push(id); }
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(pos.length), 3));
    this.iso = iso;
    this.dots = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.014, vertexColors: true }));
    this.root.add(this.dots);
    this.borders = new THREE.Group(); this.root.add(this.borders);
  };

  P.paint = function () {
    if (!this.dots || !this.data) return;
    var ids = {}, me = this, c = this.dots.geometry.attributes.color, k = new THREE.Color();
    this.data.forEach(function (d) { ids[d.iso] = d.id; });
    this.iso.forEach(function (iso, i) { k.setHex(ids[iso] ? (ids[iso] === selected ? COL.sel : COL.active) : COL.land); c.setXYZ(i, k.r, k.g, k.b); });
    c.needsUpdate = true;
    this.clear(this.borders);
    this.data.forEach(function (d) {
      var on = d.id === selected, pts = [];
      (me.map.borders[d.iso] || []).forEach(function (rg) { for (var j = 1; j < rg.length; j++) { pts.push(vec(rg[j - 1][1], rg[j - 1][0], R * 1.002), vec(rg[j][1], rg[j][0], R * 1.002)); } });
      if (pts.length) me.borders.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: on ? COL.sel : COL.active, transparent: true, opacity: on ? 1 : .55 })));
    });
  };

  P.clear = function (g) { while (g && g.children.length) { var o = g.children.pop(); o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); } };

  P.markers = function () {
    var me = this; this.pins = [];
    this.data.forEach(function (d) {
      var v = vec(d.lat, d.lon, R * 1.008), m = new THREE.Mesh(new THREE.SphereGeometry(0.013, 12, 12), new THREE.MeshBasicMaterial({ color: COL.sel }));
      m.position.copy(v); me.root.add(m);
      var lab = document.createElement("div"); lab.className = "globe-label"; lab.innerHTML = d.name + "<b>" + d.obs + " %</b>"; me.labels.appendChild(lab);
      me.pins.push({ d: d, v: v, m: m, lab: lab });
    });
  };

  P.focus = function (id, instant) {
    if (!this.data) return;
    var d = this.data.find(function (x) { return x.id === id; });
    if (!d) return;
    this.target = { x: Math.max(-.55, Math.min(.7, d.lat * RAD * .8)), y: -(d.lon + 90) * RAD - 0.1 };
    while (this.target.y - this.rot.y > Math.PI) this.target.y -= 2 * Math.PI;
    while (this.target.y - this.rot.y < -Math.PI) this.target.y += 2 * Math.PI;
    if (instant || reduced) { this.rot.x = this.target.x; this.rot.y = this.target.y; }
    this.hold = performance.now() + 5000;
    this.paint();
  };

  P.bind = function () {
    var me = this, cv = this.renderer.domElement;
    this.ro = new ResizeObserver(function () { me.resize(); }); this.ro.observe(this.el);
    this.io = new IntersectionObserver(function (e) { me.visible = e[0].isIntersecting; }); this.io.observe(this.el);
    cv.addEventListener("pointerdown", function (e) { me.drag = { x: e.clientX, y: e.clientY, rx: me.rot.x, ry: me.rot.y, moved: 0 }; me.target = null; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointermove", function (e) {
      if (me.drag) {
        var dx = e.clientX - me.drag.x, dy = e.clientY - me.drag.y; me.drag.moved = Math.max(me.drag.moved, Math.abs(dx) + Math.abs(dy));
        me.rot.y = me.drag.ry + dx * .005; me.rot.x = Math.max(-1.1, Math.min(1.1, me.drag.rx + dy * .004));
      } else { var h = me.pick(e); cv.style.cursor = h ? "pointer" : "grab"; cv.title = h ? h.name : ""; me.hover = h && h.id; }
    });
    function up(e) {
      if (!me.drag) return;
      var click = me.drag.moved < 6; me.drag = null; me.hold = performance.now() + 2500;
      if (click) { var h = me.pick(e); if (h && window.dash_clientside && dash_clientside.set_props) dash_clientside.set_props("pais", { value: h.id }); }
    }
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", function () { me.drag = null; });
  };

  P.pick = function (e) {
    if (!this.data) return null;
    var r = this.renderer.domElement.getBoundingClientRect(), ray = new THREE.Raycaster();
    ray.setFromCamera({ x: (e.clientX - r.left) / r.width * 2 - 1, y: -(e.clientY - r.top) / r.height * 2 + 1 }, this.cam);
    var hit = ray.ray.intersectSphere(new THREE.Sphere(this.root.position, R), new THREE.Vector3());
    if (!hit) return null;
    var p = this.root.worldToLocal(hit.clone()).normalize();
    var near = this.data.find(function (d) { return vec(d.lat, d.lon, 1).distanceTo(p) < .06; });
    if (near) return near;
    if (!this.map) return null;
    var lat = 90 - Math.acos(p.y) / RAD, lon = Math.atan2(p.z, -p.x) / RAD - 180;
    if (lon < -180) lon += 360;
    var iso = this.map.at(lat, lon);
    return this.data.find(function (d) { return d.iso === iso; }) || null;
  };

  P.resize = function () {
    var w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h); this.cam.aspect = w / h;
    this.cam.position.z = w / h < 1 ? 4.3 / Math.max(.55, w / h) : 4.3; this.cam.updateProjectionMatrix();
  };

  P.loop = function (t) {
    if (this.dead) return;
    requestAnimationFrame(this.loop);
    if (!this.visible || document.hidden) return;
    var dt = Math.min(50, t - this.last); this.last = t;
    if (!this.drag) {
      if (this.target) {
        this.rot.x += (this.target.x - this.rot.x) * Math.min(1, dt * .003); this.rot.y += (this.target.y - this.rot.y) * Math.min(1, dt * .003);
        if (performance.now() > this.hold) this.target = null;
      } else if (!reduced && performance.now() > (this.hold || 0)) this.rot.y += this.spin * dt * .06;
    }
    this.root.rotation.set(this.rot.x, this.rot.y, 0);
    var me = this;
    if (this.pins) {
      this.root.updateMatrixWorld();
      var w = this.el.clientWidth, h = this.el.clientHeight, cdir = this.cam.position.clone().normalize();
      this.pins.forEach(function (p) {
        var on = p.d.id === selected;
        var wp = p.v.clone().applyMatrix4(me.root.matrixWorld), front = wp.clone().normalize().dot(cdir) > .28;
        var show = front && (on || me.hover === p.d.id);
        p.lab.style.opacity = show ? 1 : 0; p.lab.className = "globe-label" + (on ? " sel" : "");
        if (show) { wp.project(me.cam); p.lab.style.left = (wp.x * .5 + .5) * w + "px"; p.lab.style.top = (-wp.y * .5 + .5) * h + "px"; }
      });
    }
    this.renderer.render(this.scene, this.cam);
  };

  P.destroy = function () {
    this.dead = true; this.ro && this.ro.disconnect(); this.io && this.io.disconnect();
    if (!this.renderer) return;
    this.scene.traverse(function (o) { o.geometry && o.geometry.dispose(); o.material && o.material.dispose && o.material.dispose(); });
    this.renderer.dispose(); this.renderer.forceContextLoss && this.renderer.forceContextLoss();
  };

  window.LIGlobe = {
    scan: function () {
      if (!window.THREE) return;
      live = live.filter(function (g) { if (!document.body.contains(g.el)) { g.destroy(); return false; } return true; });
      document.querySelectorAll(".li-globe").forEach(function (el) { if (!el.dataset.on) { el.dataset.on = "1"; live.push(new Globe(el)); } });
    },
    select: function (id) { if (id && id !== selected) { selected = id; live.forEach(function (g) { g.focus(id); }); } },
    /* Posición en pantalla del país en el globo de la portada (para la introducción). Gira el globo hacia él. */
    anchor: function (id) {
      var g = live.filter(function (x) { return !x.dead && x.pins && x.el.classList.contains("hero-globe"); })[0];
      if (!g) return null;
      g.focus(id, true);
      g.root.rotation.set(g.rot.x, g.rot.y, 0); g.root.updateMatrixWorld();
      var p = g.pins.filter(function (q) { return q.d.id === id; })[0];
      if (!p) return null;
      var wp = p.v.clone().applyMatrix4(g.root.matrixWorld).project(g.cam), r = g.el.getBoundingClientRect();
      return { x: r.left + (wp.x * .5 + .5) * r.width, y: r.top + (-wp.y * .5 + .5) * r.height };
    },
    live: function () { return live; }
  };
})();

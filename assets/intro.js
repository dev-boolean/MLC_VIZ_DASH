(function () {

  function init() {
    var splash  = document.getElementById('splash');
    var video   = document.getElementById('intro-video');
    var flying  = document.getElementById('flying-logo');
    var white   = document.getElementById('splash-white');
    var mainApp = document.getElementById('main-app');
    var target  = document.getElementById('logo-target');
    var skipBtn = document.getElementById('skip-intro');

    if (!splash || !mainApp || !target || !flying || !white) {
      setTimeout(init, 150);
      return;
    }

    target.style.opacity = '0';
    var transitioning = false;
    var done          = false;
    document.body.style.overflow = 'hidden';

    function getTargetPos() {
      var rect = target.getBoundingClientRect();
      var size = window.innerWidth < 860 ? 60 : 80;
      return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2, size: size };
    }

    function closeSplash() {
      splash.style.transition = 'opacity 0.4s ease';
      splash.style.opacity    = '0';
      setTimeout(function () {
        splash.style.display = 'none';
        document.body.style.overflow = '';
      }, 420);
    }

    function flyToTarget(instant) {
      var pos = getTargetPos();
      flying.style.opacity    = '1';
      flying.style.transition = instant ? 'none' : 'all 1.2s cubic-bezier(0.25, 0.8, 0.25, 1)';
      requestAnimationFrame(function () {
        flying.style.width     = pos.size + 'px';
        flying.style.left      = pos.x + 'px';
        flying.style.top       = pos.y + 'px';
        flying.style.transform = 'translate(-50%, -50%)';
      });
      mainApp.style.transition = 'opacity 0.8s ease 0.1s';
      mainApp.style.opacity    = '1';
      setTimeout(function () {
        target.style.opacity = '1';
        flying.style.opacity = '0';
        closeSplash();
      }, instant ? 60 : 1180);
    }

    function startTransition(instant) {
      if (done) return;
      done = true;
      transitioning = true;
      if (video) { video.pause(); video.style.opacity = '0'; }
      white.style.opacity     = '1';
      splash.style.background = '#ffffff';
      if (skipBtn) skipBtn.style.opacity = '0';
      flying.style.opacity = '1';
      if (instant) {
        flyToTarget(true);
      } else {
        setTimeout(function () { flyToTarget(false); }, 260);
      }
    }

    // ── Video ──────────────────────────────────────────────────
    if (video) {
      video.addEventListener('ended', function () { startTransition(false); });
      video.addEventListener('error', function () { startTransition(true); });

      var fallback = setTimeout(function () {
        if (!done) startTransition(false);
      }, 8000);

      var playPromise = video.play();
      if (playPromise !== undefined) {
        playPromise.catch(function () {
          clearTimeout(fallback);
          startTransition(true);
        });
      }
    } else {
      setTimeout(function () { startTransition(true); }, 400);
    }

    // ── Botón saltar ───────────────────────────────────────────
    if (skipBtn) {
      skipBtn.addEventListener('click', function () {
        clearTimeout(fallback);
        startTransition(true);
      });
      skipBtn.addEventListener('mouseover', function () {
        skipBtn.style.background = 'rgba(0,0,0,0.85)';
        skipBtn.style.transform  = 'translateY(-1px)';
      });
      skipBtn.addEventListener('mouseout', function () {
        skipBtn.style.background = 'rgba(0,0,0,0.60)';
        skipBtn.style.transform  = 'translateY(0)';
      });
    }

    // ── Resize ─────────────────────────────────────────────────
    window.addEventListener('resize', function () {
      if (transitioning && splash.style.display !== 'none') {
        var pos = getTargetPos();
        flying.style.left = pos.x + 'px';
        flying.style.top  = pos.y + 'px';
      }
    });
  }

  setTimeout(init, 300);

})();
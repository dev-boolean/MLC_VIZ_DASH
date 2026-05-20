(function () {

  function init() {
    var splash  = document.getElementById('splash');
    var video   = document.getElementById('intro-video');
    var flying  = document.getElementById('flying-logo');
    var white   = document.getElementById('splash-white');
    var mainApp = document.getElementById('main-app');
    var target  = document.getElementById('logo-target');
    var skipBtn = document.getElementById('skip-intro');

    // Si Dash aún no montó los elementos, reintentar
    if (!splash || !mainApp || !target || !flying || !white) {
      setTimeout(init, 150);
      return;
    }

    target.style.opacity = '0';
    var hasTransitioned = false;
    document.body.style.overflow = 'hidden';

    function getTargetPos() {
      var rect = target.getBoundingClientRect();
      var size = window.innerWidth < 860 ? 60 : 80;
      return {
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
        size: size
      };
    }

    function flyToTarget(instant) {
      if (hasTransitioned && !instant) return;
      hasTransitioned = true;

      var pos = getTargetPos();

      flying.style.opacity = '1';
      flying.style.transition = instant
        ? 'none'
        : 'all 1.2s cubic-bezier(0.25, 0.8, 0.25, 1)';

      requestAnimationFrame(function () {
        flying.style.width     = pos.size + 'px';
        flying.style.left      = pos.x + 'px';
        flying.style.top       = pos.y + 'px';
        flying.style.transform = 'translate(-50%, -50%)';
      });

      mainApp.style.transition = 'opacity 0.8s ease 0.1s';
      mainApp.style.opacity    = '1';

      setTimeout(function () {
        if (target) target.style.opacity = '1';
        flying.style.opacity = '0';
        splash.style.transition = 'opacity 0.4s ease';
        splash.style.opacity    = '0';
        setTimeout(function () {
          splash.style.display = 'none';
          document.body.style.overflow = '';
        }, 420);
      }, instant ? 60 : 1180);
    }

    function startTransition() {
      if (hasTransitioned) return;   // evitar doble llamada
      if (video) video.style.opacity = '0';
      white.style.opacity       = '1';
      splash.style.background   = '#ffffff';
      if (skipBtn) skipBtn.style.opacity = '0';
      flying.style.opacity = '1';
      setTimeout(function () { flyToTarget(false); }, 260);
    }

    // ── Video ────────────────────────────────────────────────────
    if (video) {
      video.addEventListener('ended', startTransition);
      video.addEventListener('error', startTransition);

      // Fallback: si el video tarda más de 7s en terminar, pasar igual
      var fallback = setTimeout(function () {
        if (!hasTransitioned) startTransition();
      }, 7000);

      var playPromise = video.play();
      if (playPromise !== undefined) {
        playPromise
          .then(function () {
            // Video corriendo OK — el evento 'ended' disparará startTransition
          })
          .catch(function () {
            // Autoplay bloqueado → saltar al dashboard de inmediato
            clearTimeout(fallback);
            startTransition();
            flyToTarget(true);
          });
      }
    } else {
      // No hay video en absoluto
      setTimeout(startTransition, 400);
    }

    // ── Botón saltar ─────────────────────────────────────────────
    if (skipBtn) {
      skipBtn.addEventListener('click', function () {
        if (video) video.pause();
        startTransition();
        flyToTarget(true);
      });
      skipBtn.addEventListener('mouseover', function () {
        skipBtn.style.background  = 'rgba(0,0,0,0.85)';
        skipBtn.style.transform   = 'translateY(-1px)';
      });
      skipBtn.addEventListener('mouseout', function () {
        skipBtn.style.background  = 'rgba(0,0,0,0.60)';
        skipBtn.style.transform   = 'translateY(0)';
      });
    }

    // ── Reajustar posición en resize ─────────────────────────────
    window.addEventListener('resize', function () {
      if (hasTransitioned && splash.style.display !== 'none') {
        var pos = getTargetPos();
        flying.style.left = pos.x + 'px';
        flying.style.top  = pos.y + 'px';
      }
    });
  }

  // Arrancar con polling hasta que Dash haya montado el DOM
  setTimeout(init, 300);

})();

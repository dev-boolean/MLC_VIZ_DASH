(function () {
  var messages = [];
  var preguntasUsadas = 0;
  var LIMITE = 20;
  var AVISO_EN = 16;
  var bloqueado = false;

  var AUTORES = [
    {
      nombre: "Andrés Parejo",
      github: "https://github.com/dev-boolean",
      linkedin: "https://www.linkedin.com/in/andres-parejo-40a653168/",
      email: "mailto:aeparejo@uninorte.edu.co",
      iniciales: "AP"
    },
    {
      nombre: "Santiago Hurtado",
      github: "https://github.com/SHurtado26",
      linkedin: "https://www.linkedin.com/in/santiago-hurtado-369687291/",
      email: "mailto:hurtadosantiago@uninorte.edu.co",
      iniciales: "SH"
    }
  ];

  var SUGGESTIONS = [
    "¿Cuál fue el mes con más desempleo?",
    "¿Cómo afectó el COVID al mercado laboral?",
    "¿Qué modelo SARIMA se usó y por qué?",
    "¿Qué diferencia hay entre TGD nacional y 13 áreas?",
    "¿Cómo interpreto la matriz de correlación?",
  ];

  function buildUI() {
    var style = document.createElement('style');
    style.textContent = `
      #mlc-chat-btn {
        position: fixed; bottom: 28px; right: 28px;
        width: 56px; height: 56px; border-radius: 50%;
        background: linear-gradient(135deg, #1e40af, #2563eb);
        border: none; cursor: pointer; z-index: 8000;
        box-shadow: 0 4px 20px rgba(37,99,235,0.45);
        display: flex; align-items: center; justify-content: center;
        transition: transform 0.2s, box-shadow 0.2s;
      }
      #mlc-chat-btn:hover { transform: scale(1.08); box-shadow: 0 6px 28px rgba(37,99,235,0.55); }
      #mlc-chat-btn svg { width: 24px; height: 24px; fill: white; }
      #mlc-chat-badge {
        position: absolute; top: -3px; right: -3px;
        background: #ef4444; color: white; font-size: 10px;
        font-weight: 700; border-radius: 50%; width: 18px; height: 18px;
        display: none; align-items: center; justify-content: center;
        font-family: Inter, sans-serif;
      }
      #mlc-chat-panel {
        position: fixed; bottom: 96px; right: 28px;
        width: 370px; max-width: calc(100vw - 40px);
        height: 540px; max-height: calc(100vh - 120px);
        background: #fff; border-radius: 20px;
        box-shadow: 0 20px 60px rgba(15,23,42,0.18), 0 0 0 1px rgba(0,0,0,0.06);
        display: none; flex-direction: column;
        z-index: 8000; overflow: hidden;
        animation: chatSlideIn 0.25s cubic-bezier(0.4,0,0.2,1);
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
      }
      @keyframes chatSlideIn {
        from { opacity: 0; transform: translateY(16px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
      }
      #mlc-chat-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
        padding: 14px 18px; display: flex; align-items: center; gap: 12px;
        flex-shrink: 0;
      }
      .mlc-chat-avatar {
        width: 36px; height: 36px; border-radius: 50%;
        background: rgba(255,255,255,0.15);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; flex-shrink: 0;
      }
      .mlc-chat-header-info { flex: 1; }
      .mlc-chat-header-name { color: white; font-weight: 700; font-size: 14px; line-height: 1.2; }
      .mlc-chat-header-sub  { color: rgba(255,255,255,0.65); font-size: 11px; margin-top: 2px; }
      #mlc-chat-close {
        background: none; border: none; color: rgba(255,255,255,0.7);
        cursor: pointer; font-size: 20px; padding: 4px; line-height: 1;
        transition: color 0.15s;
      }
      #mlc-chat-close:hover { color: white; }
      #mlc-counter-bar {
        display: flex; align-items: center; justify-content: flex-end;
        padding: 5px 14px; background: #f8fafc;
        border-bottom: 1px solid #f1f5f9; flex-shrink: 0; gap: 6px;
      }
      #mlc-counter-label {
        font-size: 11px; color: #94a3b8; font-family: inherit;
      }
      #mlc-counter-dots {
        display: flex; gap: 3px;
      }
      .mlc-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #e2e8f0; transition: background 0.3s;
      }
      .mlc-dot.used   { background: #2563eb; }
      .mlc-dot.warn   { background: #f59e0b; }
      .mlc-dot.danger { background: #ef4444; }
      #mlc-chat-messages {
        flex: 1; overflow-y: auto; padding: 14px 14px 8px;
        display: flex; flex-direction: column; gap: 10px;
        scroll-behavior: smooth;
      }
      #mlc-chat-messages::-webkit-scrollbar { width: 4px; }
      #mlc-chat-messages::-webkit-scrollbar-track { background: transparent; }
      #mlc-chat-messages::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
      .mlc-msg { max-width: 88%; font-size: 13.5px; line-height: 1.55; }
      .mlc-msg-user {
        align-self: flex-end;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white; padding: 9px 14px; border-radius: 18px 18px 4px 18px;
      }
      .mlc-msg-bot {
        align-self: flex-start;
        background: #f1f5f9; color: #1e293b;
        padding: 9px 14px; border-radius: 18px 18px 18px 4px;
        border: 1px solid #e2e8f0;
      }
      .mlc-msg-warn {
        align-self: stretch;
        background: #fffbeb; color: #92400e;
        padding: 9px 14px; border-radius: 12px;
        border: 1px solid #fde68a; font-size: 12.5px; text-align: center;
      }
      .mlc-msg-bot-typing {
        align-self: flex-start;
        background: #f1f5f9; padding: 12px 16px;
        border-radius: 18px 18px 18px 4px; border: 1px solid #e2e8f0;
      }
      .mlc-dots { display: flex; gap: 4px; }
      .mlc-dots span {
        width: 7px; height: 7px; border-radius: 50%;
        background: #94a3b8; animation: mlcDot 1.2s infinite;
      }
      .mlc-dots span:nth-child(2) { animation-delay: 0.2s; }
      .mlc-dots span:nth-child(3) { animation-delay: 0.4s; }
      @keyframes mlcDot {
        0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
        40%            { transform: scale(1);   opacity: 1; }
      }
      #mlc-suggestions {
        display: flex; flex-wrap: wrap; gap: 6px;
        padding: 0 14px 10px; flex-shrink: 0;
      }
      .mlc-suggestion {
        background: #eff6ff; color: #1d4ed8;
        border: 1px solid #bfdbfe; border-radius: 20px;
        padding: 5px 12px; font-size: 11.5px; cursor: pointer;
        transition: background 0.15s; white-space: nowrap; font-family: inherit;
      }
      .mlc-suggestion:hover { background: #dbeafe; }
      #mlc-chat-input-row {
        display: flex; gap: 8px; padding: 12px 14px;
        border-top: 1px solid #f1f5f9; flex-shrink: 0; background: #fff;
      }
      #mlc-chat-input {
        flex: 1; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 9px 13px; font-size: 13px; outline: none;
        font-family: inherit; color: #1e293b; transition: border-color 0.2s;
      }
      #mlc-chat-input:focus { border-color: #93c5fd; }
      #mlc-chat-input:disabled { background: #f8fafc; color: #94a3b8; cursor: not-allowed; }
      #mlc-chat-send {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        border: none; border-radius: 12px; color: white;
        cursor: pointer; padding: 0 14px; font-size: 18px;
        transition: opacity 0.2s; flex-shrink: 0;
      }
      #mlc-chat-send:disabled { opacity: 0.4; cursor: not-allowed; }
      #mlc-chat-send:hover:not(:disabled) { opacity: 0.88; }

      /* Card de contacto al agotar preguntas */
      .mlc-agotado-card {
        align-self: stretch;
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        border-radius: 16px; padding: 16px; margin: 4px 0;
      }
      .mlc-agotado-title {
        color: white; font-weight: 700; font-size: 13px;
        margin-bottom: 4px;
      }
      .mlc-agotado-sub {
        color: rgba(255,255,255,0.65); font-size: 11.5px;
        margin-bottom: 14px; line-height: 1.5;
      }
      .mlc-autor-row {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 10px;
      }
      .mlc-autor-avatar {
        width: 36px; height: 36px; border-radius: 50%;
        background: rgba(255,255,255,0.12);
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 12px; flex-shrink: 0;
        border: 1px solid rgba(255,255,255,0.2);
      }
      .mlc-autor-nombre {
        color: white; font-weight: 600; font-size: 13px; flex: 1;
      }
      .mlc-autor-links { display: flex; gap: 8px; }
      .mlc-autor-link {
        display: flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 8px;
        background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18);
        text-decoration: none; font-size: 14px; transition: background 0.15s;
      }
      .mlc-autor-link:hover { background: rgba(255,255,255,0.22); }
      .mlc-yt-btn {
        display: flex; align-items: center; gap: 6px;
        background: #dc2626; border-radius: 10px;
        padding: 8px 14px; text-decoration: none;
        color: white; font-size: 12px; font-weight: 600;
        margin-top: 4px; transition: background 0.15s; font-family: inherit;
        border: none; cursor: pointer; width: 100%; justify-content: center;
      }
      .mlc-yt-btn:hover { background: #b91c1c; }
    `;
    document.head.appendChild(style);

    // Botón flotante
    var btn = document.createElement('button');
    btn.id = 'mlc-chat-btn';
    btn.title = 'Pregúntale a los datos';
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
      </svg>
      <span id="mlc-chat-badge"></span>
    `;
    document.body.appendChild(btn);

    // Construir dots del contador (20 dots = 20 preguntas)
    var dotsHTML = '';
    for (var i = 0; i < LIMITE; i++) dotsHTML += '<div class="mlc-dot" id="mlc-dot-' + i + '"></div>';

    // Panel
    var panel = document.createElement('div');
    panel.id = 'mlc-chat-panel';
    panel.innerHTML = `
      <div id="mlc-chat-header">
        <div class="mlc-chat-avatar">📊</div>
        <div class="mlc-chat-header-info">
          <div class="mlc-chat-header-name">Asistente MLC</div>
          <div class="mlc-chat-header-sub">Datos: DANE · Banco de la República</div>
        </div>
        <button id="mlc-chat-close" title="Cerrar">✕</button>
      </div>
      <div id="mlc-counter-bar">
        <span id="mlc-counter-label">20 preguntas disponibles</span>
        <div id="mlc-counter-dots">${dotsHTML}</div>
      </div>
      <div id="mlc-chat-messages"></div>
      <div id="mlc-suggestions"></div>
      <div id="mlc-chat-input-row">
        <input id="mlc-chat-input" type="text" placeholder="Pregunta sobre el mercado laboral..." maxlength="300" />
        <button id="mlc-chat-send" title="Enviar">➤</button>
      </div>
    `;
    document.body.appendChild(panel);

    // Sugerencias
    var sugBox = panel.querySelector('#mlc-suggestions');
    SUGGESTIONS.forEach(function (s) {
      var chip = document.createElement('button');
      chip.className = 'mlc-suggestion';
      chip.textContent = s;
      chip.onclick = function () { sendMessage(s); };
      sugBox.appendChild(chip);
    });

    // Mensaje de bienvenida
    appendBot('¡Hola! Soy el asistente del Mercado Laboral Colombiano 🇨🇴. Tienes 20 preguntas disponibles. ¿En qué te ayudo?');

    // Eventos
    btn.addEventListener('click', togglePanel);
    panel.querySelector('#mlc-chat-close').addEventListener('click', closePanel);
    panel.querySelector('#mlc-chat-send').addEventListener('click', function () {
      var inp = document.getElementById('mlc-chat-input');
      if (inp && inp.value.trim() && !bloqueado) sendMessage(inp.value.trim());
    });
    document.getElementById('mlc-chat-input').addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey && !bloqueado) {
        e.preventDefault();
        if (this.value.trim()) sendMessage(this.value.trim());
      }
    });
  }

  function actualizarContador() {
    var restantes = LIMITE - preguntasUsadas;
    var label = document.getElementById('mlc-counter-label');
    if (label) {
      label.textContent = restantes + ' pregunta' + (restantes !== 1 ? 's' : '') + ' disponible' + (restantes !== 1 ? 's' : '');
      if (restantes <= 4)       label.style.color = '#ef4444';
      else if (restantes <= 8)  label.style.color = '#f59e0b';
      else                      label.style.color = '#94a3b8';
    }
    // Actualizar dots
    for (var i = 0; i < LIMITE; i++) {
      var dot = document.getElementById('mlc-dot-' + i);
      if (!dot) continue;
      dot.className = 'mlc-dot';
      if (i < preguntasUsadas) {
        if (preguntasUsadas >= LIMITE)           dot.classList.add('danger');
        else if (preguntasUsadas >= AVISO_EN)    dot.classList.add('warn');
        else                                      dot.classList.add('used');
      }
    }
  }

  function mostrarCardAgotado() {
    bloqueado = true;
    var msgs = document.getElementById('mlc-chat-messages');
    var inputRow = document.getElementById('mlc-chat-input-row');

    // Ocultar input
    if (inputRow) inputRow.style.display = 'none';

    var card = document.createElement('div');
    card.className = 'mlc-agotado-card';

    var autoresHTML = AUTORES.map(function (a) {
      return `
        <div class="mlc-autor-row">
          <div class="mlc-autor-avatar">${a.iniciales}</div>
          <div class="mlc-autor-nombre">${a.nombre}</div>
          <div class="mlc-autor-links">
            <a class="mlc-autor-link" href="${a.github}" target="_blank" title="GitHub">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
              </svg>
            </a>
            <a class="mlc-autor-link" href="${a.linkedin}" target="_blank" title="LinkedIn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
              </svg>
            </a>
            <a class="mlc-autor-link" href="${a.email}" title="Enviar correo">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white">
                <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>
              </svg>
            </a>
          </div>
        </div>
      `;
    }).join('');

    card.innerHTML = `
      <div class="mlc-agotado-title">🎯 Has usado todas tus preguntas</div>
      <div class="mlc-agotado-sub">Este es un proyecto académico con recursos limitados. Si tienes más preguntas, contáctate directamente con los autores del análisis.</div>
      ${autoresHTML}
      <a class="mlc-yt-btn" href="https://www.youtube.com/watch?v=2pBUO-to_KM" target="_blank">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="white"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
        Ver video del proyecto en YouTube
      </a>
    `;

    if (msgs) { msgs.appendChild(card); msgs.scrollTop = msgs.scrollHeight; }
  }

  function togglePanel() {
    var panel = document.getElementById('mlc-chat-panel');
    var badge = document.getElementById('mlc-chat-badge');
    var isOpen = panel.style.display === 'flex';
    panel.style.display = isOpen ? 'none' : 'flex';
    if (!isOpen) {
      badge.style.display = 'none';
      badge.textContent = '';
      if (!bloqueado) document.getElementById('mlc-chat-input').focus();
    }
  }

  function closePanel() {
    document.getElementById('mlc-chat-panel').style.display = 'none';
  }

  function appendUser(text) {
    var div = document.createElement('div');
    div.className = 'mlc-msg mlc-msg-user';
    div.textContent = text;
    document.getElementById('mlc-chat-messages').appendChild(div);
    scrollBottom();
  }

  function appendBot(text) {
    var div = document.createElement('div');
    div.className = 'mlc-msg mlc-msg-bot';
    div.textContent = text;
    document.getElementById('mlc-chat-messages').appendChild(div);
    scrollBottom();
    var panel = document.getElementById('mlc-chat-panel');
    if (panel.style.display !== 'flex') {
      var badge = document.getElementById('mlc-chat-badge');
      badge.style.display = 'flex';
      badge.textContent = (parseInt(badge.textContent) || 0) + 1;
    }
  }

  function appendWarn(text) {
    var div = document.createElement('div');
    div.className = 'mlc-msg mlc-msg-warn';
    div.textContent = text;
    document.getElementById('mlc-chat-messages').appendChild(div);
    scrollBottom();
  }

  function showTyping() {
    var div = document.createElement('div');
    div.className = 'mlc-msg-bot-typing';
    div.id = 'mlc-typing';
    div.innerHTML = '<div class="mlc-dots"><span></span><span></span><span></span></div>';
    document.getElementById('mlc-chat-messages').appendChild(div);
    scrollBottom();
  }

  function removeTyping() {
    var t = document.getElementById('mlc-typing');
    if (t) t.remove();
  }

  function scrollBottom() {
    var msgs = document.getElementById('mlc-chat-messages');
    if (msgs) msgs.scrollTop = msgs.scrollHeight;
  }

  function setLoading(on) {
    var send = document.getElementById('mlc-chat-send');
    var inp  = document.getElementById('mlc-chat-input');
    if (send) send.disabled = on;
    if (inp && !bloqueado) inp.disabled = on;
  }

  function sendMessage(text) {
    if (bloqueado) return;

    var inp = document.getElementById('mlc-chat-input');
    if (inp) inp.value = '';

    appendUser(text);
    messages.push({ role: 'user', content: text });

    var sug = document.getElementById('mlc-suggestions');
    if (sug) sug.style.display = 'none';

    showTyping();
    setLoading(true);

    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: messages }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        removeTyping();
        var respuesta = data.respuesta || data.error || 'Sin respuesta.';
        appendBot(respuesta);
        messages.push({ role: 'assistant', content: respuesta });
        if (messages.length > 20) messages = messages.slice(-20);

        // Actualizar contador
        preguntasUsadas++;
        actualizarContador();

        // Aviso cuando quedan pocas
        if (preguntasUsadas === AVISO_EN) {
          appendWarn('⚠️ Te quedan ' + (LIMITE - AVISO_EN) + ' preguntas en esta sesión.');
        }

        // Límite alcanzado
        if (preguntasUsadas >= LIMITE) {
          setTimeout(mostrarCardAgotado, 400);
        }
      })
      .catch(function () {
        removeTyping();
        appendBot('Error al conectar con el servidor. Intenta de nuevo.');
      })
      .finally(function () {
        setLoading(false);
        if (!bloqueado) {
          var i = document.getElementById('mlc-chat-input');
          if (i) i.focus();
        }
      });
  }

  function tryInit() {
    if (document.body) { buildUI(); actualizarContador(); }
    else { setTimeout(tryInit, 200); }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { setTimeout(tryInit, 400); });
  } else {
    setTimeout(tryInit, 400);
  }
})();
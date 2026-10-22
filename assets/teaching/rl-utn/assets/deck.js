/* =====================================================================
   RL-UTN deck engine — small, dependency-free.
   Keys:  → / Space / PgDn  next      ← / PgUp  previous
          Home / End  first/last      O  overview        N  speaker notes
          S  presenter window         B  blank screen    F  fullscreen
          T  hide TODO boxes          ?  help
   URL:   index.html#12 opens slide 12 (1-based).
   Print: browser "Print → Save as PDF" gives one page per slide,
          all fragments visible (use Chrome, margins "none", background graphics on).
   ===================================================================== */
(function () {
  const deckEl = document.querySelector('.deck');
  const slides = Array.from(deckEl.querySelectorAll(':scope > .slide'));
  const meta = {
    title: document.body.dataset.lecture || document.title,
    author: document.body.dataset.author || '',
  };
  const isPresenter = new URLSearchParams(location.search).has('presenter');
  const channel = 'BroadcastChannel' in window ? new BroadcastChannel('rl-utn-deck:' + location.pathname) : null;

  let cur = 0;         // slide index
  let frag = 0;        // number of visible fragments on current slide

  // ---------- footer, numbering ----------
  slides.forEach((s, i) => {
    const f = document.createElement('div');
    f.className = 'footer';
    f.innerHTML = `<span>${meta.author}${meta.author ? ' · ' : ''}${meta.title}</span><span>${i + 1} / ${slides.length}</span>`;
    s.appendChild(f);
  });
  const progress = document.createElement('div'); progress.className = 'progress'; document.body.appendChild(progress);
  const notesBox = document.createElement('div'); notesBox.className = 'notes-overlay'; document.body.appendChild(notesBox);
  const blank = document.createElement('div'); blank.className = 'blank'; document.body.appendChild(blank);
  const help = document.createElement('div'); help.className = 'help';
  help.innerHTML = `<b>Keys</b><br><kbd>→</kbd>/<kbd>Space</kbd> next · <kbd>←</kbd> back<br><kbd>O</kbd> overview · <kbd>N</kbd> notes · <kbd>S</kbd> presenter window<br><kbd>B</kbd> blank screen · <kbd>F</kbd> fullscreen · <kbd>T</kbd> hide TODOs<br>Print → Save as PDF for handouts`;
  document.body.appendChild(help);

  // group fragments sharing the same data-frag into one step
  function fragSteps(s) {
    const list = Array.from(s.querySelectorAll('.fragment'));
    let auto = 0; const steps = new Map();
    list.forEach(el => {
      const k = el.dataset.frag !== undefined ? +el.dataset.frag : ++auto;
      if (el.dataset.frag !== undefined) auto = Math.max(auto, k);
      if (!steps.has(k)) steps.set(k, []);
      steps.get(k).push(el);
    });
    return Array.from(steps.keys()).sort((a, b) => a - b).map(k => steps.get(k));
  }

  // ---------- scaling ----------
  function fit() {
    if (document.body.classList.contains('overview')) return;
    const sc = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    slides.forEach(s => s.style.transform = `scale(${sc})`);
  }

  // ---------- rendering ----------
  function render(broadcast = true) {
    slides.forEach((s, i) => s.classList.toggle('active', i === cur));
    const steps = fragSteps(slides[cur]);
    steps.forEach((group, i) => group.forEach(el => {
      el.classList.toggle('visible', i < frag);
      el.classList.toggle('past', i < frag - 1);
    }));
    progress.style.width = (100 * (cur + 1) / slides.length) + '%';
    const n = slides[cur].querySelector('aside.notes');
    notesBox.innerHTML = n ? n.innerHTML : '<span style="opacity:.5">No notes for this slide.</span>';
    if (location.hash !== '#' + (cur + 1)) history.replaceState(null, '', '#' + (cur + 1));
    document.dispatchEvent(new CustomEvent('slidechange', { detail: { index: cur, slide: slides[cur], fragment: frag } }));
    if (broadcast && channel && !isPresenter) channel.postMessage({ cur, frag });
  }
  function go(i, f = 0) { cur = Math.max(0, Math.min(slides.length - 1, i)); frag = f; render(); }
  function next() {
    const n = fragSteps(slides[cur]).length;
    if (frag < n) { frag++; render(); } else if (cur < slides.length - 1) go(cur + 1, 0);
  }
  function prev() {
    if (frag > 0) { frag--; render(); } else if (cur > 0) go(cur - 1, fragSteps(slides[cur - 1]).length);
  }

  // ---------- keyboard ----------
  document.addEventListener('keydown', e => {
    if (e.target.closest('input, textarea, select, [contenteditable]')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (['ArrowRight', 'ArrowDown', ' ', 'PageDown', 'Enter'].includes(k)) { e.preventDefault(); next(); }
    else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(k)) { e.preventDefault(); prev(); }
    else if (k === 'Home') go(0);
    else if (k === 'End') go(slides.length - 1);
    else if (k === 'o' || k === 'O' || k === 'Escape') toggleOverview(k === 'Escape' ? false : undefined);
    else if (k === 'n' || k === 'N') document.body.classList.toggle('show-notes');
    else if (k === 'b' || k === 'B' || k === '.') document.body.classList.toggle('blanked');
    else if (k === 't' || k === 'T') document.body.classList.toggle('hide-todo');
    else if (k === '?') document.body.classList.toggle('show-help');
    else if (k === 'f' || k === 'F') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
    else if (k === 's' || k === 'S') openPresenter();
  });
  // clicker / touch: click on empty slide area advances
  let touchX = null;
  document.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, { passive: true });
  document.addEventListener('touchend', e => {
    if (touchX === null) return; const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 50) (dx < 0 ? next : prev)(); touchX = null;
  });

  // ---------- overview ----------
  function toggleOverview(force) {
    const on = force !== undefined ? force : !document.body.classList.contains('overview');
    document.body.classList.toggle('overview', on);
    if (on) { slides.forEach(s => s.style.transform = ''); slides[cur].scrollIntoView({ block: 'center' }); }
    else fit();
  }
  slides.forEach((s, i) => s.addEventListener('click', () => {
    if (document.body.classList.contains('overview')) { toggleOverview(false); go(i); }
  }));

  // ---------- presenter window ----------
  function openPresenter() {
    window.open(location.pathname + '?presenter' + location.hash, 'presenter', 'width=1100,height=750');
  }
  if (isPresenter) {
    document.body.classList.add('presenter');
    const pv = document.createElement('div');
    pv.style.cssText = 'position:fixed;inset:0;background:#1b1d1a;color:#eee;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto 1fr;gap:16px;padding:20px;z-index:50;font-family:var(--font-text)';
    pv.innerHTML = `<div id="pv-head" style="grid-column:1/3;display:flex;justify-content:space-between;font-size:22px"><span id="pv-num"></span><span id="pv-clock"></span><span id="pv-timer" title="click to reset" style="cursor:pointer"></span></div>
      <div><div style="opacity:.6;font-size:14px">CURRENT</div><div id="pv-cur" style="font-size:26px;font-weight:700;margin:6px 0 14px"></div>
      <div style="opacity:.6;font-size:14px">NEXT</div><div id="pv-next" style="font-size:20px;opacity:.8;margin-top:6px"></div></div>
      <div id="pv-notes" style="font-size:21px;line-height:1.5;overflow:auto;background:#262924;padding:16px 20px;border-radius:8px"></div>`;
    document.body.appendChild(pv);
    let t0 = Date.now();
    pv.querySelector('#pv-timer').onclick = () => (t0 = Date.now());
    setInterval(() => {
      const d = Math.floor((Date.now() - t0) / 1000);
      pv.querySelector('#pv-timer').textContent = `⏱ ${Math.floor(d / 60)}:${String(d % 60).padStart(2, '0')}`;
      pv.querySelector('#pv-clock').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }, 1000);
    const title = s => (s.querySelector('h1, h2') || {}).textContent || '(untitled)';
    document.addEventListener('slidechange', e => {
      const i = e.detail.index;
      pv.querySelector('#pv-num').textContent = `Slide ${i + 1} / ${slides.length} · step ${e.detail.fragment}/${fragSteps(slides[i]).length}`;
      pv.querySelector('#pv-cur').textContent = title(slides[i]);
      pv.querySelector('#pv-next').textContent = slides[i + 1] ? title(slides[i + 1]) : '— end —';
      const n = slides[i].querySelector('aside.notes');
      pv.querySelector('#pv-notes').innerHTML = n ? n.innerHTML : '<i style="opacity:.5">No notes.</i>';
      if (window.renderMathInElement) renderMathInElement(pv.querySelector('#pv-notes'), window.KATEX_OPTS);
    });
    // keys in presenter window drive the main window
    if (channel) channel.onmessage = m => { cur = m.data.cur; frag = m.data.frag; render(false); };
    const relay = () => channel && channel.postMessage({ cur, frag, fromPresenter: true });
    document.addEventListener('keydown', () => setTimeout(relay, 0));
  } else if (channel) {
    channel.onmessage = m => { if (m.data.fromPresenter) { cur = m.data.cur; frag = m.data.frag; render(false); } };
  }

  // ---------- math ----------
  window.KATEX_OPTS = {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\[', right: '\\]', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\(', right: '\\)', display: false },
    ],
    macros: window.RL_MACROS || {},
    throwOnError: false,
    trust: ctx => ctx.command === '\\htmlClass',
  };
  function typeset(root) { if (window.renderMathInElement) renderMathInElement(root || document.body, window.KATEX_OPTS); }
  window.RLDeck = { go, next, prev, typeset, slides };

  // ---------- preload math fonts (hidden slides would otherwise print without them) ----------
  if (document.fonts) ['KaTeX_Main', 'KaTeX_Math', 'KaTeX_AMS', 'KaTeX_Size1', 'KaTeX_Size2', 'KaTeX_Size3', 'KaTeX_Size4', 'KaTeX_Caligraphic', 'KaTeX_Fraktur', 'KaTeX_SansSerif', 'KaTeX_Script', 'KaTeX_Typewriter']
    .forEach(f => ['normal 1em', 'italic 1em', 'bold 1em', 'italic bold 1em'].forEach(st => document.fonts.load(`${st} ${f}`).catch(() => {})));

  // ---------- start ----------
  typeset(document.body);
  const h = parseInt(location.hash.slice(1), 10);
  window.addEventListener('resize', fit);
  window.addEventListener('hashchange', () => { const n = parseInt(location.hash.slice(1), 10); if (n && n - 1 !== cur) go(n - 1); });
  fit();
  go(h ? h - 1 : 0);
})();

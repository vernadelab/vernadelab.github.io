/* =====================================================================
   Model-free prediction in the two-exit maze: Monte Carlo, TD(0), TD(λ).
   Needs maze.js. The agent never reads P or r: it only sees sampled episodes.

   1) Step-by-step learner (one or several algorithms fed the SAME episodes)
      <div class="learner"
           data-algo="mc"            mc | td | tdl, or a comma list ("mc,td") for side-by-side
           data-alpha="mean"         MC: "mean" (1/N(s)) or a number; TD: a number, e.g. 0.5
           data-lambda="0.8"         for tdl
           data-gamma="0.9" data-eps="0.2"   policy = ε-greedy w.r.t. π* (ε=0 → π* itself)
           data-episodes="1 2 3 4 7 10 G | 1 2 g"   scripted episodes (state numbers, G/g = exits);
                                     after these, episodes are sampled from π (seeded RNG)
           data-seed="3"
           data-visit="first"        MC: first | every
           data-controls="step,episode,run,reset,truth"
           data-info="#log-id"       element where the update log is written (optional)
           data-size="360"></div>

   2) Learning-curve chart (RMS error vs episodes, averaged over runs)
      <div class="learn-chart" data-series="mc:mean; mc:0.05; td:0.1; td:0.3; tdl:0.1:0.8"
           data-episodes="100" data-runs="100" data-ymax="2"></div>
      or the λ-sweep: data-mode="lambda" data-series="0.05; 0.1; 0.2" (α values), data-episodes="20"
   ===================================================================== */
(function () {
  const M = window.RLMaze;
  const fmt = (x, d = 2) => (Math.abs(x) < 5e-13 ? 0 : x).toFixed(d);

  // ------------------------------------------------------------ RNG + environment sampling
  function rng(seed) {             // mulberry32
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function sampleEpisode(m, pi, rand, start = m.start, maxLen = 200) {
    const ep = []; let s = start;
    for (let t = 0; t < maxLen; t++) {
      let u = rand(), a = 0; while (a < 3 && u >= pi[s][a]) { u -= pi[s][a]; a++; }
      const outs = m.P[s][a]; let v = rand(), k = 0; while (k < outs.length - 1 && v >= outs[k].p) { v -= outs[k].p; k++; }
      const o = outs[k];
      ep.push({ s, a, r: o.reward, next: o.next, done: o.done, exit: o.exit });
      if (o.done) break; s = o.next;
    }
    return ep;
  }
  // "1 2 3 g" → transitions (actions inferred from the deterministic moves)
  function parseEpisode(m, str) {
    const toks = str.trim().split(/\s+/); const ep = [];
    for (let i = 0; i < toks.length - 1; i++) {
      const s = parseInt(toks[i], 10) - 1, nxt = toks[i + 1];
      const target = /^[gG]$/.test(nxt) ? nxt : parseInt(nxt, 10) - 1;
      const a = M.ACTIONS.findIndex((_, a) => { const o = m.P[s][a][0]; return typeof target === 'string' ? o.done && o.exit === target : !o.done && o.next === target; });
      if (a < 0) throw new Error('Impossible move in scripted episode: ' + str);
      const o = m.P[s][a][0];
      ep.push({ s, a, r: o.reward, next: o.next, done: o.done, exit: o.exit });
    }
    return ep;
  }

  // ------------------------------------------------------------ learners
  // each learner exposes run(ep): a generator yielding {kind: 'move'|'update', t, s, tex}
  const sname = s => `s_{${s + 1}}`;
  class MC {
    constructor(n, g, alpha = 'mean', visit = 'first') { this.n = n; this.g = g; this.alpha = alpha; this.visit = visit; this.reset(); }
    reset() { this.V = new Array(this.n).fill(0); this.N = new Array(this.n).fill(0); }
    get label() { return this.alpha === 'mean' ? 'MC (running mean)' : `MC (α = ${this.alpha})`; }
    *run(ep) {
      for (let t = 0; t < ep.length; t++) yield { kind: 'move', t };
      let G = 0; const firstT = {};
      ep.forEach((e, t) => { if (!(e.s in firstT)) firstT[e.s] = t; });
      for (let t = ep.length - 1; t >= 0; t--) {
        const e = ep[t], Gnext = G; G = e.r + this.g * G;
        const texG = `G_{${t}} = R_{${t + 1}} + \\gamma G_{${t + 1}} = ${e.r} + ${fmt(this.g)}\\cdot ${fmt(Gnext)} = ${fmt(G)}`;
        if (this.visit === 'first' && firstT[e.s] !== t) { yield { kind: 'update', t, s: e.s, tex: `${texG}\\quad\\text{(not first visit of } ${sname(e.s)}\\text{: skip)}` }; continue; }
        const old = this.V[e.s]; this.N[e.s]++;
        const a = this.alpha === 'mean' ? 1 / this.N[e.s] : this.alpha;
        this.V[e.s] = old + a * (G - old);
        const aTex = this.alpha === 'mean' ? `\\tfrac{1}{${this.N[e.s]}}` : fmt(a, 2);
        yield { kind: 'update', t, s: e.s, tex: `${texG}\\\\ V(${sname(e.s)}) \\leftarrow ${fmt(old)} + ${aTex}\\,(${fmt(G)} - ${fmt(old)}) = ${fmt(this.V[e.s])}` };
      }
    }
  }
  class TD {
    constructor(n, g, alpha = 0.5) { this.n = n; this.g = g; this.alpha = +alpha; this.reset(); }
    reset() { this.V = new Array(this.n).fill(0); this.N = new Array(this.n).fill(0); }
    get label() { return `TD(0) (α = ${this.alpha})`; }
    *run(ep) {
      for (let t = 0; t < ep.length; t++) {
        yield { kind: 'move', t };
        const e = ep[t], old = this.V[e.s], vn = e.done ? 0 : this.V[e.next];
        const target = e.r + this.g * vn, delta = target - old;
        this.V[e.s] = old + this.alpha * delta; this.N[e.s]++;
        const nxt = e.done ? '0\\;(\\text{exit})' : `${fmt(this.g)}\\cdot ${fmt(vn)}`;
        yield { kind: 'update', t, s: e.s, tex: `V(${sname(e.s)}) \\leftarrow ${fmt(old)} + ${fmt(this.alpha)}\\,\\big(\\underbrace{${e.r} + ${nxt}}_{\\text{TD target}} - ${fmt(old)}\\big) = ${fmt(this.V[e.s])}` };
      }
    }
  }
  // backward-view TD(λ), accumulating traces (online)
  class TDL {
    constructor(n, g, alpha = 0.1, lambda = 0.8) { this.n = n; this.g = g; this.alpha = +alpha; this.lambda = +lambda; this.reset(); }
    reset() { this.V = new Array(this.n).fill(0); this.N = new Array(this.n).fill(0); this.e = new Array(this.n).fill(0); }
    get label() { return `TD(λ = ${this.lambda}) (α = ${this.alpha})`; }
    *run(ep) {
      this.e.fill(0);
      for (let t = 0; t < ep.length; t++) {
        yield { kind: 'move', t };
        const x = ep[t], vn = x.done ? 0 : this.V[x.next];
        const delta = x.r + this.g * vn - this.V[x.s];
        for (let i = 0; i < this.n; i++) this.e[i] *= this.g * this.lambda;
        this.e[x.s] += 1; this.N[x.s]++;
        for (let i = 0; i < this.n; i++) if (this.e[i]) this.V[i] += this.alpha * delta * this.e[i];
        const traced = this.e.map((v, i) => [i, v]).filter(([, v]) => v > 1e-3).map(([i, v]) => `e(${sname(i)})=${fmt(v)}`).join(',\\ ');
        yield { kind: 'update', t, s: x.s, tex: `\\delta_{${t}} = ${fmt(delta)};\\quad ${traced}` };
      }
    }
  }
  function makeLearner(spec, n, g, opts = {}) {
    const [algo, p1, p2] = spec.split(':').map(x => x.trim());
    if (algo === 'mc') return new MC(n, g, p1 && p1 !== 'mean' ? +p1 : 'mean', opts.visit || 'first');
    if (algo === 'td') return new TD(n, g, p1 !== undefined ? +p1 : 0.5);
    if (algo === 'tdl') return new TDL(n, g, p1 !== undefined ? +p1 : 0.1, p2 !== undefined ? +p2 : 0.8);
    throw new Error('unknown learner ' + spec);
  }
  // on-policy state distribution μ (expected visits per episode, normalised): errors are weighted by μ,
  // so states the policy (almost) never visits do not dominate the error
  function visits(m, pi) {
    let d = new Array(m.n).fill(0); const tot = new Array(m.n).fill(0); d[m.start] = 1;
    for (let t = 0; t < 1000; t++) {
      const nd = new Array(m.n).fill(0);
      d.forEach((p, s) => { if (!p) return; tot[s] += p; pi[s].forEach((pa, a) => m.P[s][a].forEach(o => { if (!o.done) nd[o.next] += p * pa * o.p; })); });
      d = nd;
    }
    const Z = tot.reduce((a, b) => a + b); return tot.map(x => x / Z);
  }
  // the environment a widget works in: maze, slipperiness, fixed policy π, its true value V^π, visit weights μ
  function envFrom(ds) {
    const maze = ds.maze || 'two-exit', D = M.MAZES[maze].defaults || {};
    const g = parseFloat(ds.gamma || D.gamma || '0.9'), eps = parseFloat(ds.eps || D.eps || '0.2');
    const slip = parseFloat(ds.slip !== undefined ? ds.slip : (D.slip || 0));
    const m = M.buildModel(slip, maze), pi = M.namedPolicy(m, ds.policy || D.policy || 'egreedy', g, eps);
    const truth = M.evaluate(m, g, pi);
    return { maze, D, g, eps, slip, m, pi, truth, mu: visits(m, pi) };
  }
  // batch methods: replay a fixed set of episodes until convergence
  function batchMC(m, g, eps) {       // average of first-visit returns
    const sum = new Array(m.n).fill(0), cnt = new Array(m.n).fill(0);
    for (const ep of eps) {
      let G = 0; const Gs = []; for (let t = ep.length - 1; t >= 0; t--) { G = ep[t].r + g * G; Gs[t] = G; }
      const seen = new Set(); ep.forEach((e, t) => { if (seen.has(e.s)) return; seen.add(e.s); sum[e.s] += Gs[t]; cnt[e.s]++; });
    }
    return { V: sum.map((x, i) => cnt[i] ? x / cnt[i] : 0), cnt };
  }
  function batchTD(m, g, eps, sweeps = 800) {   // fixed point of batch TD(0) = value of the MDP estimated from the data
    const T = Array.from({ length: m.n }, () => ({})), Rs = new Array(m.n).fill(0), N = new Array(m.n).fill(0);
    for (const ep of eps) for (const e of ep) { N[e.s]++; Rs[e.s] += e.r; if (!e.done) T[e.s][e.next] = (T[e.s][e.next] || 0) + 1; }
    const nxt = T.map(o => Object.entries(o).map(([k, c]) => [+k, c]));
    let V = new Array(m.n).fill(0);
    for (let k = 0; k < sweeps; k++) V = V.map((_, s) => N[s] ? (Rs[s] + g * nxt[s].reduce((a, [sp, c]) => a + c * V[sp], 0)) / N[s] : 0);
    return V;
  }
  const rms = (V, W, mu) => Math.sqrt(V.reduce((a, v, i) => a + (mu ? mu[i] : 1 / V.length) * (v - W[i]) ** 2, 0));

  // ------------------------------------------------------------ interactive learner widget
  class LearnerWidget {
    constructor(root) {
      const ds = root.dataset; this.root = root;
      this.env = envFrom(ds);
      Object.assign(this, { g: this.env.g, eps: this.env.eps, m: this.env.m, pi: this.env.pi, truth: this.env.truth, mu: this.env.mu });
      this.scripts = (ds.episodes || '').split('|').map(s => s.trim()).filter(Boolean).map(s => parseEpisode(this.m, s));
      this.seed = parseInt(ds.seed || '3', 10);
      const algos = (ds.algo || 'mc').split(',').map(a => a.trim());
      const single = algos.length === 1 ? ds.alpha : undefined;
      const aMc = ds.alphaMc || single || 'mean', aTd = ds.alphaTd || single || '0.5';
      this.specs = algos.map(a => a.includes(':') ? a : a === 'mc' ? `mc:${aMc}` : a === 'td' ? `td:${aTd}` : `tdl:${aTd}:${ds.lambda || 0.8}`);
      this.short = this.specs.map(s => ({ mc: 'MC', td: 'TD', tdl: 'TD(λ)' })[s.split(':')[0]]);
      this.visit = ds.visit || 'first';
      this.size = parseFloat(ds.size || (this.specs.length > 1 ? '300' : '360'));
      this.controls = (ds.controls || 'step,episode,run,reset,truth').split(',').map(s => s.trim());
      this.logEl = ds.info ? document.querySelector(ds.info) : null;
      this.build(); this.reset();
    }
    reset() {
      this.rand = rng(this.seed);
      this.learners = this.specs.map(s => makeLearner(s, this.m.n, this.g, { visit: this.visit }));
      this.k = 0; this.ep = null; this.gens = null; this.t = -1; this.updated = new Set(); this.showTruth = false; this.log = [];
      if (this.logEl) this.logEl.innerHTML = '<span class="muted">Press <b>Step</b> to sample the first transition.</span>';
      this.render();
    }
    nextEpisode() { const e = this.k < this.scripts.length ? this.scripts[this.k] : sampleEpisode(this.m, this.pi, this.rand); this.k++; return e; }
    startEpisode() {
      this.ep = this.nextEpisode(); this.t = -1; this.updated = new Set(); this.log = [];
      this.gens = this.learners.map(L => L.run(this.ep));
    }
    // one atomic event (a move or an update); all learners advance together
    step(render = true) {
      if (!this.ep || this.done) { this.startEpisode(); this.done = false; }
      let last = null;
      const results = this.gens.map(gen => gen.next());
      if (results.every(r => r.done)) { this.done = true; return this.step(render); }
      results.forEach((r, i) => {
        if (r.done) return; const ev = r.value; last = ev;
        if (ev.kind === 'move') this.t = ev.t;
        if (ev.kind === 'update') { this.updated.add(ev.s); if (render) this.log.push((this.learners.length > 1 ? `\\textbf{${this.short[i]}}\\;` : '') + ev.tex); }
      });
      // peek whether the episode is over
      if (render) this.render(last);
      return last;
    }
    finishEpisode(render = true) {
      if (!this.ep || this.done) { this.startEpisode(); this.done = false; }
      this.gens.forEach(gen => { for (const ev of gen) { if (ev.kind === 'update') this.updated.add(ev.s); if (ev.kind === 'move') this.t = ev.t; } });
      this.done = true;
      if (render) { this.log = ['\\text{episode } ' + this.k + '\\text{ finished: } ' + this.epTex()]; this.render(); }
    }
    run(n) { for (let i = 0; i < n; i++) { this.ep = null; this.finishEpisode(false); } this.render(); if (this.logEl) this.logEl.innerHTML = `<span class="muted">${this.k} episodes so far.</span>`; this.log = []; }
    epTex() {
      const end = this.ep[this.ep.length - 1];
      const G = this.ep.reduceRight((acc, e) => e.r + this.g * acc, 0);
      if (this.m.n > 20) return `T = ${this.ep.length}\\text{ steps, exit } ${end.exit === 'G' ? '+5' : end.exit === 'g' ? '+1' : '–'},\\; G_0 = ${fmt(G)}`;
      return this.ep.map(e => sname(e.s)).join('\\to ') + '\\to' + (end.exit === 'G' ? '\\text{big}' : '\\text{small}') + ` \\;(T=${this.ep.length})`;
    }
    build() {
      const r = this.root; r.innerHTML = ''; r.classList.add('learner-ready');
      this.grids = document.createElement('div'); this.grids.className = 'learner-grids'; r.appendChild(this.grids);
      this.mazes = this.specs.map((spec, i) => {
        const wrap = document.createElement('figure'); this.grids.appendChild(wrap);
        const host = document.createElement('div'); host.dataset.values = 'iterate'; host.dataset.size = this.size; host.dataset.controls = '';
        host.dataset.maze = this.env.maze; host.dataset.slip = this.env.slip; host.dataset.policy = 'none';
        if (this.root.dataset.numbers) host.dataset.numbers = this.root.dataset.numbers;
        wrap.appendChild(host);
        const mz = new M.Maze(host);
        if (this.specs.length > 1) { const cap = document.createElement('figcaption'); cap.className = 'learner-cap'; wrap.appendChild(cap); mz.cap = cap; }
        return mz;
      });
      const bar = document.createElement('div'); bar.className = 'maze-controls'; r.appendChild(bar);
      const btn = (label, fn, title) => { const b = document.createElement('button'); b.textContent = label; b.title = title || ''; b.onclick = e => { e.stopPropagation(); fn(); }; bar.appendChild(b); return b; };
      if (this.controls.includes('step')) btn('Step', () => this.step(), 'one transition / one update');
      if (this.controls.includes('episode')) btn('Episode', () => this.finishEpisode(), 'finish the current episode');
      if (this.controls.includes('run')) { btn('+10', () => this.run(10)); btn('+100', () => this.run(100)); }
      if (this.controls.includes('reset')) btn('Reset', () => this.reset());
      if (this.controls.includes('truth')) this.truthBtn = btn('', () => { this.showTruth = !this.showTruth; this.render(); });
      this.status = document.createElement('div'); this.status.className = 'maze-status'; r.appendChild(this.status);
    }
    render(lastEv) {
      const cur = this.t;
      this.mazes.forEach((mz, i) => {
        const L = this.learners[i];
        mz.V = this.showTruth ? this.truth.slice() : L.V.slice();
        mz.prevV = null;
        mz.overlay = (svg, cs) => this.drawOverlay(svg, cs, mz, L, lastEv);
        mz.draw();
        if (mz.cap) mz.cap.innerHTML = (this.showTruth ? 'true $V^\\pi$' : L.label) + (this.showTruth ? '' : ` · RMS error ${fmt(rms(L.V, this.truth, this.mu), 2)}`);
      });
      if (this.truthBtn) this.truthBtn.innerHTML = this.showTruth ? 'Show estimate' : 'Show true V<sup>π</sup>';
      const L0 = this.learners[0];
      this.status.innerHTML = `episode ${this.k}` + (this.ep && cur >= 0 ? ` · $t = ${cur}$` : '') +
        (this.specs.length === 1 ? ` · RMS error vs. $V^\\pi$: ${fmt(rms(L0.V, this.truth, this.mu), 2)}` : '') + (this.showTruth ? ' · <b>showing true values</b>' : '');
      if (this.logEl && this.log.length) this.logEl.innerHTML = this.log.slice(-(+(this.root.dataset.logLines || 4))).map(t => `$$${t}$$`).join('');
      if (window.RLDeck) { this.mazes.forEach(mz => mz.cap && RLDeck.typeset(mz.cap)); RLDeck.typeset(this.status); if (this.logEl) RLDeck.typeset(this.logEl); }
    }
    drawOverlay(svg, cs, mz, L, lastEv) {
      const NS = 'http://www.w3.org/2000/svg';
      const el = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const center = s => [this.m.states[s].c * cs + cs / 2, this.m.states[s].r * cs + cs / 2];
      // visit counts
      if (!this.showTruth && cs >= 60) L.N.forEach((n, s) => { if (!n) return; const [x, y] = center(s); el('text', { x: x + cs / 2 - 6, y: y + cs / 2 - 6, 'text-anchor': 'end', 'font-size': cs * 0.12, fill: '#8a9086' }).textContent = 'n=' + n; });
      // trajectory so far
      if (this.ep && this.t >= 0 && !this.showTruth) {
        const pts = []; const jitter = i => ((i * 37) % 7 - 3) * cs * 0.025;
        for (let t = 0; t <= this.t; t++) {
          const e = this.ep[t]; const [x, y] = center(e.s); pts.push([x + jitter(t), y + jitter(t + 3)]);
          if (t === this.t) {
            if (e.done) { const ex = this.m.states[e.s], d = M.ACTIONS[e.a].d; pts.push([ (ex.c + d[1]) * cs + cs / 2, (ex.r + d[0]) * cs + cs / 2]); }
            else { const [nx, ny] = center(e.next); pts.push([nx + jitter(t + 1), ny + jitter(t + 4)]); }
          }
        }
        el('polyline', { points: pts.map(p => p.join(',')).join(' '), fill: 'none', stroke: '#b5541c', 'stroke-width': cs < 40 ? 2.5 : 4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: 0.75 });
        const [hx, hy] = pts[pts.length - 1];
        el('circle', { cx: hx, cy: hy, r: cs * 0.09, fill: '#b5541c', stroke: '#fff', 'stroke-width': 2 });
      }
      // highlight the state just updated
      if (lastEv && lastEv.kind === 'update' && !this.showTruth) {
        const st = this.m.states[lastEv.s];
        el('rect', { x: st.c * cs + 3, y: st.r * cs + 3, width: cs - 6, height: cs - 6, fill: 'none', stroke: '#2b5b84', 'stroke-width': 5 });
      }
    }
  }

  // ------------------------------------------------------------ learning-curve chart
  const PALETTE = ['#2f5a24', '#b5541c', '#2b5b84', '#8a6a00', '#7a3e8e', '#6f9a3c'];
  function curves(series, E, R, env) {
    const { m, pi, truth, mu, g } = env;
    return series.map(spec => {
      const err = new Array(E + 1).fill(0);
      for (let r = 0; r < R; r++) {
        const rand = rng(1000 + r), L = makeLearner(spec, m.n, g);
        err[0] += rms(L.V, truth, mu);
        for (let k = 1; k <= E; k++) { const ep = sampleEpisode(m, pi, rand); for (const _ of L.run(ep)); err[k] += rms(L.V, truth, mu); }
      }
      return { spec, label: makeLearner(spec, m.n, g).label, y: err.map(e => e / R) };
    });
  }
  function lambdaSweep(alphas, lambdas, E, R, env) {
    const { m, pi, truth, mu, g } = env;
    return alphas.map(a => ({ label: `α = ${a}`, y: lambdas.map(lam => {
      let tot = 0;
      for (let r = 0; r < R; r++) { const rand = rng(1000 + r), L = new TDL(m.n, g, a, lam); for (let k = 1; k <= E; k++) { for (const _ of L.run(sampleEpisode(m, pi, rand))); tot += rms(L.V, truth, mu); } }
      return tot / (R * E);
    }) }));
  }
  function drawChart(root, xs, lines, opts) {
    const W = +(root.dataset.width || 620), H = +(root.dataset.height || 340), pad = { l: 58, r: 16, t: 14, b: 46 };
    const ymax = opts.ymax || Math.max(...lines.flatMap(l => l.y)) * 1.05;
    const X = x => pad.l + (x - xs[0]) / (xs[xs.length - 1] - xs[0]) * (W - pad.l - pad.r);
    const Y = y => pad.t + (1 - Math.min(y, ymax) / ymax) * (H - pad.t - pad.b);
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="learn-svg">`;
    for (let i = 0; i <= 4; i++) { const y = ymax * i / 4; svg += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="#e4e4dc"/><text x="${pad.l - 8}" y="${Y(y) + 5}" text-anchor="end" font-size="14" fill="#8a9086">${y.toFixed(2)}</text>`; }
    const ticks = opts.xticks || xs;
    ticks.forEach(x => svg += `<text x="${X(x)}" y="${H - pad.b + 20}" text-anchor="middle" font-size="14" fill="#8a9086">${x}</text>`);
    svg += `<text x="${(W + pad.l) / 2}" y="${H - 6}" text-anchor="middle" font-size="15" fill="#4a4f47">${opts.xlabel}</text>`;
    const sup = t => t.replace(/V\^π/g, 'V<tspan baseline-shift="super" font-size="11">π</tspan>');
    svg += `<text transform="translate(16 ${(H - pad.b + pad.t) / 2}) rotate(-90)" text-anchor="middle" font-size="15" fill="#4a4f47">${sup(opts.ylabel)}</text>`;
    lines.forEach((l, i) => {
      svg += `<polyline fill="none" stroke="${PALETTE[i % PALETTE.length]}" stroke-width="2.5" stroke-linejoin="round" ${l.dash ? 'stroke-dasharray="7 5"' : ''} points="${l.y.map((y, j) => `${X(xs[j])},${Y(y)}`).join(' ')}"/>`;
      if (opts.markers) l.y.forEach((y, j) => svg += `<circle cx="${X(xs[j])}" cy="${Y(y)}" r="3.5" fill="${PALETTE[i % PALETTE.length]}"/>`);
    });
    svg += '</svg>';
    const legend = '<div class="learn-legend">' + lines.map((l, i) => `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${l.label}</span>`).join('') + '</div>';
    root.innerHTML = svg + legend;
  }
  // batch comparison: K episodes, replayed to convergence by batch MC and batch TD; error on the visited states
  function batchCurves(Ks, R, env) {
    const { m, pi, truth, mu, g } = env, out = { mc: [], td: [] };
    for (const K of Ks) {
      let emc = 0, etd = 0;
      for (let r = 0; r < R; r++) {
        const rand = rng(500 + r), eps = []; for (let k = 0; k < K; k++) eps.push(sampleEpisode(m, pi, rand));
        const mc = batchMC(m, g, eps), td = batchTD(m, g, eps);
        const w = mu.map((x, i) => mc.cnt[i] ? x : 0), Z = w.reduce((a, b) => a + b);
        emc += Math.sqrt(w.reduce((a, x, i) => a + x * (mc.V[i] - truth[i]) ** 2, 0) / Z);
        etd += Math.sqrt(w.reduce((a, x, i) => a + x * (td[i] - truth[i]) ** 2, 0) / Z);
      }
      out.mc.push(emc / R); out.td.push(etd / R);
    }
    return [{ label: 'batch MC (average of returns)', y: out.mc, dash: true }, { label: 'batch TD(0) (replay TD updates until convergence)', y: out.td }];
  }
  // "new start" experiment: 50 ordinary episodes, then ONE episode from a new cell. Error of the estimate at that cell.
  function newStart(cell, R, K, env) {
    const { m, pi, truth, g } = env, s0 = m.index[cell.join(',')];
    let emc = 0, etd = 0;
    for (let k = 0; k < R; k++) {
      const rand = rng(700 + k), eps = []; for (let i = 0; i < K; i++) eps.push(sampleEpisode(m, pi, rand));
      const ne = sampleEpisode(m, pi, rand, s0); eps.push(ne);
      let G = 0; for (let t = ne.length - 1; t >= 0; t--) G = ne[t].r + g * G;
      emc += Math.abs(G - truth[s0]); etd += Math.abs(batchTD(m, g, eps)[s0] - truth[s0]);
    }
    return { mc: emc / R, td: etd / R, v: truth[s0] };
  }
  function drawBars(root, rows, unit) {
    const W = +(root.dataset.width || 560), rowH = 54, L = 210, H = rows.length * rowH + 30, vmax = Math.max(...rows.map(r => r.v)) * 1.15;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="learn-svg">`;
    rows.forEach((r, i) => {
      const y = i * rowH + 10, w = (W - L - 70) * r.v / vmax;
      svg += `<text x="${L - 12}" y="${y + 26}" text-anchor="end" font-size="18" fill="#1b1d1a">${r.label}</text>`;
      svg += `<rect x="${L}" y="${y + 6}" width="${w}" height="30" rx="4" fill="${r.color}"/>`;
      svg += `<text x="${L + w + 8}" y="${y + 27}" font-size="18" font-weight="700" fill="${r.color}">${r.v.toFixed(2)}</text>`;
    });
    svg += `<text x="${L}" y="${H - 4}" font-size="14" fill="#8a9086">${unit}</text></svg>`;
    root.innerHTML = svg;
  }
  // bias / variance of the estimate at one state: mean and sd over R independent runs, after each episode
  function bvCurves(series, sIdx, E, R, env) {
    const { m, pi, g } = env;
    return series.map(spec => {
      const sum = new Array(E + 1).fill(0), sq = new Array(E + 1).fill(0);
      for (let r = 0; r < R; r++) {
        const rand = rng(3000 + r), L = makeLearner(spec, m.n, g, { visit: 'every' });
        for (let k = 1; k <= E; k++) { for (const _ of L.run(sampleEpisode(m, pi, rand))); sum[k] += L.V[sIdx]; sq[k] += L.V[sIdx] ** 2; }
      }
      const mean = sum.map(x => x / R), sd = sq.map((x, i) => Math.sqrt(Math.max(0, x / R - mean[i] ** 2)));
      return { spec, label: makeLearner(spec, m.n, g).label, mean, sd };
    });
  }
  function drawBands(host, xs, lines, truth, opts) {
    const W = opts.width, H = opts.height, pad = { l: 46, r: 14, t: 26, b: 42 };
    const ymin = opts.ymin, ymax = opts.ymax;
    const X = x => pad.l + (x - xs[0]) / (xs[xs.length - 1] - xs[0]) * (W - pad.l - pad.r);
    const Y = y => pad.t + (1 - (Math.max(ymin, Math.min(ymax, y)) - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="learn-svg">`;
    svg += `<text x="${pad.l}" y="16" font-size="15" fill="#1b1d1a" font-weight="700">${opts.title}</text>`;
    for (let i = 0; i <= 4; i++) { const y = ymin + (ymax - ymin) * i / 4; svg += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="#e4e4dc"/><text x="${pad.l - 6}" y="${Y(y) + 5}" text-anchor="end" font-size="13" fill="#8a9086">${y.toFixed(opts.dec ?? 1)}</text>`; }
    (opts.xticks || []).forEach(x => svg += `<text x="${X(x)}" y="${H - pad.b + 18}" text-anchor="middle" font-size="13" fill="#8a9086">${x}</text>`);
    svg += `<text x="${(W + pad.l) / 2}" y="${H - 6}" text-anchor="middle" font-size="14" fill="#4a4f47">episodes</text>`;
    lines.forEach((l, i) => {
      const c = opts.colors[i];
      if (opts.band) {
        const up = l.mean.map((m, j) => `${X(xs[j])},${Y(m + l.sd[j])}`), lo = l.mean.map((m, j) => `${X(xs[j])},${Y(m - l.sd[j])}`).reverse();
        svg += `<polygon points="${up.concat(lo).join(' ')}" fill="${c}" opacity="0.18"/>`;
      }
      const ys = opts.band ? l.mean : l.sd;
      svg += `<polyline fill="none" stroke="${c}" stroke-width="2.5" stroke-linejoin="round" points="${ys.map((y, j) => `${X(xs[j])},${Y(y)}`).join(' ')}"/>`;
    });
    if (truth !== null) svg += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(truth)}" y2="${Y(truth)}" stroke="#1b1d1a" stroke-width="1.5" stroke-dasharray="6 4"/><text x="${W - pad.r}" y="${Y(truth) - 6}" text-anchor="end" font-size="13" fill="#1b1d1a">true value ${truth.toFixed(2)}</text>`;
    host.insertAdjacentHTML('beforeend', svg + '</svg>');
  }
  function initChart(root) {
    const ds = root.dataset, env = envFrom(ds);
    const E = parseInt(ds.episodes || '100', 10), R = parseInt(ds.runs || '100', 10);
    const series = (ds.series || '').split(';').map(s => s.trim()).filter(Boolean);
    root.innerHTML = '<span class="muted">computing…</span>';
    setTimeout(() => {
      if (ds.mode === 'lambda') {
        const lambdas = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1];
        const lines = lambdaSweep(series.map(Number), lambdas, E, R, env);
        drawChart(root, lambdas, lines, { xlabel: 'λ', ylabel: `RMS error (avg. over first ${E} episodes)`, markers: true, ymax: ds.ymax ? +ds.ymax : null });
      } else if (ds.mode === 'bv') {
        const sIdx = parseInt(ds.state || '2', 10) - 1;
        const lines = bvCurves(series, sIdx, E, R, env).map(l => ({ ...l, mean: l.mean.slice(1), sd: l.sd.slice(1) }));
        const xs = Array.from({ length: E }, (_, i) => i + 1), colors = ['#6f9a3c', '#b5541c', '#2b5b84'];
        root.innerHTML = '<div class="bv-row" style="display:flex; gap:18px"></div>';
        const row = root.firstChild, w = +(ds.width || 1100), h = +(ds.height || 300);
        const vmax = env.truth[sIdx];
        drawBands(row, xs, lines, env.truth[sIdx], { width: w * 0.55, height: h, ymin: 0, ymax: Math.ceil(vmax + 0.8), band: true, colors, title: `estimate of V(s<tspan baseline-shift="sub" font-size="11">${sIdx + 1}</tspan>): mean ± 1 sd over ${R} runs`, xticks: [1, 25, 50, 75, 100] });
        const sdmax = Math.max(...lines.flatMap(l => l.sd)) * 1.15;
        drawBands(row, xs, lines, null, { width: w * 0.43, height: h, ymin: 0, ymax: Math.ceil(sdmax * 10) / 10, band: false, colors, dec: 2, title: 'standard deviation of the estimate', xticks: [1, 25, 50, 75, 100] });
        root.insertAdjacentHTML('beforeend', '<div class="learn-legend">' + lines.map((l, i) => `<span><i style="background:${colors[i]}"></i>${l.label}</span>`).join('') + '</div>');
      } else if (ds.mode === 'newstart') {
        const res = newStart((ds.cell || '2,3').split(',').map(Number), R, +(ds.episodes || 50), env);
        drawBars(root, [{ label: 'MC: its one return', v: res.mc, color: '#2f5a24' }, { label: 'batch TD(0)', v: res.td, color: '#b5541c' }], `average error at the new cell, over ${R} runs`);
      } else if (ds.mode === 'batch') {
        const Ks = (ds.ks || '5,10,20,50,100').split(',').map(Number);
        drawChart(root, Ks, batchCurves(Ks, R, env), { xlabel: 'number of episodes in the batch', ylabel: 'RMS error vs. V^π', markers: true, ymax: ds.ymax ? +ds.ymax : null });
      } else {
        const lines = curves(series, E, R, env);
        lines.forEach(l => l.dash = l.spec.startsWith('mc'));
        const xs = Array.from({ length: E + 1 }, (_, i) => i);
        drawChart(root, xs, lines, { xlabel: 'episodes', ylabel: 'RMS error vs. V^π (visit-weighted)', xticks: [0, E / 4, E / 2, 3 * E / 4, E].map(Math.round), ymax: ds.ymax ? +ds.ymax : null });
      }
    }, 30);
  }

  // ------------------------------------------------------------ spread of the targets (bias / variance picture)
  // <div class="target-spread" data-state="1" data-samples="200" data-rows="mc,td,td0"></div>
  function initSpread(root) {
    const ds = root.dataset, env = envFrom(ds), { m, pi, truth, g } = env;
    const s0 = ds.state ? parseInt(ds.state, 10) - 1 : m.start, N = parseInt(ds.samples || '300', 10), rand = rng(parseInt(ds.seed || '11', 10));
    const rowsWanted = (ds.rows || 'mc,td,td0').split(',').map(x => x.trim());
    const samples = { mc: [], td: [], td0: [] };
    for (let i = 0; i < N; i++) {
      const ep = sampleEpisode(m, pi, rand, s0);
      let G = 0; for (let t = ep.length - 1; t >= 0; t--) G = ep[t].r + g * G;
      const e = ep[0];
      samples.mc.push(G); samples.td.push(e.r + (e.done ? 0 : g * truth[e.next])); samples.td0.push(e.r);
    }
    const label = { mc: `MC target $G_t$`, td: `TD target, $V = V^\\pi$`, td0: `TD target, $V = V_0 = 0$` };
    const all = rowsWanted.flatMap(k => samples[k]).concat([truth[s0]]);
    const lo = Math.floor(Math.min(...all)), hi = Math.ceil(Math.max(...all) + 1e-9);
    const W = +(ds.width || 1000), rowH = +(ds.rowHeight || 70), pad = { l: 250, r: 190, t: 10, b: 40 }, H = pad.t + pad.b + rowH * rowsWanted.length;
    const X = v => pad.l + (v - lo) / (hi - lo) * (W - pad.l - pad.r);
    const nb = Math.round((hi - lo) / (+(ds.bin || 0.1))), bw = (hi - lo) / nb;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="learn-svg">`;
    const step = hi - lo > 8 ? 2 : 1;
    for (let v = lo; v <= hi; v += step) svg += `<line x1="${X(v)}" x2="${X(v)}" y1="${pad.t}" y2="${H - pad.b}" stroke="#ecece4"/><text x="${X(v)}" y="${H - pad.b + 22}" text-anchor="middle" font-size="15" fill="#8a9086">${v}</text>`;
    const stats = [];
    rowsWanted.forEach((k, i) => {
      const yb = pad.t + (i + 1) * rowH - 8, a = samples[k];
      const mean = a.reduce((x, y) => x + y) / a.length, sd = Math.sqrt(a.reduce((x, y) => x + (y - mean) ** 2, 0) / a.length);
      const counts = new Array(nb).fill(0); a.forEach(v => counts[Math.min(nb - 1, Math.max(0, Math.floor((v - lo) / bw)))]++);
      const cmax = Math.max(...counts);
      svg += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${yb}" y2="${yb}" stroke="#c9ccc0"/>`;
      counts.forEach((c, b) => { if (!c) return; const h = (rowH - 16) * c / cmax; svg += `<rect x="${X(lo + b * bw) + 0.5}" y="${yb - h}" width="${Math.max(1, X(lo + bw) - X(lo) - 1)}" height="${h}" fill="${PALETTE[i]}" opacity="0.8"/>`; });
      svg += `<line x1="${X(mean)}" x2="${X(mean)}" y1="${yb - rowH + 12}" y2="${yb + 3}" stroke="#1b1d1a" stroke-width="3"/>`;
      stats.push({ k, mean, sd, y0: yb - rowH / 2 + 4 });
    });
    svg += `<line x1="${X(truth[s0])}" x2="${X(truth[s0])}" y1="${pad.t - 4}" y2="${H - pad.b + 4}" stroke="#2f5a24" stroke-width="2.5" stroke-dasharray="6 4"/>`;
    svg += `<text x="${X(truth[s0]) + 6}" y="${H - 6}" font-size="15" fill="#2f5a24" font-weight="700">true value ${truth[s0].toFixed(2)}</text>`;
    svg += '</svg>';
    root.style.position = 'relative';
    root.innerHTML = svg + stats.map(st => `<div style="position:absolute;left:0;top:${st.y0 - 14}px;width:${pad.l - 14}px;text-align:right;font-size:18px">${label[st.k]}</div>
      <div style="position:absolute;left:${W - pad.r + 14}px;top:${st.y0 - 14}px;font-size:17px;color:var(--ink-soft)">mean ${st.mean.toFixed(2)} · sd <b>${st.sd.toFixed(2)}</b></div>`).join('');
    if (window.RLDeck) RLDeck.typeset(root);
  }

  const css = `
  .learner { display: flex; flex-direction: column; gap: 10px; font-family: var(--font-text); }
  .learner-grids { display: flex; gap: 28px; }
  .learner-grids figure { margin: 0; }
  .learner-cap { font-size: 18px; color: var(--ink-soft); margin-top: 6px; }
  .learner .maze-status { font-size: 18px; }
  .learn-chart { font-family: var(--font-text); }
  .learn-svg text { font-family: var(--font-text); }
  .learn-legend { display: flex; flex-wrap: wrap; gap: 6px 20px; font-size: 17px; color: var(--ink-soft); margin-top: 4px; }
  .learn-legend i { display: inline-block; width: 22px; height: 4px; border-radius: 2px; vertical-align: middle; margin-right: 7px; }
  .learn-log .katex-display { margin: 6px 0; text-align: left; }
  .learn-log .katex-display > .katex { text-align: left; }
  .learn-log { font-size: 18px; }`;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);

  window.RLLearn = { bvCurves, newStart, envFrom, batchMC, batchTD, batchCurves, visits, rng, sampleEpisode, parseEpisode, MC, TD, TDL, makeLearner, curves, lambdaSweep, rms, widgets: [] };
  function init() {
    document.querySelectorAll('.learner').forEach(d => RLLearn.widgets.push(new LearnerWidget(d)));
    document.querySelectorAll('.learn-chart').forEach(initChart);
    document.querySelectorAll('.target-spread').forEach(initSpread);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();

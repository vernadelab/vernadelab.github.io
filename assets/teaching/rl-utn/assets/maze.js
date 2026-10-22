/* =====================================================================
   The two-exit maze — running example for the RL-UTN lectures.

   Layout (row 0 at the top):        Rewards (received on entering an exit,
       . . # G                       which ends the episode):
       . . # .                          G = big exit   (+5)
       . g . .                          g = small exit (+1)
       S . . .                       All other transitions: reward 0.
                                     Bumping into a wall/border: stay, reward 0.

   With these numbers, from the start S the big exit is worth it iff
   γ^5·5 > γ·1  ⇔  γ > 5^(-1/4) ≈ 0.669.

   Usage in a slide — a <div class="maze"> with data attributes:
     data-gamma="0.9"               discount factor
     data-policy="uniform"          none | uniform | optimal | small | big
     data-values="exact"            none | exact | blank | iterate
     data-algo="vi"                 for data-values="iterate": pe | vi | pi
     data-arrows="policy"           none | policy | greedy   (greedy w.r.t. displayed V)
     data-controls="step,reset"     any of: step, run, reset, gamma, reveal, improve, evaluate
     data-labels="ids"              ids | none   (state names s1…s12)
     data-click="backup"            click a cell → show its Bellman backup
     data-size="360"                grid size in px
     data-vmax="5"                  top of the colour scale
     data-slip="0"                  slippery variant: w.p. 1−slip the intended move, otherwise one of the
                                    3 other directions uniformly (slip=0.2 → 0.8 / 0.067 / 0.067 / 0.067)
     data-show="s6:right"          draw P(·|s,a) for one state-action pair on the grid
     data-controls="slip"          adds a deterministic ↔ slippery toggle (uses data-slip-value, default 0.2)
   ===================================================================== */
(function () {
  // Mazes. Cells: '.' free, '#' wall, 'S' start, exits (terminal, reward on entry),
  // tiles (non-terminal, reward each time the agent steps on them).
  const MAZES = {
    "two-exit": { layout: ["..#G", "..#.", ".g..", "S..."], exits: { G: 5, g: 1 }, tiles: {} },
    // a larger maze with rewards along the way: coins (+1) and traps (−1) each time you step on them.
    // Used with a GIVEN policy ("shortest route to the big exit", 20% random actions) and slippery moves.
    coins: {
      layout: [
        "c...#...xG",
        ".##.#.##..",
        "..c.#..c..",
        ".##...##x.",
        "....c.....",
        "x.###.#.c.",
        "..c...#...",
        ".#..x.###.",
        ".#.c......",
        "S..g.#c...",
      ],
      exits: { G: 5, g: 1 },
      tiles: { c: 1, x: -1 },
      defaults: { slip: 0.2, gamma: 0.95, eps: 0.2, policy: "route", vmin: 0, vmax: 5 },
    },
  };
  const LAYOUT = MAZES["two-exit"].layout;
  const EXIT_REWARD = MAZES["two-exit"].exits;
  const TILE_STYLE = { c: { fill: "#f3d97a", ink: "#5b4300", label: "coin" }, x: { fill: "#e9b8ae", ink: "#7a2410", label: "trap" } };
  const ACTIONS = [
    { name: "up", sym: "↑", tex: "\\uparrow", d: [-1, 0] },
    { name: "right", sym: "→", tex: "\\rightarrow", d: [0, 1] },
    { name: "down", sym: "↓", tex: "\\downarrow", d: [1, 0] },
    { name: "left", sym: "←", tex: "\\leftarrow", d: [0, -1] },
  ];
  const NA = 4;

  // ---------------------------------------------------------------- model
  function buildModel(slip = 0, which = "two-exit") {
    const spec = typeof which === "string" ? MAZES[which] : which;
    const L = spec.layout,
      exits = spec.exits,
      tiles = spec.tiles || {};
    const R = L.length,
      C = L[0].length;
    const cell = (r, c) => L[r][c];
    // states: non-wall, non-exit cells, numbered from the start, bottom row first, left→right
    const states = [];
    for (let r = R - 1; r >= 0; r--)
      for (let c = 0; c < C; c++) {
        const ch = cell(r, c);
        if (ch !== "#" && !(ch in exits)) states.push({ r, c, start: ch === "S", tile: ch in tiles ? ch : null });
      }
    const index = {};
    states.forEach((s, i) => {
      s.id = i;
      s.name = "s" + (i + 1);
      index[s.r + "," + s.c] = i;
    });
    function move(s, a) {
      const r = s.r + ACTIONS[a].d[0],
        c = s.c + ACTIONS[a].d[1];
      if (r < 0 || r >= R || c < 0 || c >= C || cell(r, c) === "#") return { next: s.id, reward: 0, done: false };
      const ch = cell(r, c);
      if (ch in exits) return { next: null, reward: exits[ch], done: true, exit: ch };
      return { next: index[r + "," + c], reward: tiles[ch] || 0, done: false };
    }
    // P[s][a] = list of {p, next, reward, done}
    const splitRandom = (o) => {
      // a random tile reward {r, p} becomes two outcomes with the same next state
      const t = typeof o.reward === "object" ? o.reward : null;
      if (!t) return [o];
      return [
        { ...o, p: o.p * t.p, reward: t.r },
        { ...o, p: o.p * (1 - t.p), reward: 0 },
      ].filter((x) => x.p > 0);
    };
    const P = states.map((s) =>
      ACTIONS.map((_, a) => {
        if (!slip) return splitRandom({ p: 1, dir: a, ...move(s, a) });
        // slippery: intended move w.p. 1−slip, otherwise one of the 3 other directions uniformly
        return ACTIONS.flatMap((_, b) => splitRandom({ p: b === a ? 1 - slip : slip / (NA - 1), dir: b, ...move(s, b) }));
      })
    );
    return {
      R,
      C,
      cell,
      states,
      index,
      P,
      n: states.length,
      slip,
      exits,
      tiles,
      name: typeof which === "string" ? which : "custom",
      start: states.findIndex((s) => s.start),
    };
  }

  // ---------------------------------------------------------------- arrays, as in code
  // states 0..n-1 = s1..s12, then n = big exit G, n+1 = small exit g (absorbing, reward 0)
  // P[a][s][s'] = P(s'|s,a)  (|A| matrices of size |S|x|S|),  r[s][a] = r(s,a) = E[R | s, a]
  function matrices(m) {
    const S = m.n + 2,
      idx = { G: m.n, g: m.n + 1 };
    const P = ACTIONS.map(() => Array.from({ length: S }, () => new Array(S).fill(0)));
    const r = Array.from({ length: S }, () => new Array(NA).fill(0));
    for (let s = 0; s < m.n; s++)
      ACTIONS.forEach((_, a) =>
        m.P[s][a].forEach((o) => {
          P[a][s][o.done ? idx[o.exit] : o.next] += o.p;
          r[s][a] += o.p * o.reward;
        })
      );
    ACTIONS.forEach((_, a) => {
      P[a][idx.G][idx.G] = 1;
      P[a][idx.g][idx.g] = 1;
    });
    const names = m.states.map((st) => st.name).concat(["G", "g"]);
    return { P, r, S, names };
  }

  // ---------------------------------------------------------------- algorithms
  const qValue = (m, V, g, s, a) => m.P[s][a].reduce((acc, t) => acc + t.p * (t.reward + (t.done ? 0 : g * V[t.next])), 0);
  const Q = (m, V, g) => m.states.map((_, s) => ACTIONS.map((_, a) => qValue(m, V, g, s, a)));
  const applyTpi = (m, V, g, pi) => m.states.map((_, s) => pi[s].reduce((acc, p, a) => acc + (p ? p * qValue(m, V, g, s, a) : 0), 0));
  const applyTstar = (m, V, g) => m.states.map((_, s) => Math.max(...ACTIONS.map((_, a) => qValue(m, V, g, s, a))));
  const sup = (U, V) => Math.max(...U.map((u, i) => Math.abs(u - V[i])));
  function evaluate(m, g, pi, tol = 1e-12) {
    let V = m.states.map(() => 0);
    for (let k = 0; k < 100000; k++) {
      const W = applyTpi(m, V, g, pi);
      if (sup(W, V) < tol) return W;
      V = W;
    }
    return V;
  }
  function optimal(m, g, tol = 1e-12) {
    let V = m.states.map(() => 0);
    for (let k = 0; k < 100000; k++) {
      const W = applyTstar(m, V, g);
      if (sup(W, V) < tol) return W;
      V = W;
    }
    return V;
  }
  // greedy policy; ties split uniformly (shown as several arrows)
  function greedy(m, V, g, eps = 1e-9) {
    return Q(m, V, g).map((q) => {
      const best = Math.max(...q);
      const arg = q.map((x) => (x >= best - eps ? 1 : 0));
      const n = arg.reduce((a, b) => a + b);
      return arg.map((x) => x / n);
    });
  }
  const uniform = (m) => m.states.map(() => [0.25, 0.25, 0.25, 0.25]);
  // shortest-path policy towards one exit (deterministic; first action in ↑→↓← order on ties)
  function towards(m, exit) {
    const dist = m.states.map(() => Infinity);
    let changed = true;
    while (changed) {
      changed = false;
      m.states.forEach((s, i) =>
        ACTIONS.forEach((_, a) => {
          const t = m.P[i][a].find((o) => o.dir === a);
          const d = t.done ? (t.exit === exit ? 1 : Infinity) : 1 + dist[t.next];
          if (d < dist[i]) {
            dist[i] = d;
            changed = true;
          }
        })
      );
    }
    return m.states.map((s, i) => {
      let best = -1,
        bd = Infinity;
      ACTIONS.forEach((_, a) => {
        const t = m.P[i][a].find((o) => o.dir === a);
        const d = t.done ? (t.exit === exit ? 1 : Infinity) : 1 + dist[t.next];
        if (d < bd) {
          bd = d;
          best = a;
        }
      });
      return ACTIONS.map((_, a) => (a === best ? 1 : 0));
    });
  }
  // deterministic greedy policy (ties broken in the order ↑ → ↓ ←)
  function greedyDet(m, V, g) {
    return Q(m, V, g).map((q) => {
      const b = q.indexOf(Math.max(...q));
      return q.map((_, a) => (a === b ? 1 : 0));
    });
  }
  // ε-greedy w.r.t. the (deterministic) optimal policy: follow π* w.p. 1−ε, else a uniformly random action
  function epsGreedy(m, g, eps) {
    return greedyDet(m, optimal(m, g), g).map((p) => p.map((x) => (1 - eps) * x + eps / NA));
  }
  // mix a deterministic policy with uniform random actions: w.p. 1−ε follow it, else a random action
  const epsMix = (pi, eps) => pi.map((p) => p.map((x) => (1 - eps) * x + eps / NA));
  function namedPolicy(m, name, g, eps = 0.2) {
    if (name === "route") return epsMix(towards(m, "G"), eps); // "shortest route to the big exit", with ε noise
    if (name === "uniform") return uniform(m);
    if (name === "egreedy") return epsGreedy(m, g, eps);
    if (name === "optimal-det") return greedyDet(m, optimal(m, g), g);
    if (name === "optimal") return greedy(m, optimal(m, g), g);
    if (name === "small") return towards(m, "g");
    if (name === "big") return towards(m, "G");
    return null;
  }

  // ---------------------------------------------------------------- drawing helpers
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  // sequential scale: warm paper → deep green
  const C0 = [247, 246, 239],
    C1 = [47, 90, 36];
  function heat(t) {
    t = Math.max(0, Math.min(1, t));
    t = Math.pow(t, 0.8);
    return `rgb(${C0.map((c, i) => Math.round(c + (C1[i] - c) * t)).join(",")})`;
  }
  const fmt = (x, d = 2) => (Math.abs(x) < 5e-13 ? 0 : x).toFixed(d);

  // ---------------------------------------------------------------- component
  class Maze {
    constructor(root) {
      this.root = root;
      this.uid = "mz" + (Maze.count = (Maze.count || 0) + 1);
      const ds = root.dataset;
      this.mazeName = ds.maze || "two-exit";
      const D = MAZES[this.mazeName].defaults || {};
      this.g = parseFloat(ds.gamma || D.gamma || "0.9");
      this.eps = parseFloat(ds.eps || D.eps || "0.2");
      this.slipValue = parseFloat(ds.slipValue || ds.slip || "0.2");
      this.m = buildModel(parseFloat(ds.slip !== undefined ? ds.slip : D.slip || 0), this.mazeName);
      this.showSA = ds.show ? ds.show.split(":") : null; // e.g. ['s6', 'right']
      // trajectories to draw, e.g. data-paths="1 2 3 6 7 10 G | 1 2 g" (state numbers; G/g = exits)
      this.paths = ds.paths ? ds.paths.split("|").map((p) => p.trim().split(/\s+/)) : null;
      this.pathColors = (ds.pathColors || "#2b5b84,#b5541c,#2f5a24").split(",");
      this.mark = ds.mark ? ds.mark.split(",").map(Number) : null; // a cell "r,c" to outline
      this.fixedV = ds.v ? ds.v.split(",").map(Number) : null; // values to display as given
      this.size = parseFloat(ds.size || "360");
      this.vmax = parseFloat(ds.vmax || D.vmax || "5");
      this.vmin = parseFloat(ds.vmin || D.vmin || "0");
      this.numbers = ds.numbers !== undefined ? ds.numbers !== "none" : null; // null: automatic (hidden on small cells)
      this.mode = ds.values || "exact";
      this.algo = ds.algo || "vi";
      this.arrows = ds.arrows || "none";
      this.labels = ds.labels || (this.m.C > 4 ? "none" : "ids");
      this.policyName = ds.policy || D.policy || (this.mode === "iterate" && this.algo !== "vi" ? "uniform" : "none");
      this.controls = (ds.controls !== undefined ? ds.controls : this.mode === "iterate" ? "step,run,reset" : "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      this.clickBackup = ds.click === "backup";
      this.digits = parseInt(ds.digits || "2", 10);
      this.revealed = new Set();
      this.selected = null;
      this.reset();
      this.build();
      this.draw();
    }
    reset() {
      this.k = 0;
      this.prevV = null;
      this.lastDelta = null;
      this.piStage = "evaluate";
      this.revealed.clear();
      this.pi = namedPolicy(this.m, this.policyName, this.g, this.eps) || uniform(this.m);
      if (this.mode === "iterate") this.V = this.m.states.map(() => 0);
      else if (this.mode === "fixed") this.V = this.fixedV.slice();
      else if (this.mode === "exact" || this.mode === "blank")
        this.V = this.policyName === "optimal" || this.policyName === "none" ? optimal(this.m, this.g) : evaluate(this.m, this.g, this.pi);
      else this.V = null;
      if (this.policyName === "optimal") this.pi = greedy(this.m, this.V, this.g);
    }
    step() {
      this.prevV = this.V.slice();
      if (this.algo === "vi") this.V = applyTstar(this.m, this.V, this.g);
      else if (this.algo === "pe") this.V = applyTpi(this.m, this.V, this.g, this.pi);
      else if (this.algo === "pi") {
        // policy iteration: alternate full evaluation and greedy improvement
        if (this.piStage === "evaluate") {
          this.V = evaluate(this.m, this.g, this.pi);
          this.piStage = "improve";
        } else {
          const old = JSON.stringify(this.pi);
          this.pi = greedy(this.m, this.V, this.g);
          this.piStage = "evaluate";
          this.stable = old === JSON.stringify(this.pi);
        }
      }
      this.lastDelta = sup(this.V, this.prevV);
      this.k++;
      this.draw();
    }
    build() {
      const r = this.root;
      r.innerHTML = "";
      r.classList.add("maze-ready");
      this.svg = el("svg", { viewBox: `0 0 ${this.size} ${this.size}`, width: this.size, height: this.size, class: "maze-svg" }, r);
      if (this.controls.length || (this.clickBackup && !this.root.dataset.info)) {
        this.side = document.createElement("div");
        this.side.className = "maze-side";
        r.appendChild(this.side);
      }
      if (this.controls.length) {
        const bar = document.createElement("div");
        bar.className = "maze-controls";
        this.side.appendChild(bar);
        const btn = (label, fn) => {
          const b = document.createElement("button");
          b.textContent = label;
          b.onclick = (e) => {
            e.stopPropagation();
            fn();
          };
          bar.appendChild(b);
          return b;
        };
        if (this.controls.includes("step")) this.stepBtn = btn(this.algo === "pi" ? "Evaluate" : "Step  k→k+1", () => this.step());
        if (this.controls.includes("run"))
          btn("Run ×10", () => {
            for (let i = 0; i < 10; i++) this.step();
          });
        if (this.controls.includes("reset"))
          btn("Reset", () => {
            this.reset();
            this.draw();
          });
        if (this.controls.includes("reveal"))
          btn("Reveal all", () => {
            this.m.states.forEach((_, i) => this.revealed.add(i));
            this.draw();
          });
        if (this.controls.includes("slip")) {
          const b = btn("", () => {
            this.m = buildModel(this.m.slip ? 0 : this.slipValue, this.mazeName);
            this.reset();
            this.draw();
            upd();
          });
          const upd = () => {
            b.textContent = this.m.slip
              ? `slippery (${Math.round(100 * (1 - this.m.slip))}% intended) — switch off`
              : "deterministic — make it slippery";
          };
          upd();
        }
        if (this.controls.includes("gamma")) {
          const wrap = document.createElement("label");
          wrap.className = "maze-gamma";
          wrap.innerHTML = `<span>γ = <b>${this.g.toFixed(2)}</b></span><input type="range" min="0.05" max="0.99" step="0.01" value="${this.g}">`;
          const inp = wrap.querySelector("input");
          inp.oninput = () => {
            this.g = parseFloat(inp.value);
            wrap.querySelector("b").textContent = this.g.toFixed(2);
            this.reset();
            this.draw();
          };
          inp.addEventListener("keydown", (e) => e.stopPropagation());
          bar.appendChild(wrap);
        }
        this.status = document.createElement("div");
        this.status.className = "maze-status";
        this.side.appendChild(this.status);
      }
      if (this.clickBackup) {
        const tgt = this.root.dataset.info && document.querySelector(this.root.dataset.info);
        this.info = tgt || document.createElement("div");
        this.info.classList.add("maze-info");
        this.info.innerHTML = '<span class="muted">Click a state to see its Bellman backup.</span>';
        if (!tgt) this.side.appendChild(this.info);
      }
    }
    draw() {
      const m = this.m,
        svg = this.svg,
        cs = this.size / m.C;
      svg.innerHTML = "";
      el("defs", {}, svg).innerHTML =
        `<pattern id="${this.uid}-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#3a3d37"/><line x1="0" y1="0" x2="0" y2="8" stroke="#55594f" stroke-width="3"/></pattern>
         <marker id="${this.uid}-ah-o" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#b5541c"/></marker>
         <marker id="${this.uid}-ah-g" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="2.6" markerHeight="2.6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2f5a24"/></marker>
         <marker id="${this.uid}-ah-w" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#ffffff"/></marker>`;
      const showVals = this.V && this.mode !== "none";
      for (let r = 0; r < m.R; r++)
        for (let c = 0; c < m.C; c++) {
          const ch = m.cell(r, c),
            x = c * cs,
            y = r * cs;
          const g = el("g", {}, svg);
          if (ch === "#") {
            el("rect", { x, y, width: cs, height: cs, fill: `url(#${this.uid}-hatch)` }, g);
            continue;
          }
          if (ch in m.exits) {
            const big = ch === "G",
              small = cs < 60;
            el("rect", { x, y, width: cs, height: cs, fill: big ? "#f3d97a" : "#f6c9a7" }, g);
            el(
              "text",
              {
                x: x + cs / 2,
                y: y + cs * (small ? 0.62 : 0.46),
                "text-anchor": "middle",
                "font-size": cs * (small ? 0.4 : 0.3),
                "font-weight": 700,
                fill: big ? "#5b4300" : "#7a3410",
              },
              g
            ).textContent = "+" + m.exits[ch];
            if (!small)
              el(
                "text",
                {
                  x: x + cs / 2,
                  y: y + cs * 0.74,
                  "text-anchor": "middle",
                  "font-size": cs * 0.14,
                  fill: big ? "#5b4300" : "#7a3410",
                  "letter-spacing": 1,
                },
                g
              ).textContent = big ? "BIG EXIT" : "SMALL EXIT";
            continue;
          }
          const s = m.index[r + "," + c];
          const v = showVals ? this.V[s] : null;
          const hidden = this.mode === "blank" && !this.revealed.has(s);
          const pol = this.arrows === "policy" ? this.pi : this.arrows === "greedy" && this.V ? greedy(m, this.V, this.g) : null;
          const tval = v === null ? 0 : (v - this.vmin) / (this.vmax - this.vmin);
          const fill = v !== null && !hidden ? heat(tval) : "#fbfaf6";
          const rect = el("rect", { x, y, width: cs, height: cs, fill }, g);
          const tile = m.states[s].tile;
          if (tile) {
            // coin / trap marker in the top-right corner
            const T = TILE_STYLE[tile] || TILE_STYLE.c,
              rr = cs * 0.17,
              tx = x + cs - rr - 3,
              ty = y + rr + 3;
            el("circle", { cx: tx, cy: ty, r: rr, fill: T.fill, stroke: T.ink, "stroke-width": 1.2 }, g);
            const tv = typeof m.tiles[tile] === "object" ? m.tiles[tile].r : m.tiles[tile];
            el(
              "text",
              { x: tx, y: ty + rr * 0.42, "text-anchor": "middle", "font-size": rr * 1.15, "font-weight": 700, fill: T.ink },
              g
            ).textContent = (tv > 0 ? "+" : "−") + Math.abs(tv);
          }
          const dark = v !== null && !hidden && tval > 0.5;
          if (v !== null && !hidden) el("title", {}, g).textContent = `${m.states[s].name}: ${fmt(v, 2)}`;
          const showNum = this.numbers !== null ? this.numbers : cs >= 48;
          const ink = dark ? "#ffffff" : "#1b1d1a";
          if (this.labels === "ids")
            el("text", { x: x + 6, y: y + cs * 0.17, "font-size": cs * 0.13, fill: dark ? "rgba(255,255,255,.75)" : "#8a9086" }, g).textContent =
              m.states[s].name;
          if (m.states[s].start)
            el(
              "rect",
              { x: x + 3, y: y + 3, width: cs - 6, height: cs - 6, fill: "none", stroke: "#6f9a3c", "stroke-width": 3, "stroke-dasharray": "6 4" },
              g
            );
          if (m.states[s].start && !pol)
            el(
              "text",
              { x: x + cs - 6, y: y + cs * 0.17, "text-anchor": "end", "font-size": cs * 0.13, "font-weight": 700, fill: dark ? "#fff" : "#2f5a24" },
              g
            ).textContent = "START";
          const both = v !== null && !!pol;
          if (v !== null && showNum) {
            const changed = this.prevV && Math.abs(this.prevV[s] - v) > 1e-9;
            el(
              "text",
              {
                x: x + cs / 2,
                y: y + cs * (both ? 0.9 : 0.6),
                "text-anchor": "middle",
                "font-size": cs * (both ? 0.17 : 0.22),
                "font-weight": changed ? 700 : 500,
                fill: hidden ? "#c9c9bf" : ink,
                "font-variant-numeric": "tabular-nums",
              },
              g
            ).textContent = hidden ? "?" : fmt(v, this.digits);
          }
          // arrows
          if (pol) {
            const cx = x + cs / 2,
              cy = y + cs * (both ? 0.43 : 0.5),
              L = cs * (both ? 0.23 : 0.34),
              L0 = cs * (both ? 0.05 : 0.1);
            pol[s].forEach((p, a) => {
              if (p <= 0) return;
              const [dy, dx] = ACTIONS[a].d;
              const off = 0;
              el(
                "line",
                {
                  x1: cx + dx * L0,
                  y1: cy + dy * L0 + off,
                  x2: cx + dx * L,
                  y2: cy + dy * L + off,
                  stroke: dark ? "#fff" : "#b5541c",
                  "stroke-width": Math.max(2, cs * 0.035 * Math.sqrt(p) * 1.6),
                  "stroke-linecap": "round",
                  "marker-end": `url(#${this.uid}-ah-${dark ? "w" : "o"})`,
                  opacity: 0.35 + 0.65 * p,
                },
                g
              );
            });
          }
          if (this.clickBackup || this.mode === "blank") {
            rect.style.cursor = "pointer";
            g.style.cursor = "pointer";
            g.addEventListener("click", (e) => {
              e.stopPropagation();
              this.onCell(s);
            });
          }
          if (this.selected === s)
            el("rect", { x: x + 2, y: y + 2, width: cs - 4, height: cs - 4, fill: "none", stroke: "#b5541c", "stroke-width": 4 }, g);
        }
      // grid lines + border
      for (let i = 1; i < m.C; i++) {
        el("line", { x1: i * cs, y1: 0, x2: i * cs, y2: this.size, stroke: "#d6d6cc", "stroke-width": 1 }, svg);
        el("line", { x1: 0, y1: i * cs, x2: this.size, y2: i * cs, stroke: "#d6d6cc", "stroke-width": 1 }, svg);
      }
      el("rect", { x: 1, y: 1, width: this.size - 2, height: this.size - 2, fill: "none", stroke: "#1b1d1a", "stroke-width": 2 }, svg);
      if (this.status) {
        let t = "";
        if (this.mode === "iterate") {
          const label = this.algo === "vi" ? "V_k" : this.algo === "pe" ? "V_k" : "V^{\\pi_k}";
          t = `$k=${this.k}$` + (this.lastDelta !== null ? ` &nbsp; $\\|V_{k}-V_{k-1}\\|_\\infty = ${fmt(this.lastDelta, 3)}$` : "");
          if (this.algo === "pi") {
            t = `iteration ${Math.floor(this.k / 2)} · next: <b>${this.piStage}</b>` + (this.stable ? ' · <b class="green">policy stable ✓</b>' : "");
            if (this.stepBtn) this.stepBtn.textContent = this.piStage === "evaluate" ? "Evaluate π" : "Improve π";
          }
        } else if (this.controls.includes("gamma")) {
          t = `start state: $V(s_1) = ${fmt(this.V[this.m.start], 3)}$`;
        }
        this.status.innerHTML = t;
        if (window.RLDeck) RLDeck.typeset(this.status);
      }
      if (this.showSA) this.drawTransition(svg, cs);
      if (this.paths) this.drawPaths(svg, cs);
      if (this.mark)
        el(
          "rect",
          {
            x: this.mark[1] * cs + 2,
            y: this.mark[0] * cs + 2,
            width: cs - 4,
            height: cs - 4,
            fill: "none",
            stroke: "#2b5b84",
            "stroke-width": 4,
            "stroke-dasharray": "5 3",
          },
          svg
        );
      if (this.overlay) this.overlay(svg, cs);
      if (this.selected !== null && this.info) this.showBackup(this.selected);
    }
    drawPaths(svg, cs) {
      const m = this.m,
        exitCell = (ch) => {
          for (let r = 0; r < m.R; r++) for (let c = 0; c < m.C; c++) if (m.cell(r, c) === ch) return [r, c];
        };
      this.paths.forEach((toks, k) => {
        const col = this.pathColors[k % this.pathColors.length],
          off = (k - (this.paths.length - 1) / 2) * cs * 0.13;
        const pts = toks.map((t) => {
          const [r, c] = /^[gG]$/.test(t) ? exitCell(t) : [m.states[+t - 1].r, m.states[+t - 1].c];
          return [c * cs + cs / 2 + off, r * cs + cs / 2 + off];
        });
        el(
          "marker",
          { id: `${this.uid}-pm${k}`, viewBox: "0 0 10 10", refX: 6, refY: 5, markerWidth: 3.2, markerHeight: 3.2, orient: "auto" },
          svg.querySelector("defs")
        ).innerHTML = `<path d="M0,0 L10,5 L0,10 z" fill="${col}"/>`;
        el(
          "polyline",
          {
            points: pts.map((p) => p.join(",")).join(" "),
            fill: "none",
            stroke: col,
            "stroke-width": Math.max(3, cs * 0.05),
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
            opacity: 0.7,
            "marker-end": `url(#${this.uid}-pm${k})`,
          },
          svg
        );
        el("circle", { cx: pts[0][0], cy: pts[0][1], r: cs * 0.07, fill: col }, svg);
      });
    }
    drawTransition(svg, cs) {
      const m = this.m,
        s = parseInt(this.showSA[0].replace(/\D/g, ""), 10) - 1;
      const a = ACTIONS.findIndex((A) => A.name === this.showSA[1] || A.sym === this.showSA[1]);
      if (s < 0 || a < 0) return;
      const st = m.states[s],
        add = (tag, at) => el(tag, at, svg);
      add("rect", {
        x: st.c * cs + 3,
        y: st.r * cs + 3,
        width: cs - 6,
        height: cs - 6,
        fill: "rgba(47,90,36,.10)",
        stroke: "#2f5a24",
        "stroke-width": 4,
      });
      const [dy, dx] = ACTIONS[a].d,
        cx = st.c * cs + cs / 2,
        cy = st.r * cs + cs / 2;
      add("line", {
        x1: cx - dx * cs * 0.15,
        y1: cy - dy * cs * 0.15,
        x2: cx + dx * cs * 0.32,
        y2: cy + dy * cs * 0.32,
        stroke: "#2f5a24",
        "stroke-width": 7,
        "stroke-linecap": "round",
        "marker-end": `url(#${this.uid}-ah-g)`,
      });
      // merge outcomes landing in the same cell
      const land = {};
      m.P[s][a].forEach((o) => {
        const stay = !o.done && o.next === s;
        const r = stay ? st.r : st.r + ACTIONS[o.dir].d[0],
          c = stay ? st.c : st.c + ACTIONS[o.dir].d[1];
        const key = r + "," + c;
        land[key] = land[key] || { r, c, p: 0 };
        land[key].p += o.p;
      });
      Object.values(land).forEach((L) => {
        const same = L.r === st.r && L.c === st.c,
          isExit = m.cell(L.r, L.c) in m.exits;
        const x = L.c * cs + cs / 2,
          y = L.r * cs + (same ? cs * 0.86 : isExit ? cs * 0.12 : cs * 0.5);
        add("rect", {
          x: x - cs * 0.3,
          y: y - cs * (isExit ? 0.1 : 0.15),
          width: cs * 0.6,
          height: cs * (isExit ? 0.19 : 0.27),
          rx: 6,
          fill: "#ffffff",
          stroke: "#b5541c",
          "stroke-width": 2,
        });
        add("text", {
          x,
          y: y + cs * (isExit ? 0.045 : 0.07),
          "text-anchor": "middle",
          "font-size": cs * (isExit ? 0.13 : 0.17),
          "font-weight": 700,
          fill: "#b5541c",
        }).textContent = (same ? "stay " : "") + (Math.abs(L.p - 1) < 1e-9 ? "1" : L.p.toFixed(2).replace(/^0/, ""));
      });
    }
    onCell(s) {
      if (this.mode === "blank" && !this.revealed.has(s)) {
        this.revealed.add(s);
        this.draw();
        return;
      }
      if (!this.clickBackup) return;
      this.selected = this.selected === s ? null : s;
      this.draw();
      if (this.selected === null) this.info.innerHTML = '<span class="muted">Click a state to see its Bellman backup.</span>';
    }
    // Bellman backup of state s from the *previous* iterate (iterate mode) or from the displayed V
    showBackup(s) {
      const m = this.m,
        g = this.g,
        name = "s_{" + (s + 1) + "}";
      const V = this.mode === "iterate" && this.prevV ? this.prevV : this.V;
      const Vname = this.mode === "iterate" && this.prevV ? "V_{" + (this.k - 1) + "}" : "V";
      const lines = ACTIONS.map((A, a) => {
        const terms = m.P[s][a]
          .map((t) => {
            const nxt = t.done ? "0" : `${fmt(g, 2)}\\cdot ${fmt(V[t.next], this.digits)}`;
            return (t.p < 1 ? `${fmt(t.p, 2)}(` : "") + `${t.reward} + ${nxt}` + (t.p < 1 ? ")" : "");
          })
          .join(" + ");
        return `Q(${name},${A.tex}) &= ${terms} = ${fmt(qValue(m, V, g, s, a), 3)}`;
      });
      const usePi = this.algo === "pe" || (this.mode !== "iterate" && this.policyName !== "optimal" && this.policyName !== "none");
      const agg = usePi
        ? `\\textstyle\\sum_a \\pi(a|${name})\\,Q(${name},a) = ${fmt(
            this.pi[s].reduce((acc, p, a) => acc + p * qValue(m, V, g, s, a), 0),
            3
          )}`
        : `\\max_a Q(${name},a) = ${fmt(Math.max(...ACTIONS.map((_, a) => qValue(m, V, g, s, a))), 3)}`;
      const tex = `\\begin{aligned}${lines.join("\\\\")}\\\\[4pt] \\text{new } V(${name}) &= ${agg}\\end{aligned}`;
      this.info.innerHTML = `<div class="small muted">backup from $${Vname}$, $\\gamma=${fmt(g, 2)}$</div>$$${tex}$$`;
      if (window.RLDeck) RLDeck.typeset(this.info);
    }
  }

  // ---------------------------------------------------------------- P as matrices (one |S|x|S| matrix per action)
  // <div class="pmatrix" data-action="right" data-slip="0" data-row="s6" data-slip-value="0.2"></div>
  class PMatrix {
    constructor(root) {
      const ds = root.dataset;
      this.root = root;
      this.a = Math.max(
        0,
        ACTIONS.findIndex((A) => A.name === (ds.action || "right"))
      );
      this.slip = parseFloat(ds.slip || "0");
      this.slipValue = parseFloat(ds.slipValue || "0.2");
      this.row = ds.row || "s6";
      this.cell = parseFloat(ds.cell || "25");
      root.innerHTML = "";
      this.bar = document.createElement("div");
      this.bar.className = "maze-controls";
      root.appendChild(this.bar);
      this.btns = ACTIONS.map((A, a) => {
        const b = document.createElement("button");
        b.textContent = `P[${A.sym}]`;
        b.onclick = (e) => {
          e.stopPropagation();
          this.a = a;
          this.draw();
        };
        this.bar.appendChild(b);
        return b;
      });
      this.slipBtn = document.createElement("button");
      this.slipBtn.onclick = (e) => {
        e.stopPropagation();
        this.slip = this.slip ? 0 : this.slipValue;
        this.draw();
      };
      this.bar.appendChild(this.slipBtn);
      this.host = document.createElement("div");
      root.appendChild(this.host);
      this.cap = document.createElement("div");
      this.cap.className = "maze-status";
      root.appendChild(this.cap);
      this.draw();
    }
    draw() {
      const { P, S, names } = matrices(buildModel(this.slip));
      const M = P[this.a],
        c = this.cell,
        L = 34,
        W = L + S * c + 2,
        H = L + S * c + 2;
      const ri = names.indexOf(this.row);
      let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="pmatrix-svg">`;
      names.forEach((n, i) => {
        const lab = n.replace(/^s/, "");
        svg += `<text x="${L + i * c + c / 2}" y="${L - 8}" text-anchor="middle" font-size="12" fill="${
          i >= S - 2 ? "#8a6a00" : "#8a9086"
        }">${lab}</text>`;
        svg += `<text x="${L - 6}" y="${L + i * c + c / 2 + 4}" text-anchor="end" font-size="12" fill="${
          i === ri ? "#b5541c" : i >= S - 2 ? "#8a6a00" : "#8a9086"
        }" font-weight="${i === ri ? 700 : 400}">${lab}</text>`;
      });
      for (let i = 0; i < S; i++)
        for (let j = 0; j < S; j++) {
          const v = M[i][j],
            x = L + j * c,
            y = L + i * c;
          svg += `<rect x="${x}" y="${y}" width="${c}" height="${c}" fill="${
            v ? heat(0.25 + 0.75 * v) : i >= S - 2 || j >= S - 2 ? "#faf6e8" : "#fbfaf6"
          }" stroke="#e4e4dc"/>`;
          if (v)
            svg += `<text x="${x + c / 2}" y="${y + c / 2 + 4}" text-anchor="middle" font-size="${v === 1 ? 12 : 10}" fill="${
              v > 0.5 ? "#fff" : "#1b1d1a"
            }">${v === 1 ? "1" : v.toFixed(2).replace(/^0/, "")}</text>`;
        }
      if (ri >= 0) svg += `<rect x="${L - 1}" y="${L + ri * c}" width="${S * c + 2}" height="${c}" fill="none" stroke="#b5541c" stroke-width="2.5"/>`;
      svg += `<text x="4" y="14" font-size="12" fill="#4a4f47">s ↓  s′ →</text></svg>`;
      this.host.innerHTML = svg;
      this.btns.forEach((b, a) => b.classList.toggle("on", a === this.a));
      this.slipBtn.textContent = this.slip ? "slippery" : "deterministic";
      const nz = M.reduce((acc, row) => acc + row.filter(Boolean).length, 0);
      this.cap.innerHTML = `$P[${ACTIONS[this.a].tex}] \\in [0,1]^{${S}\\times${S}}$ · ${nz} non-zeros · row ${this.row} sums to ${
        ri >= 0 ? M[ri].reduce((x, y) => x + y).toFixed(2) : "–"
      }`;
      if (window.RLDeck) RLDeck.typeset(this.cap);
    }
  }

  // ---------------------------------------------------------------- styles
  const css = `
  .maze { display: flex; gap: 28px; align-items: flex-start; font-family: var(--font-text); }
  .maze.vertical, .cols .maze { flex-direction: column; align-items: flex-start; gap: 10px; }
  .maze.vertical { align-items: center; }
  .cols .maze-side { min-width: 0; width: 100%; }
  .maze-svg { flex: none; display: block; }
  .maze-svg text { font-family: var(--font-text); }
  .maze-side { display: flex; flex-direction: column; gap: 12px; min-width: 260px; font-size: 20px; }
  .maze-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .maze-controls button { font: 600 17px var(--font-text); padding: 7px 14px; border-radius: 8px; border: 1.5px solid var(--accent); background: #fff; color: var(--accent); cursor: pointer; }
  .maze-controls button:hover { background: var(--accent-bg); }
  .maze-gamma { display: flex; align-items: center; gap: 10px; font-size: 20px; }
  .maze-gamma input { width: 200px; accent-color: var(--accent); }
  .maze-status { font-size: 20px; color: var(--ink-soft); min-height: 1.4em; }
  .maze-info { font-size: 20px; margin-top: 8px; }
  .maze-info .katex-display { margin: 4px 0; text-align: left; }
  .maze-info .katex { font-size: 1em; }
  .maze-controls button.on { background: var(--accent); color: #fff; }
  .pmatrix { display: flex; flex-direction: column; gap: 8px; font-family: var(--font-text); }
  .pmatrix-svg text { font-family: var(--font-text); }`;
  const st = document.createElement("style");
  st.textContent = css;
  document.head.appendChild(st);

  // ---------------------------------------------------------------- public API (for later lectures: MC/TD reuse the same model)
  window.RLMaze = {
    MAZES,
    TILE_STYLE,
    LAYOUT,
    EXIT_REWARD,
    ACTIONS,
    buildModel,
    evaluate,
    optimal,
    greedy,
    greedyDet,
    epsGreedy,
    epsMix,
    towards,
    matrices,
    PMatrix,
    applyTpi,
    applyTstar,
    Q,
    namedPolicy,
    Maze,
    heat,
    instances: [],
  };
  function init() {
    document.querySelectorAll(".maze").forEach((d) => RLMaze.instances.push(new Maze(d)));
    document.querySelectorAll(".pmatrix").forEach((d) => new PMatrix(d));
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

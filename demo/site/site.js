// Project site behaviour: citation scroll-and-highlight, pipeline detail
// panel, Figure 2 (paired differences). Numbers come from the page itself:
// panel text from <template id="node-KEY">, figure data from #fig-data.
(() => {
  "use strict";
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Citations: delegated, so links cloned into the pipeline panel work too.
  document.addEventListener("click", e => {
    const a = e.target.closest("a.cite"); if (!a) return;
    const li = document.getElementById("ref-" + a.dataset.ref); if (!li) return;
    e.preventDefault();
    document.querySelectorAll(".refs li.flash").forEach(x => x.classList.remove("flash"));
    li.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "center" });
    li.classList.add("flash");
    li.focus({ preventScroll: true });
    clearTimeout(li._t); li._t = setTimeout(() => li.classList.remove("flash"), 2600);
  });

  // Pipeline: hover, focus, click or Enter/Space on a node shows its template.
  const panel = document.getElementById("pipe-panel");
  const nodes = document.querySelectorAll("svg.pipeline .node");
  const show = k => {
    const tpl = document.getElementById("node-" + k);
    if (!tpl || !panel) return;
    panel.replaceChildren(tpl.content.cloneNode(true));
    nodes.forEach(n => n.classList.toggle("on", n.dataset.k === k));
  };
  nodes.forEach(n => {
    const k = n.dataset.k;
    n.addEventListener("mouseenter", () => show(k));
    n.addEventListener("focus", () => show(k));
    n.addEventListener("click", () => show(k));
    n.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); show(k); }
    });
  });
  show("split");

  // Figure 2: per-seed paired differences from the baseline, mean tick and CI bar.
  const src = document.getElementById("fig-data"); if (!src) return;
  const FIG = JSON.parse(src.textContent);
  const rows = [["hed", "HED", "--c-hed"], ["mac", "Macenko †", "--c-mac"],
                ["full", "HED + Macenko †", "--c-full"]];
  const NS = "http://www.w3.org/2000/svg";
  const W = 460, H = 214, L = 118, Rm = 16, T = 10, B = 40, x0 = -0.16, x1 = 0.14;
  const sx = v => L + (v - x0) / (x1 - x0) * (W - L - Rm);
  const rowH = (H - T - B) / rows.length;
  const el = (p, tag, a, txt) => {
    const e = document.createElementNS(NS, tag);
    for (const k in a) e.setAttribute(k, a[k]);
    if (txt != null) e.textContent = txt;
    p.appendChild(e); return e;
  };
  const css = getComputedStyle(document.documentElement);

  function draw(svg) {
    const split = svg.dataset.split, runs = FIG.runs[split];
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    for (let t = -0.15; t <= 0.1001; t += 0.05) {
      t = +t.toFixed(2); const x = sx(t);
      el(svg, "line", { x1: x, x2: x, y1: T, y2: H - B, class: Math.abs(t) < 1e-9 ? "zero" : "grid" });
      el(svg, "text", { x, y: H - B + 16, "text-anchor": "middle" },
         (t > 0 ? "+" : t < 0 ? "−" : "") + Math.abs(t).toFixed(2));
    }
    el(svg, "line", { x1: L, x2: W - Rm, y1: H - B, y2: H - B, class: "axis" });
    el(svg, "text", { x: L + (W - L - Rm) / 2, y: H - 6, "text-anchor": "middle" },
       "Δ Dice vs baseline (same seed)");
    const ink = css.getPropertyValue("--ink").trim();
    rows.forEach(([k, name, c], i) => {
      const cy = T + rowH * (i + 0.5);
      el(svg, "text", { x: L - 10, y: cy + 4, "text-anchor": "end", class: "lab" }, name);
      const col = css.getPropertyValue(c).trim();
      const d = runs[k].map((v, j) => v - runs.base[j]).sort((a, b) => a - b);
      // deterministic dodge: points within 4.5px stack alternately above/below
      const placed = [];
      d.forEach(v => {
        const x = sx(v); let n = 0;
        for (const p of placed) if (Math.abs(p - x) < 4.5) n++;
        placed.push(x);
        const off = n === 0 ? 0 : (n % 2 ? -1 : 1) * Math.ceil(n / 2) * 4.6;
        el(svg, "circle", { cx: x, cy: cy - 9 + off, r: 3, fill: col, "fill-opacity": 0.8 });
      });
      const m = d.reduce((a, b) => a + b, 0) / d.length, [lo, hi] = FIG.ci[split][k];
      const by = cy + 13;
      el(svg, "line", { x1: sx(lo), x2: sx(hi), y1: by, y2: by, stroke: ink, "stroke-width": 2 });
      el(svg, "line", { x1: sx(m), x2: sx(m), y1: by - 5, y2: by + 5, stroke: ink, "stroke-width": 2 });
    });
  }
  document.querySelectorAll("svg[data-split]").forEach(draw);
})();

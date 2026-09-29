// Study page: bar chart, cost scatter and the full table, all from results/*/metrics.json via site-data.js.
(() => {
  const { runs: RUNS, roi_sahi: ROI } = window.BALLHAWK;
  const ROWS = [
    ["baseline_yolo11s_640", "02 Baseline", "none", "ref"],
    ["baseline_yolo11s_640_seed1", "02b Baseline, seed 1", "seed 0 → 1", "ref"],
    ["res_yolo11s_960", "03 Train at 960", "input 640 → 960", "int"],
    ["res_yolo11s_1280", "03 Train at 1280", "input 640 → 1280, batch 8", "int"],
    ["aug_scale02_yolo11s_640", "04a Scale jitter 0.2", "scale 0.5 → 0.2", "int"],
    ["aug_scale02_mblur_yolo11s_640", "04b + Motion blur", "on top of 04a", "int"],
    ["p2_yolo11s_640", "05 P2 head", "detect at stride 4", "int"],
    ["clsw05_yolo11s_640", "05b Class weights", "cls_pw 0 → 0.5", "int"],
    ["ctrl_reinit17_yolo11s_640", "05c Control for 05", "head layers re-initialised", "ref"],
    ["sahi_baseline_s960", "06 Sliced inference", "960 px slices on 02", "int"],
  ].filter((r) => RUNS[r[0]]);
  const fmt = (v, d = 3) => (v == null ? "n/a" : v.toFixed(d));
  const tipHtml = (r, m) => `<b>${r[1]}</b>${r[2]}<br>Ball test ${fmt(m.ball)} · val ${fmt(m.val_ball)}<br>All classes ${fmt(m.all)}<br>${fmt(m.fps, 1)} FPS · batch ${m.batch}`;
  const colour = (r) => `var(${r[3] === "ref" ? "--ref" : "--accent"})`;

  // ---------- bar chart ----------
  const MAX = 0.32, base = RUNS.baseline_yolo11s_640.ball;
  // Seed noise: sd of the baseline's test ball mAP50-95 over seeds 0, 1, 2 (Kaggle replication, results/kaggle_seeds).
  const noise = 0.0088, pct = (v) => (100 * v / MAX) + "%";
  const bars = document.getElementById("bars");
  const fills = [];
  ROWS.forEach((r) => {
    const m = RUNS[r[0]];
    const row = document.createElement("div");
    row.className = "bar-row"; row.tabIndex = 0; row.setAttribute("role", "listitem");
    row.setAttribute("aria-label", `${r[1]}: ball mAP50-95 ${fmt(m.ball)}`);
    row.innerHTML = `<span class="lbl">${r[1]}</span>
      <div class="track"><div class="band" style="left:${pct(base - noise)};width:${pct(2 * noise)}"></div>
      <div class="fill" style="background:${colour(r)}"></div>
      <div class="base" style="left:${pct(base)}"></div></div>
      <span class="val">${fmt(m.ball)}</span>`;
    fills.push([row.querySelector(".fill"), pct(m.ball)]);
    bhTip.bind(row, () => tipHtml(r, m));
    bars.appendChild(row);
  });
  const ticks = document.getElementById("ticks");
  [0, 0.1, 0.2, 0.3].forEach((t) => { const s = document.createElement("span"); s.style.left = pct(t); s.textContent = t.toFixed(1); ticks.appendChild(s); });
  // Grow the bars when the chart first scrolls into view.
  const grow = () => fills.forEach(([el, w]) => { el.style.width = w; });
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((en) => { if (en.some((e) => e.isIntersecting)) { grow(); io.disconnect(); } }, { threshold: 0.2 });
    io.observe(bars);
  } else grow();

  // ---------- table ----------
  const tb = document.querySelector("#runs-table tbody");
  ROWS.forEach((r) => {
    const m = RUNS[r[0]], tr = document.createElement("tr");
    tr.innerHTML = `<td>${r[1]}</td><td>${r[2]}</td><td>${fmt(m.ball)}</td><td>${fmt(m.val_ball)}</td><td>${fmt(m.ballR, 2)}</td><td>${fmt(m.all)}</td><td>${fmt(m.fps, 1)}</td><td>${m.batch}</td><td>${m.min ? m.min.toFixed(1) : "inference only"}</td>`;
    tb.appendChild(tr);
  });

  // ---------- scatter: ball mAP vs FPS (log x) ----------
  const svg = document.getElementById("scatter"), NS = "http://www.w3.org/2000/svg";
  const P = { l: 56, r: 24, t: 16, b: 44 }, W = 760, H = 380;
  const fx = (v) => P.l + (Math.log10(v) - Math.log10(4)) / (Math.log10(70) - Math.log10(4)) * (W - P.l - P.r);
  const fy = (v) => H - P.b - (v / 0.32) * (H - P.t - P.b);
  const el = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text) e.textContent = text; svg.appendChild(e); return e; };
  [0, 0.1, 0.2, 0.3].forEach((t) => {
    el("line", { x1: P.l, x2: W - P.r, y1: fy(t), y2: fy(t), stroke: "var(--rule)" });
    el("text", { x: P.l - 10, y: fy(t) + 4, "text-anchor": "end" }, t.toFixed(1));
  });
  [5, 10, 20, 50].forEach((t) => {
    el("line", { x1: fx(t), x2: fx(t), y1: P.t, y2: H - P.b, stroke: "var(--rule)" });
    el("text", { x: fx(t), y: H - P.b + 18, "text-anchor": "middle" }, String(t));
  });
  el("text", { x: (P.l + W - P.r) / 2, y: H - 6, "text-anchor": "middle" }, "inference FPS (log scale)");
  el("text", { x: 14, y: (P.t + H - P.b) / 2, transform: `rotate(-90 14 ${(P.t + H - P.b) / 2})`, "text-anchor": "middle" }, "ball mAP50-95");
  // The 640 runs cluster near 45 FPS at the right edge: label them on their left so nothing clips.
  const LABEL = { res_yolo11s_1280: [-10, -10, "end"], res_yolo11s_960: [-12, -8, "end"], sahi_baseline_s960: [10, 4, "start"],
    p2_yolo11s_640: [-12, 10, "end"], ctrl_reinit17_yolo11s_640: [-12, -2, "end"], baseline_yolo11s_640: [-14, 16, "end"],
    aug_scale02_yolo11s_640: [-14, -10, "end"] };
  ROWS.forEach((r) => {
    const m = RUNS[r[0]], cx = fx(m.fps), cy = fy(m.ball);
    const c = el("circle", { cx, cy, r: 7, fill: colour(r), stroke: "var(--raised)", "stroke-width": 2, tabindex: 0,
      "aria-label": `${r[1]}: ${fmt(m.ball)} at ${fmt(m.fps, 1)} FPS` });
    bhTip.bind(c, () => tipHtml(r, m));
    if (LABEL[r[0]]) { const [dx, dy, a] = LABEL[r[0]]; el("text", { x: cx + dx, y: cy + dy, "text-anchor": a, class: "pt-label" }, r[1]); }
  });

  // ---------- ROI slicing (add-on study; hidden until results/roi_sahi exists) ----------
  if (ROI) {
    document.getElementById("roi").hidden = false;
    const num = (v) => (Number.isInteger(v) ? String(v) : Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(1));
    document.querySelectorAll("[data-roi]").forEach((e) => {
      const v = e.dataset.roi.split(".").reduce((o, k) => o?.[k], ROI);
      if (v != null) e.textContent = typeof v === "number" ? num(v) : v;
    });
    document.querySelectorAll("[data-roi-ms]").forEach((e) => { e.textContent = Math.round(ROI[e.dataset.roiMs].ms); });
    const b = ROI.baseline, derived = {
      share: `${Math.round(100 * (ROI.roi.ball - b.ball) / (ROI.sahi.ball - b.ball))} %`,
      cost: `${(ROI.roi.ms / b.ms).toFixed(1)}×`,
      sahi_cost: `${(ROI.sahi.ms / b.ms).toFixed(1)}×`,
    };
    document.querySelectorAll("[data-roi-derived]").forEach((e) => { e.textContent = derived[e.dataset.roiDerived]; });
  }
})();

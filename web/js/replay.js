// Tactical page: the 30-second clip replayed on a pitch drawn in the browser from data/replay.json
// (per-frame smoothed positions written by ballhawk_tactical.render_clip). Layers can be switched on and off.
(() => {
  const root = document.getElementById("replay");
  if (!root) return;
  const $ = (id) => document.getElementById(id);
  const canvas = $("rp-canvas"), ctx = canvas.getContext("2d");
  // Same colours as the rendered video (ballhawk_hud): team A, team B, goalkeeper, referee, unassigned.
  const KIND = ["#2a78d6", "#46be50", "#dc3cc8", "#ffc800", "#aaaaaa"];
  const OFFSIDE = "#eb3c3c", M = 4;  // metres of margin around the pitch
  const S = { data: null, f: 0, playing: false, t0: 0, f0: 0, layers: { offside: true, space: true, shapes: true, camera: false } };
  let L = 105, Wd = 68, scale = 1;

  const px = (x) => (x + M) * scale, py = (y) => (y + M) * scale;
  function resize() {
    const w = canvas.clientWidth, dpr = window.devicePixelRatio || 1;
    scale = w / (L + 2 * M);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round((Wd + 2 * M) * scale * dpr);
    canvas.style.height = `${(Wd + 2 * M) * scale}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function pitch() {
    ctx.fillStyle = "#16251c"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let k = 0; k < 12; k += 2) { ctx.fillStyle = "#1b2d22"; ctx.fillRect(px(k * L / 12), py(0), (L / 12) * scale, Wd * scale); }
    ctx.strokeStyle = "rgba(230,235,230,0.75)"; ctx.lineWidth = 1.5;
    ctx.strokeRect(px(0), py(0), L * scale, Wd * scale);
    ctx.beginPath(); ctx.moveTo(px(L / 2), py(0)); ctx.lineTo(px(L / 2), py(Wd)); ctx.stroke();
    ctx.beginPath(); ctx.arc(px(L / 2), py(Wd / 2), 9.15 * scale, 0, 2 * Math.PI); ctx.stroke();
    for (const [x0, s] of [[0, 1], [L, -1]]) {
      ctx.strokeRect(px(Math.min(x0, x0 + s * 16.5)), py((Wd - 40.32) / 2), 16.5 * scale, 40.32 * scale);
      ctx.strokeRect(px(Math.min(x0, x0 + s * 5.5)), py((Wd - 18.32) / 2), 5.5 * scale, 18.32 * scale);
    }
  }

  // Point in a convex polygon (the camera footprint): space control covers only what the camera saw.
  const inside = (poly, x, y) => {
    let sign = 0;
    for (let i = 0; i < poly.length; i++) {
      const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
      const c = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
      if (c !== 0) { if (sign === 0) sign = Math.sign(c); else if (Math.sign(c) !== sign) return false; }
    }
    return true;
  };
  const cells = document.createElement("canvas"), cctx = cells.getContext("2d");
  cells.width = 105; cells.height = 68;
  function space(fr) {
    const outfield = fr.p.filter((q) => q[1] <= 1);
    if (!fr.v || new Set(outfield.map((q) => q[1])).size < 2) return;
    const img = cctx.createImageData(105, 68), rgb = [[42, 120, 214], [70, 190, 80]];
    for (let j = 0; j < 68; j++) for (let i = 0; i < 105; i++) {
      const x = i + 0.5, y = j + 0.5;
      if (!inside(fr.v, x, y)) continue;
      let best = Infinity, team = 0;
      for (const q of outfield) { const d = (q[2] - x) ** 2 + (q[3] - y) ** 2; if (d < best) { best = d; team = q[1]; } }
      img.data.set([...rgb[team], 70], 4 * (j * 105 + i));
    }
    cctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cells, px(0), py(0), L * scale, Wd * scale);
  }

  function hull(points) {  // monotone chain
    const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (p.length < 3) return [];
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [], upper = [];
    for (const q of p) { while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), q) <= 0) lower.pop(); lower.push(q); }
    for (const q of p.reverse()) { while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), q) <= 0) upper.pop(); upper.push(q); }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  const clock = (f) => `${Math.floor(f / S.data.fps / 60)}:${String(Math.floor(f / S.data.fps) % 60).padStart(2, "0")}`;
  function draw() {
    pitch();
    const fr = S.data?.frames[S.f];
    if (!fr) return;
    if (S.layers.camera && fr.v) {
      ctx.beginPath(); fr.v.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)))); ctx.closePath();
      ctx.fillStyle = "rgba(255,255,255,0.08)"; ctx.fill();
      ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
    }
    if (S.layers.space) space(fr);
    if (S.layers.shapes) for (const t of [0, 1]) {
      const h = hull(fr.p.filter((q) => q[1] === t).map((q) => [q[2], q[3]]));
      if (!h.length) continue;
      ctx.beginPath(); h.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)))); ctx.closePath();
      ctx.fillStyle = KIND[t] + "26"; ctx.fill(); ctx.strokeStyle = KIND[t]; ctx.lineWidth = 2; ctx.stroke();
    }
    if (S.layers.offside && fr.l != null) {
      ctx.strokeStyle = OFFSIDE; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(px(fr.l), py(0)); ctx.lineTo(px(fr.l), py(Wd)); ctx.stroke();
    }
    const r = Math.max(4, 0.95 * scale);
    for (const [, k, x, y] of fr.p) {
      ctx.beginPath(); ctx.arc(px(x), py(y), r, 0, 2 * Math.PI);
      ctx.fillStyle = KIND[k]; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.stroke();
    }
    if (fr.b) {
      ctx.beginPath(); ctx.arc(px(fr.b[0]), py(fr.b[1]), r * 0.6, 0, 2 * Math.PI);
      ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#0d1210"; ctx.stroke();
    }
    const last = S.data.frames.length - 1;
    $("rp-time").textContent = `${clock(S.f)} / ${clock(last)}`;
    $("rp-seek").value = S.f;
    $("rp-seek").setAttribute("aria-valuetext", `${clock(S.f)} of ${clock(last)}`);
    $("rp-line").textContent = fr.l != null ? `${fr.l.toFixed(1)} m` : "not drawn";
  }

  function tick(now) {
    if (!S.playing) return;
    S.f = (S.f0 + Math.floor((now - S.t0) / 1000 * S.data.fps)) % S.data.frames.length;
    draw();
    requestAnimationFrame(tick);
  }
  function setPlaying(on) {
    S.playing = on;
    $("rp-play").setAttribute("aria-label", on ? "Pause" : "Play");
    $("rp-play").innerHTML = on
      ? '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>'
      : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l11-6.5z"/></svg>';
    if (on) { S.t0 = performance.now(); S.f0 = S.f; requestAnimationFrame(tick); }
  }

  async function load() {
    try {
      const r = await fetch("data/replay.json");
      if (!r.ok) return;
      S.data = await r.json();
    } catch (e) { return; }  // no data published: the section stays hidden
    L = S.data.length; Wd = S.data.width;
    root.hidden = false;
    $("rp-seek").max = S.data.frames.length - 1;
    $("rp-play").addEventListener("click", () => setPlaying(!S.playing));
    $("rp-seek").addEventListener("input", (e) => { S.f = +e.target.value; S.f0 = S.f; S.t0 = performance.now(); draw(); });
    document.querySelectorAll("#rp-layers button").forEach((b) => {
      b.setAttribute("aria-pressed", String(S.layers[b.dataset.layer]));
      b.addEventListener("click", () => {
        S.layers[b.dataset.layer] = !S.layers[b.dataset.layer];
        b.setAttribute("aria-pressed", String(S.layers[b.dataset.layer]));
        draw();
      });
    });
    new ResizeObserver(resize).observe(canvas);
    setPlaying(false);
    // Starts playing when it scrolls into view, unless the reader prefers reduced motion.
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches && "IntersectionObserver" in window) {
      const io = new IntersectionObserver((en) => { if (en.some((x) => x.isIntersecting)) { setPlaying(true); io.disconnect(); } }, { threshold: 0.5 });
      io.observe(canvas);
    }
  }
  load();
})();

// Live demo: talks to the BallHawk server's /api (demo/app.py). Frame requests return the image inline;
// clips run as background jobs that this page polls. Manual calibration sends clicked point pairs.
(() => {
  const $ = (id) => document.getElementById(id);
  const S = { mode: "frame", source: null, frameUrl: null, pitch: null, pairs: [], pending: {}, busy: false };
  const LIMIT_MB = { frame: 20, clip: 200 };

  // ---------- helpers ----------
  const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
  const pct = (v) => `${Math.round(100 * v)} %`;
  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    let body = null;
    try { body = await r.json(); } catch (e) { /* non-JSON error page */ }
    if (!r.ok) throw new Error(body?.detail || `The server answered ${r.status}.`);
    return body;
  }
  function show(which) {  // one of: ph, busy, img, video, progress
    $("ph").hidden = which !== "ph"; $("busy").hidden = which !== "busy" && which !== "progress";
    $("out-img").hidden = which !== "img"; $("out-video").hidden = which !== "video";
    $("progress").hidden = which !== "progress";
  }
  function error(msg) { $("error").hidden = !msg; $("error-msg").textContent = msg || ""; }
  function tiles(list) {
    $("tiles").replaceChildren(...list.map(([label, value, sub]) => el("div", { className: "tile" },
      el("span", { className: "eyebrow", textContent: label }), el("span", { className: "num", textContent: value }),
      ...(sub ? [el("span", { className: "small muted", textContent: sub })] : []))));
  }
  function notes(list) {
    $("notes").replaceChildren(...list.map((n) => el("li", { className: n.warn ? "warn" : "", textContent: n.text })));
  }
  function setBusy(on) {
    S.busy = on;
    $("run").disabled = on || !S.source;
    $("apply").disabled = on || S.pairs.length < 4;
  }

  // ---------- mode and source ----------
  let EX = { frames: [], clips: [] };
  function setMode(mode) {
    S.mode = mode;
    document.querySelectorAll("#mode button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    $("file").accept = mode === "frame" ? "image/*" : "video/*,.mov,.mkv";
    $("drop-hint").textContent = mode === "frame" ? "JPG or PNG broadcast frame, up to 20 MB" : "MP4, MOV or WebM clip, up to 200 MB; only the first seconds are used";
    $("clip-opts").hidden = mode !== "clip";
    $("run").textContent = mode === "frame" ? "Analyse frame" : "Render clip";
    setSource(null);
    renderExamples();
  }
  function setSource(src) {
    S.source = src;
    $("file-name").hidden = !(src && src.kind === "file");
    if (src?.kind === "file") $("file-name").textContent = src.file.name;
    else $("file").value = "";
    document.querySelectorAll("#examples button").forEach((b) => b.setAttribute("aria-pressed", String(src?.kind === "example" && b.dataset.name === src.name)));
    $("run-hint").textContent = src ? (src.kind === "file" ? "Ready." : `Example: ${src.name}`) : "Choose a file or an example.";
    setBusy(S.busy);
  }
  function takeFile(file) {
    if (!file) return;
    const isVideo = file.type.startsWith("video/") || /\.(mov|mkv|m4v|avi|mp4|webm)$/i.test(file.name);
    if (isVideo !== (S.mode === "clip")) setMode(isVideo ? "clip" : "frame");
    if (file.size > LIMIT_MB[S.mode] * 2 ** 20) { error(`That file is larger than ${LIMIT_MB[S.mode]} MB.`); return; }
    error(null);
    setSource({ kind: "file", file });
  }
  function renderExamples() {
    const list = S.mode === "frame" ? EX.frames : EX.clips;
    $("examples").replaceChildren(...list.map((x, k) => {
      const media = el("img", { src: S.mode === "frame" ? x.url : "img/tactical_v2_poster.jpg", alt: "", loading: "lazy" });
      const b = el("button", { type: "button", title: x.name }, media, el("span", { className: "tag", textContent: S.mode === "frame" ? `Frame ${k + 1}` : "DFL clip" }));
      b.dataset.name = x.name;
      b.setAttribute("aria-pressed", "false");
      b.setAttribute("aria-label", `Use example ${x.name}`);
      b.addEventListener("click", () => { error(null); setSource({ kind: "example", name: x.name, url: x.url }); });
      return b;
    }));
  }
  function formData(extra = {}) {
    const fd = new FormData();
    if (S.source.kind === "file") fd.append("file", S.source.file); else fd.append("example", S.source.name);
    for (const [k, v] of Object.entries(extra)) fd.append(k, v);
    return fd;
  }

  // ---------- frame ----------
  async function runFrame(points) {
    setBusy(true); error(null); show("busy");
    $("busy-msg").textContent = points ? "Applying your calibration" : "Detecting, calibrating and drawing";
    $("result-actions").hidden = true;
    try {
      const res = await api("/api/frame", { method: "POST", body: formData(points ? { points: JSON.stringify(points) } : {}) });
      $("out-img").src = res.image; show("img");
      const s = res.stats, c = s.counts, cal = s.calibration;
      tiles([
        ["People", `${c.player} players`, `${c.keeper} keeper(s), ${c.referee} referee(s)`],
        ["Ball", c.ball ? `${c.ball} detection${c.ball > 1 ? "s" : ""}` : "Not found", c.ball ? "Most confident one drawn" : "Often hidden or blurred"],
        ["Calibration", cal.method === "manual" ? "By hand" : cal.method === "auto" ? "Automatic" : "Failed",
          cal.method ? `${cal.points} landmarks${cal.residual_m != null ? `, ${Math.round(100 * cal.residual_m)} cm RMS` : ""}` : "Calibrate by hand below"],
        ["Offside line", s.offside.line_x != null ? `${s.offside.line_x.toFixed(1)} m` : "n/a", s.offside.line_x != null ? `${s.offside.beyond} attacker(s) beyond` : "Needs calibration, a keeper and two teams"],
        ["Space control", s.space_share ? `A ${pct(s.space_share.A)} · B ${pct(s.space_share.B)}` : "n/a", "Visible pitch, nearest player"],
      ]);
      notes(res.notes);
      $("download").href = res.image; $("download").download = "ballhawk_tactical_view.jpg"; $("download").textContent = "Download image";
      if (!points) {
        if (S.frameUrl?.startsWith("blob:")) URL.revokeObjectURL(S.frameUrl);
        S.frameUrl = S.source.kind === "file" ? URL.createObjectURL(S.source.file) : S.source.url;
        if (cal.method === null) openCalib(false).catch((e) => error(e.message));
      }
      $("open-calib").hidden = false; $("result-actions").hidden = false;
    } catch (e) {
      show("ph"); error(e.message);
    } finally { setBusy(false); }
  }

  // ---------- clip ----------
  async function runClip() {
    setBusy(true); error(null); show("progress");
    $("busy-msg").textContent = "Rendering the tactical view, frame by frame";
    $("progress-bar").style.width = "0%"; $("progress-pct").textContent = "0 %"; $("progress-msg").textContent = "Uploading";
    $("result-actions").hidden = true; $("calib").hidden = true;
    tiles([]); notes([]);
    try {
      const { job } = await api("/api/clip", { method: "POST", body: formData({ seconds: $("seconds").value, stride: $("stride").value }) });
      for (;;) {
        const j = await api(`/api/jobs/${job}`);
        $("progress-bar").style.width = pct(j.progress); $("progress-pct").textContent = pct(j.progress);
        $("progress-msg").textContent = j.message || j.state;
        if (j.state === "done") { clipDone(j); break; }
        if (j.state === "error") throw new Error(`The clip could not be processed: ${j.message}`);
        await new Promise((r) => setTimeout(r, 1200));
      }
    } catch (e) {
      show("ph"); error(e.message);
    } finally { setBusy(false); }
  }
  function clipDone(j) {
    const s = j.stats, v = $("out-video");
    v.src = j.video; show("video"); v.play().catch(() => {});
    tiles([
      ["Frames", `${s.frames}`, `at ${s.fps.toFixed(1)} fps after skipping`],
      ["Calibrated", pct(s.calibrated_share), `${s.rejected_calibrations} jump(s) rejected`],
      ["Ball found", `${pct(s.ball_detected_share)} → ${pct(s.ball_filled_share)}`, "Detected → after gap filling"],
      ["Possession", s.possession_share ? `A ${pct(s.possession_share.A)} · B ${pct(s.possession_share.B)}` : "n/a", "Ball within 3 m of a player"],
      ["Space control", s.space_share ? `A ${pct(s.space_share.A)} · B ${pct(s.space_share.B)}` : "n/a", "Mean over calibrated frames"],
      ["Visible shape", s.formation ? `${s.formation.A ?? "–"} · ${s.formation.B ?? "–"}` : "n/a",
        s.frames / s.fps < 8 ? "Needs a clip of 8 s or more" : "Team A · team B, from mean positions"],
    ]);
    const n = [];
    if (s.rejected_calibrations > 0.2 * s.frames) n.push({ text: "Many calibration jumps were rejected: this camera angle is far from the broadcast view the keypoint model was trained on.", warn: true });
    if (s.median_line_step_m != null) n.push({ text: `The offside line moved a median ${Math.round(100 * s.median_line_step_m)} cm between frames.` });
    const top = Object.entries(s.top_distance_m || {});
    if (top.length) n.push({ text: `Top distance covered: ${top.map(([k, d]) => `${k} ${d} m`).join(", ")}.` });
    n.push({ text: `Team colours agree with each player's track in ${pct(s.team_agreement)} of frames.` });
    notes(n);
    $("download").href = j.video; $("download").download = "ballhawk_tactical_view.mp4"; $("download").textContent = "Download video";
    $("open-calib").hidden = true; $("result-actions").hidden = false;
  }

  // ---------- manual calibration ----------
  const NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  function drawPitch() {
    const p = $("pitch"), L = S.pitch.length, W = S.pitch.width, line = { fill: "none", stroke: "rgba(255,255,255,.75)", "stroke-width": 0.35 };
    p.replaceChildren();
    const box = (x, y, w, h) => p.append(svg("rect", { x, y, width: w, height: h, ...line }));
    box(0, 0, L, W);
    p.append(svg("line", { x1: L / 2, y1: 0, x2: L / 2, y2: W, ...line }), svg("circle", { cx: L / 2, cy: W / 2, r: 9.15, ...line }));
    for (const [x0, s] of [[0, 1], [L, -1]]) {
      box(Math.min(x0, x0 + s * 16.5), (W - 40.32) / 2, 16.5, 40.32);
      box(Math.min(x0, x0 + s * 5.5), (W - 18.32) / 2, 5.5, 18.32);
    }
    S.pitch.keypoints.forEach((k) => {
      const g = svg("g", { class: "kp", tabindex: 0, role: "button", "aria-label": k.name, "data-i": k.i });
      g.append(svg("circle", { class: "hit", cx: k.x, cy: k.y, r: 3 }), svg("circle", { class: "dot", cx: k.x, cy: k.y, r: 1.3 }));
      const t = svg("title"); t.textContent = k.name; g.append(t);
      const pick = () => { S.pending.k = k.i; pair(); };
      g.addEventListener("click", pick);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      p.append(g);
    });
  }
  function pair() {
    const { img, k } = S.pending;
    if (img && k != null) {
      S.pairs = S.pairs.filter((q) => q[2] !== k);  // a landmark can be used once
      S.pairs.push([img[0], img[1], k]); S.pending = {};
    }
    renderCalib();
  }
  function renderCalib() {
    const im = $("shot-img"), m = $("shot-marks");
    m.setAttribute("viewBox", `0 0 ${im.naturalWidth || 1} ${im.naturalHeight || 1}`);
    const r = (im.naturalWidth || 1920) / 160;
    const mark = (u, v, fill) => svg("circle", { cx: u, cy: v, r, fill, stroke: "#fff", "stroke-width": r / 4 });
    m.replaceChildren(...S.pairs.map(([u, v], n) => {
      const g = svg("g");
      const t = svg("text", { x: u + r * 1.4, y: v - r * 1.2, fill: "#fff", "font-size": r * 2.4, "font-family": "IBM Plex Mono, monospace", stroke: "#0d1210", "stroke-width": r / 3, "paint-order": "stroke" });
      t.textContent = n + 1;
      g.append(mark(u, v, "#1baf7a"), t);
      return g;
    }));
    if (S.pending.img) m.append(mark(S.pending.img[0], S.pending.img[1], "#F06B55"));
    document.querySelectorAll("#pitch .kp").forEach((g) => {
      const i = +g.dataset.i;
      g.classList.toggle("used", S.pairs.some((q) => q[2] === i));
      g.classList.toggle("armed", S.pending.k === i);
    });
    const name = (i) => S.pitch.keypoints[i].name;
    $("pairs").replaceChildren(...S.pairs.map(([u, v, k], n) => {
      const del = el("button", { type: "button", textContent: "✕", title: "Remove" });
      del.setAttribute("aria-label", `Remove pair ${n + 1}`);
      del.addEventListener("click", () => { S.pairs.splice(n, 1); renderCalib(); });
      return el("li", {}, el("span", { textContent: `${n + 1}. ${name(k)} · (${Math.round(u)}, ${Math.round(v)})` }), del);
    }));
    const need = 4 - S.pairs.length;
    $("calib-status").textContent = S.pending.img ? "Now choose the matching landmark on the pitch."
      : S.pending.k != null ? `Now click “${name(S.pending.k)}” on the frame.`
      : need > 0 ? `${need} more pair${need > 1 ? "s" : ""} to go.` : "Ready. Add a fifth pair to measure click accuracy, or apply.";
    setBusy(S.busy);
  }
  async function openCalib(scroll = true) {
    if (!S.pitch) { S.pitch = await api("/api/pitch"); drawPitch(); }
    const im = $("shot-img");
    if (im.dataset.src !== S.frameUrl) { S.pairs = []; S.pending = {}; im.dataset.src = S.frameUrl; im.src = S.frameUrl; }
    $("calib").hidden = false;
    im.decode().then(renderCalib, renderCalib);
    if (scroll) $("calib").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
  $("shot").addEventListener("click", (e) => {
    const im = $("shot-img"), b = im.getBoundingClientRect();
    if (!im.naturalWidth || e.clientY > b.bottom) return;
    S.pending.img = [(e.clientX - b.left) / b.width * im.naturalWidth, (e.clientY - b.top) / b.height * im.naturalHeight];
    pair();
  });
  $("open-calib").addEventListener("click", () => openCalib().catch((e) => error(e.message)));
  $("clear").addEventListener("click", () => { S.pairs = []; S.pending = {}; renderCalib(); });
  $("apply").addEventListener("click", async () => {
    await runFrame(S.pairs);
    $("view").scrollIntoView({ behavior: "smooth", block: "center" });
  });

  // ---------- wiring ----------
  document.querySelectorAll("#mode button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("file").addEventListener("change", (e) => takeFile(e.target.files[0]));
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); takeFile(e.dataTransfer.files[0]); });
  const eta = () => {
    const n = Math.round(+$("seconds").value * 25 / +$("stride").value);
    $("seconds-out").textContent = `${$("seconds").value} s`; $("stride-out").textContent = $("stride").value;
    $("eta").textContent = `About ${n} frames at 25 fps: roughly ${Math.max(1, Math.round(n * 0.8 / 60))} to ${Math.max(2, Math.round(n * 1.6 / 60))} minutes on a laptop CPU.`;
  };
  $("seconds").addEventListener("input", eta); $("stride").addEventListener("input", eta); eta();
  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!S.source || S.busy) return;
    if (S.mode === "frame") runFrame(); else runClip();
  });

  // Is the server there? The static copy of the site has no /api.
  (async () => {
    try {
      const ctl = new AbortController(); setTimeout(() => ctl.abort(), 5000);
      await api("/api/health", { signal: ctl.signal });
      EX = await api("/api/examples");
    } catch (e) {
      $("offline").hidden = false; $("form").inert = true; $("form").style.opacity = 0.5;
    }
    setMode("frame");
  })();
})();

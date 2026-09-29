// Hero: a real test-split ball resampled to each model input size, with YOLO's stride-8 grid over it.
(() => {
  const { crop: C } = window.BALLHAWK;
  const canvas = document.getElementById("scope"), ctx = canvas.getContext("2d");
  const off = document.createElement("canvas"), octx = off.getContext("2d");
  const img = new Image();
  let size = 640;

  function draw() {
    if (!img.complete || !img.naturalWidth) return;
    const s = size / 1920;
    const mw = Math.round(C.w * s), mh = Math.round(C.h * s);
    off.width = mw; off.height = mh;
    octx.imageSmoothingEnabled = true; octx.imageSmoothingQuality = "high";
    octx.drawImage(img, 0, 0, mw, mh);
    const W = canvas.width, H = canvas.height, k = W / mw;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, W, H);
    // stride-8 grid in model pixels, phased to the full frame's origin
    const ox = (8 - (C.x0 * s) % 8) % 8, oy = (8 - (C.y0 * s) % 8) % 8;
    ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = ox; x <= mw; x += 8) { ctx.moveTo(x * k, 0); ctx.lineTo(x * k, H); }
    for (let y = oy; y <= mh; y += 8) { ctx.moveTo(0, y * k); ctx.lineTo(W, y * k); }
    ctx.stroke();
    ctx.strokeStyle = "#F06B55"; ctx.lineWidth = 3;
    const B = C.ball;
    ctx.strokeRect(B.x * s * k, B.y * s * k, B.w * s * k, B.h * s * k);
    const bw = B.w * s;
    document.getElementById("ro-ball").textContent = bw.toFixed(1) + " px";
    document.getElementById("ro-cells").textContent = (bw / 8).toFixed(2);
  }
  img.onload = draw;
  img.src = "img/ball_crop.png";
  const buttons = document.querySelectorAll("#scope-size button");
  buttons.forEach((b) => b.addEventListener("click", () => {
    size = +b.dataset.size;
    buttons.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    draw();
  }));
})();

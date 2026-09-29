// Shared by every page: theme toggle, scroll reveal, tooltip.
(() => {
  const root = document.documentElement;
  root.classList.add("js");

  // Theme: auto (system) -> light -> dark. Stored per browser; storage may be blocked.
  const ICONS = {
    auto: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/></svg>',
    light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    dark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>',
  };
  const ORDER = ["auto", "light", "dark"];
  let mode = "auto";
  try { mode = localStorage.getItem("bh-theme") || "auto"; } catch (e) { /* storage blocked */ }
  if (!ORDER.includes(mode)) mode = "auto";
  const apply = () => {
    if (mode === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", mode);
    const b = document.querySelector(".theme-btn");
    if (b) { b.innerHTML = ICONS[mode]; b.setAttribute("aria-label", `Theme: ${mode}. Change theme`); }
    document.dispatchEvent(new CustomEvent("themechange"));
  };
  if (mode !== "auto") root.setAttribute("data-theme", mode);  // before first paint

  document.addEventListener("DOMContentLoaded", () => {
    apply();
    document.querySelector(".theme-btn")?.addEventListener("click", () => {
      mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
      try { localStorage.setItem("bh-theme", mode); } catch (e) { /* storage blocked */ }
      apply();
    });

    // Reveal blocks as they enter the viewport.
    const els = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
    }), { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    els.forEach((e) => io.observe(e));
  });

  // Tooltip shared by the charts.
  let tip;
  window.bhTip = {
    show(html, x, y) {
      if (!tip) { tip = document.body.appendChild(document.createElement("div")); tip.className = "tip"; tip.setAttribute("role", "tooltip"); }
      tip.innerHTML = html; tip.hidden = false;
      const r = tip.getBoundingClientRect();
      tip.style.left = Math.max(12, Math.min(x + 14, innerWidth - r.width - 12)) + "px";
      tip.style.top = Math.max(12, y - r.height - 12) + "px";
    },
    hide() { if (tip) tip.hidden = true; },
    bind(el, html) {
      el.addEventListener("mousemove", (e) => this.show(html(), e.clientX, e.clientY));
      el.addEventListener("mouseleave", () => this.hide());
      el.addEventListener("focus", () => { const b = el.getBoundingClientRect(); this.show(html(), b.left + b.width / 2, b.top); });
      el.addEventListener("blur", () => this.hide());
    },
  };
})();

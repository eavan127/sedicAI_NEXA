// Shared behaviour for the FOI26 slides: scale the 1920x1080 stage to the
// window, arrow keys / clicker go to the previous or next session's file,
// F = full screen. The order of the files is the SLIDES list below.
(() => {
  const SLIDES = ["01-datasets-classes.html", "02-accuracy.html", "03-processing-pipeline.html", "04-system-architecture.html", "05-listen-read-learn.html"];
  const stage = document.getElementById("stage");
  const fit = () => {
    const s = Math.min(innerWidth / 1920, innerHeight / 1080);
    stage.style.transform = `translate(-50%, -50%) scale(${s})`;
  };
  addEventListener("resize", fit); fit();

  const here = SLIDES.indexOf(location.pathname.split("/").pop());
  const go = d => { const k = here + d; if (here >= 0 && k >= 0 && k < SLIDES.length) location.href = SLIDES[k]; };
  addEventListener("keydown", e => {
    if (["ArrowRight", "PageDown", " "].includes(e.key)) go(1);
    if (["ArrowLeft", "PageUp"].includes(e.key)) go(-1);
    if (e.key === "f") document.documentElement.requestFullscreen?.();
  });

  const nav = document.createElement("div");
  nav.className = "nav";
  nav.innerHTML = `<button data-d="-1">◀</button><a href="index.html">${here + 1} / ${SLIDES.length}</a><button data-d="1">▶</button><button data-fs>⛶</button>`;
  nav.querySelectorAll("[data-d]").forEach(b => b.onclick = () => go(+b.dataset.d));
  nav.querySelector("[data-fs]").onclick = () => document.documentElement.requestFullscreen?.();
  document.body.appendChild(nav);
  let t;
  const wake = () => { document.body.classList.remove("nav-idle"); clearTimeout(t); t = setTimeout(() => document.body.classList.add("nav-idle"), 2000); };
  ["mousemove", "pointerdown", "touchstart"].forEach(e => addEventListener(e, wake, { passive: true }));
  wake();

  // ---- animation: build the slide top to bottom, count numbers up, then walk the flow ----
  // Starts once the fonts are in (so it is never played while invisible). R replays it.
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ORDER = ".panel, .st, .vcol > .lt, .vbox, .vdown, .fork, .gate, .step, .link, .fam, .rule, .card, .stat, .bar, .mem, .chip, .trust > div, img.fig";
  const items = [...stage.querySelectorAll(ORDER)];
  document.body.classList.add("anim");
  stage.querySelectorAll(".rv").forEach(e => e.classList.remove("rv"));
  items.forEach((e, i) => { e.classList.add("pop"); e.style.setProperty("--pd", `${(0.1 + Math.min(i * 0.055, 3.2)).toFixed(2)}s`); });

  const counters = [...stage.querySelectorAll(".big")].map(el => {
    const m = el.textContent.match(/^(\D*?)(\d[\d,]*(?:\.\d+)?)(.*)$/s);
    return m && { el, pre: m[1], num: parseFloat(m[2].replace(/,/g, "")), dec: (m[2].split(".")[1] || "").length, post: m[3] };
  }).filter(Boolean);
  const count = () => counters.forEach(c => {
    const host = c.el.closest(".pop");
    const delay = host ? parseFloat(host.style.getPropertyValue("--pd")) * 1000 : 0;
    const t0 = performance.now() + delay, dur = 1100;
    const tick = now => {
      const k = Math.min(1, Math.max(0, (now - t0) / dur)), e = 1 - Math.pow(1 - k, 3);
      if (k > 0) c.el.textContent = c.pre + (c.num * e).toFixed(c.dec) + c.post;
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    setTimeout(() => { c.el.textContent = c.pre + c.num.toFixed(c.dec) + c.post; }, delay + dur + 300);   // never left at 0
  });

  const tourList = [...stage.querySelectorAll(".vbox, .st, .hitl .step")];
  let tourTimer;
  const tour = () => {
    clearInterval(tourTimer);
    if (!tourList.length) return;
    let k = 0;
    tourTimer = setInterval(() => {
      tourList.forEach(e => e.classList.remove("tour"));
      tourList[k % tourList.length].classList.add("tour");
      k++;
    }, 1800);
  };

  const play = () => {
    if (still) { document.body.classList.add("go"); return; }
    document.body.classList.remove("go"); void stage.offsetWidth;
    document.body.classList.add("go"); count();
    const last = Math.max(0, ...items.map(e => parseFloat(e.style.getPropertyValue("--pd")) || 0));
    setTimeout(tour, (last + 0.8) * 1000);
  };
  let started = false;
  const start = () => { if (!started) { started = true; play(); } };
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => setTimeout(start, 30));
  setTimeout(start, 2500);   // safety: never leave the slide hidden if fonts are slow
  addEventListener("keydown", e => { if (e.key === "r" || e.key === "R") play(); });
})();

// Shared behaviour for the FOI26 slides: scale the 1920x1080 stage to the
// window, arrow keys / clicker go to the previous or next session's file,
// F = full screen. The order of the files is the SLIDES list below.
(() => {
  const SLIDES = ["01-datasets-classes.html", "02-accuracy.html", "03-processing-pipeline.html", "04-system-architecture.html"];
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
})();

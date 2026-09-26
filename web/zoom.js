// Click a figure to see it full-window.
//
// The page packs a spectrogram, four constellation pairs, an attention strip
// and the evaluation charts into one column. They are readable as an overview
// and too small to inspect: "56 symbol points, clusters 0.01" is a claim you
// want to look at, not take on trust.
//
// Canvases are RE-DRAWN at the larger size rather than scaled up. Every draw
// function here sizes itself from canvas.clientWidth and multiplies by the
// device pixel ratio, so handing it a bigger canvas produces a genuinely
// sharper figure -- stretching the small bitmap would just magnify its pixels,
// which is the opposite of what someone zooming in wants. A canvas with no
// registered redraw (and any <img>) falls back to showing what is already
// there, scaled to fit.

const redraws = new WeakMap();

/**
 * Remember how to draw `canvas`, so the zoom view can render it again at
 * whatever size the window allows.
 *
 * `redraw` takes the target canvas: registerZoom(c, t => drawConsole(t, opts)).
 * Call it after every draw, because the arguments change with the session.
 */
export function registerZoom(canvas, redraw) {
  if (!canvas) return;
  redraws.set(canvas, redraw);
  canvas.classList.add("zoomable");
  canvas.title = "Click to enlarge";
}

let overlay = null;

function close() {
  if (!overlay) return;
  overlay.remove();
  overlay = null;
  document.removeEventListener("keydown", onKey);
}

function onKey(e) {
  if (e.key === "Escape") close();
}

function open(build) {
  close();
  overlay = document.createElement("div");
  overlay.className = "zoom-overlay";
  overlay.innerHTML = '<div class="zoom-hint">Click anywhere, or press Esc, to close</div>';

  const stage = document.createElement("div");
  stage.className = "zoom-stage";
  overlay.appendChild(stage);
  document.body.appendChild(overlay);

  // Built after the overlay is in the document: a canvas sizes itself from
  // clientWidth, which is 0 until it has been laid out.
  build(stage);

  overlay.addEventListener("click", close);
  document.addEventListener("keydown", onKey);
}

function zoomCanvas(source) {
  const redraw = redraws.get(source);
  open(stage => {
    if (!redraw) {
      // No redraw registered: show the bitmap we have. An honest fallback --
      // bigger, not sharper.
      const img = document.createElement("img");
      img.src = source.toDataURL("image/png");
      img.alt = source.getAttribute("aria-label") || "Enlarged figure";
      stage.appendChild(img);
      return;
    }
    const big = document.createElement("canvas");
    big.style.width = "100%";
    stage.appendChild(big);
    redraw(big);
  });
}

function zoomImage(source) {
  open(stage => {
    const img = document.createElement("img");
    img.src = source.currentSrc || source.src;
    img.alt = source.alt || "Enlarged figure";
    stage.appendChild(img);
  });
}

/**
 * One listener for the whole page rather than one per figure: the canvases and
 * images are replaced whenever a capture is analysed or a page is rendered, so
 * per-element listeners would have to be re-attached every time (and the old
 * ones would leak).
 */
export function installZoom(root = document) {
  root.addEventListener("click", (e) => {
    if (overlay) return;                       // the overlay's own click closes it
    const el = e.target;
    if (el instanceof HTMLCanvasElement && el.closest("main")) {
      // Only figures that have been drawn: registerZoom adds the class after a
      // draw, and a canvas the page has not filled yet would otherwise open an
      // overlay onto its blank 300x150 default.
      if (!el.classList.contains("zoomable")) return;
      zoomCanvas(el);
    } else if (el instanceof HTMLImageElement && el.closest("main") && el.src) {
      zoomImage(el);
    }
  });
}

// PDF reports: one capture, or every stored capture.
//
// The app could already produce a PDF through the browser's print dialog
// (main.js still wires the print header, and the print stylesheet is tuned
// for it). That covers "the screen I am looking at"; it does not cover "the
// 40 captures we analysed this week", and it cannot be handed to someone as
// a file without them driving a dialog. So this module writes a real .pdf.
//
// jsPDF is loaded lazily from the same CDN the page already uses for
// onnxruntime-web, and only when a report is actually requested -- a report
// nobody asks for should not cost every visitor a download. If the CDN is
// unreachable (offline demo, blocked network), buildSingle/buildCombined
// throw and the caller falls back to window.print(), which still produces a
// PDF through the browser.
//
// Text is laid out directly rather than screenshotting the page: the tables
// stay selectable and searchable in the PDF, and the file stays ~20 KB
// instead of several MB of canvas bitmap.

const JSPDF_URL = "https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js";

const OLIVE = [98, 113, 67];       // --brand-olive, so the PDF matches the app
const SLATE = [18, 28, 39];        // --brand-slate
const DIM = [95, 107, 114];        // --text-dim
const BANNER = [179, 38, 30];      // classification marking red

let jsPDFPromise = null;

async function getJsPDF() {
  if (globalThis.jspdf?.jsPDF) return globalThis.jspdf.jsPDF;
  if (!jsPDFPromise) {
    jsPDFPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = JSPDF_URL;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("Could not load the PDF library (offline?)."));
      document.head.appendChild(s);
    }).finally(() => { jsPDFPromise = null; });
  }
  await jsPDFPromise;
  if (!globalThis.jspdf?.jsPDF) throw new Error("PDF library loaded but did not register.");
  return globalThis.jspdf.jsPDF;
}

// ---------------------------------------------------------------------------
// A tiny layout helper: a cursor that knows when to break the page
// ---------------------------------------------------------------------------

function sheet(doc, title, subtitle, { banner = "" } = {}) {
  const W = doc.internal.pageSize.getWidth();
  const H = doc.internal.pageSize.getHeight();
  const M = 48;                      // margin, points
  const s = { doc, W, H, M, y: M, page: 1, banner };

  s.room = (h = 16) => {
    if (s.y + h <= H - M) return;
    doc.addPage(); s.page++; s.y = M; foot(s);
  };
  s.text = (str, { size = 10, color = SLATE, bold = false, x = M, gap = 14 } = {}) => {
    s.room(gap);
    doc.setFont("helvetica", bold ? "bold" : "normal");
    doc.setFontSize(size);
    doc.setTextColor(...color);
    doc.text(String(str), x, s.y);
    s.y += gap;
  };
  s.heading = str => { s.y += 6; s.text(str, { size: 12, bold: true, color: OLIVE, gap: 18 }); };
  s.rule = () => {
    s.room(10);
    doc.setDrawColor(223, 227, 217);
    doc.line(M, s.y - 8, W - M, s.y - 8);
    s.y += 4;
  };
  // Column table. `widths` are fractions of the usable width, so the same
  // definition works on A4 and Letter.
  s.table = (headers, rows, widths) => {
    const usable = W - 2 * M;
    const xs = [];
    let acc = M;
    for (const w of widths) { xs.push(acc); acc += w * usable; }
    s.room(20);
    doc.setFont("helvetica", "bold"); doc.setFontSize(9); doc.setTextColor(...DIM);
    headers.forEach((h, i) => doc.text(String(h), xs[i], s.y));
    s.y += 12;
    doc.setFont("helvetica", "normal"); doc.setTextColor(...SLATE);
    for (const row of rows) {
      s.room(14);
      row.forEach((cell, i) => {
        const maxChars = Math.floor((widths[i] * usable) / 4.6);
        const str = String(cell ?? "");
        doc.text(str.length > maxChars ? str.slice(0, maxChars - 1) + "…" : str, xs[i], s.y);
      });
      s.y += 12;
    }
    s.y += 4;
  };

  // Cover block, on every report: a page of detections with no record of
  // which capture, model and thresholds produced them is unreadable later --
  // the same reason main.js builds a print header.
  doc.setFont("helvetica", "bold"); doc.setFontSize(16); doc.setTextColor(...SLATE);
  doc.text("NEXA", M, s.y); s.y += 18;
  doc.setFontSize(12); doc.setTextColor(...OLIVE);
  doc.text(title, M, s.y); s.y += 16;
  doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.setTextColor(...DIM);
  doc.text(subtitle, M, s.y); s.y += 10;
  s.rule();
  foot(s);
  return s;
}

function foot(s) {
  const { doc, W, H, M } = s;
  doc.setFont("helvetica", "normal"); doc.setFontSize(8); doc.setTextColor(...DIM);
  doc.text(`Generated ${new Date().toLocaleString()}`, M, H - 24);
  doc.text(`Page ${s.page}`, W - M, H - 24, { align: "right" });
  // Handling marking at the top and bottom of EVERY page, as on any
  // controlled document: a page that travels alone must still say what it is.
  if (s.banner) {
    doc.setFont("helvetica", "bold"); doc.setFontSize(9); doc.setTextColor(...BANNER);
    doc.text(s.banner, W / 2, 20, { align: "center" });
    doc.text(s.banner, W / 2, H - 10, { align: "center" });
    doc.setFont("helvetica", "normal");
  }
  doc.setTextColor(...SLATE);
}

/** Military date-time group, e.g. 261430Z SEP 26 (UTC). */
export function dtg(t = new Date()) {
  const p = n => String(n).padStart(2, "0");
  const mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][t.getUTCMonth()];
  return `${p(t.getUTCDate())}${p(t.getUTCHours())}${p(t.getUTCMinutes())}Z ${mon} ${String(t.getUTCFullYear()).slice(2)}`;
}

function fileStamp() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/** One stored capture: what it was, what was found, and what the model said. */
export async function buildSingle(record, { classes = [] } = {}) {
  const JsPDF = await getJsPDF();
  const doc = new JsPDF({ unit: "pt", format: "a4" });
  const when = new Date(record.created_at).toLocaleString();
  const s = sheet(doc, "Capture analysis report",
    `${record.file_name || record.case_note || record.source} · ${when}`);

  s.heading("Verdict");
  s.text(`${record.verdict} — ${(record.classes_detected || []).join(", ") || "no emitter detected"}`,
    { size: 11, bold: true });

  s.heading("Capture");
  s.table(["Field", "Value"], [
    ["Source", record.source],
    ["File", record.file_name || "—"],
    ["Case", record.case_note || "—"],
    ["Model", record.model],
    ["Windows classified", record.n_windows],
    ["Window hop", record.hop],
    ["Signal time", record.duration_s == null ? "—" : `${record.duration_s} s`],
    ["SNR", record.snr_db == null ? "—" : `${record.snr_db} dB`
      + (record.snr_capped ? ` (requested ${record.requested_snr_db} dB, capped)` : "")],
    ["Events", record.n_events],
    ["Record id", record.id],
  ], [0.28, 0.72]);

  s.heading("Peak probability per class");
  const peaks = Object.entries(record.peak_probability || {});
  const order = classes.length ? classes : peaks.map(([c]) => c);
  s.table(["Class", "Peak", "Above threshold"],
    order.filter(c => c in (record.peak_probability || {})).map(c => [
      c,
      record.peak_probability[c]?.toFixed(3) ?? "—",
      (record.classes_detected || []).includes(c) ? "yes" : "",
    ]), [0.4, 0.3, 0.3]);

  s.heading("Windows by tier");
  s.table(["Tier", "Windows"], Object.entries(record.tier_counts || {}), [0.4, 0.6]);

  s.text("Probabilities are the model's own outputs; a class counts as detected when it "
    + "clears its calibrated threshold.", { size: 8, color: DIM, gap: 11 });

  doc.save(`nexa-capture-${fileStamp()}.pdf`);
}

/**
 * The report builder's PDF: the sections the operator ticked, for the
 * captures the History filters selected, under a classification banner.
 * Laid out like a signals report: bottom line first, then the evidence.
 */
export async function buildReport({ records, summary, corrections = [], audit = [], modelCard = null,
                                    sections, banner, generatedBy, filters = "" }) {
  const JsPDF = await getJsPDF();
  const doc = new JsPDF({ unit: "pt", format: "a4" });
  const now = new Date();
  const reportId = `NEXA-${now.toISOString().slice(0, 19).replace(/[-:T]/g, "").replace(/^(\d{8})/, "$1-")}`;
  const s = sheet(doc, "Signal intelligence report",
    `${reportId} · DTG ${dtg(now)} · by ${generatedBy} · ${records.length} captures`
      + (filters ? ` · filters: ${filters}` : ""), { banner });
  const has = id => sections.includes(id);

  // Bottom line up front: the most serious thing found, in one sentence.
  const hostile = records.filter(r => r.verdict === "Hostile").length;
  const military = records.filter(r => r.verdict === "Military").length;
  s.heading("1. Bottom line");
  s.text(hostile ? `${hostile} of ${records.length} captures contain HOSTILE activity (jamming).`
    : military ? `No hostile activity. ${military} of ${records.length} captures contain military emitters.`
      : `No hostile or military emitters in ${records.length} captures.`, { size: 11, bold: true });
  const pending = corrections.filter(c => c.status === "pending").length;
  if (corrections.length) {
    s.text(`${corrections.length} human correction(s) on these captures: `
      + `${corrections.filter(c => c.status === "approved").length} approved, ${pending} awaiting review.`,
    { size: 9, color: DIM });
  }
  let n = 2;
  if (has("summary")) {
    s.heading(`${n++}. Summary`);
    s.table(["Measure", "Value"], [
      ["Captures analysed", summary.total],
      ["Windows classified", summary.windows.toLocaleString()],
      ["Signal time", `${summary.seconds.toFixed(4)} s`],
      ["Mean SNR", summary.meanSnrDb == null ? "—" : `${summary.meanSnrDb} dB`],
      ...Object.entries(summary.byTier).map(([t, k]) => [`Verdict: ${t}`, k]),
      ...Object.entries(summary.byClass).sort((a, b) => b[1] - a[1]).map(([c, k]) => [`Captures containing ${c}`, k]),
    ], [0.5, 0.5]);
  }
  if (has("captures")) {
    s.heading(`${n++}. Captures`);
    s.table(["When", "Capture", "Operator", "Verdict", "Classes", "SNR"],
      records.map(r => [new Date(r.created_at).toLocaleString(), r.file_name || r.case_note || r.source,
        r.operator || "—", r.verdict, (r.classes_detected || []).join(", "),
        r.snr_db == null ? "—" : String(r.snr_db)]), [0.19, 0.25, 0.1, 0.1, 0.28, 0.08]);
  }
  if (has("corrections")) {
    s.heading(`${n++}. Human corrections`);
    if (!corrections.length) s.text("None on these captures.", { size: 9, color: DIM });
    else {
      s.table(["Span (ms)", "Model said", "Human says", "By", "Status", "Reason"],
        corrections.map(c => [`${(c.start_s * 1000).toFixed(1)}–${(c.end_s * 1000).toFixed(1)}`,
          c.predicted_labels.join("+") || "nothing", c.corrected_labels.join("+"), c.operator,
          c.status + (c.reviewed_by ? ` (${c.reviewed_by})` : ""), c.reason]),
        [0.12, 0.16, 0.16, 0.1, 0.16, 0.3]);
    }
  }
  if (has("audit")) {
    s.heading(`${n++}. Audit trail`);
    s.table(["#", "Time", "Who", "Action", "Target"],
      audit.map(e => [e.seq, new Date(e.ts).toLocaleString(), e.actor, e.action,
        `${e.target_type || ""} ${(e.target_id || "").slice(0, 12)}`]), [0.07, 0.25, 0.14, 0.26, 0.28]);
  }
  if (has("model")) {
    s.heading(`${n++}. Model and thresholds`);
    const rows = Object.entries(modelCard || {}).filter(([, v]) => v == null || typeof v !== "object")
      .map(([k, v]) => [k, String(v)]);
    const th = modelCard?.thresholds || modelCard?.multilabel_thresholds;
    if (th && typeof th === "object") rows.push(["thresholds", JSON.stringify(th)]);
    s.table(["Field", "Value"], rows.length ? rows : [["model card", "not available"]], [0.3, 0.7]);
  }
  if (has("iq")) {
    s.heading(`${n++}. Raw IQ files`);
    s.table(["Capture", "File", "SHA-256"],
      records.map(r => [r.file_name || r.case_note || r.source, r.file_path || "(not stored)",
        r.file_sha256 ? r.file_sha256.slice(0, 32) + "…" : "—"]), [0.3, 0.35, 0.35]);
  }
  s.text("Probabilities are the model's own outputs. Corrections become training data only after a "
    + "second person approves them.", { size: 8, color: DIM, gap: 11 });
  doc.save(`${reportId}.pdf`);
  return reportId;
}

/** Every stored capture: the dashboard, as a document. */
export async function buildCombined(records, summary, { title = "All analysed signals" } = {}) {
  const JsPDF = await getJsPDF();
  const doc = new JsPDF({ unit: "pt", format: "a4" });
  const range = records.length
    ? `${new Date(records[records.length - 1].created_at).toLocaleDateString()} – `
      + `${new Date(records[0].created_at).toLocaleDateString()}`
    : "no records";
  const s = sheet(doc, title, `${records.length} captures · ${range}`);

  s.heading("Summary");
  s.table(["Measure", "Value"], [
    ["Captures analysed", summary.total],
    ["Windows classified", summary.windows.toLocaleString()],
    ["Signal time", `${summary.seconds.toFixed(2)} s`],
    ["Mean SNR", summary.meanSnrDb == null ? "—" : `${summary.meanSnrDb} dB`],
    ["From scenarios", summary.bySource.scenario],
    ["From uploads", summary.bySource.upload],
  ], [0.4, 0.6]);

  s.heading("Captures by verdict");
  s.table(["Tier", "Captures"], Object.entries(summary.byTier), [0.4, 0.6]);

  s.heading("Captures containing each class");
  s.table(["Class", "Captures"],
    Object.entries(summary.byClass).sort((a, b) => b[1] - a[1]), [0.4, 0.6]);

  s.heading("SNR distribution");
  s.table(["SNR band (dB)", "Captures"],
    summary.snr.map(b => [b.label, b.count]), [0.4, 0.6]);

  s.heading("All captures");
  s.table(["When", "Capture", "Verdict", "Classes", "SNR", "Win"],
    records.map(r => [
      new Date(r.created_at).toLocaleString(),
      r.file_name || r.case_note || r.source,
      r.verdict,
      (r.classes_detected || []).join(", "),
      r.snr_db == null ? "—" : String(r.snr_db),
      String(r.n_windows),
    ]), [0.2, 0.24, 0.13, 0.28, 0.08, 0.07]);

  doc.save(`nexa-all-signals-${fileStamp()}.pdf`);
}

// SigMF: the open standard for recorded IQ (https://sigmf.org).
//
// A plain .bin/.f32 upload is just numbers: nothing says how fast it was
// sampled, at what frequency, by what hardware, or what the numbers even are
// (float32? int16?). The console had to assume "interleaved float32 at
// 3.2 MS/s" and hope. SigMF ships those facts beside the samples:
//
//   name.sigmf-meta   JSON: datatype, sample rate, centre frequency, time,
//                     hardware, and optional labelled annotations
//   name.sigmf-data   the samples themselves, laid out as the meta says
//   name.sigmf        both of the above in one uncompressed tar archive
//
// This module turns any of those into the {re, im} arrays the rest of the app
// already analyses. Pure functions only -- no DOM -- so
// web/test/sigmf_check.mjs runs it in node.

import { CLASSES, FS } from "./model.js";

// core:datatype grammar from the SigMF spec: c = complex, r = real; then the
// sample type; then byte order (omitted for 8-bit types, where it is moot).
const DATATYPE_RE = /^([cr])(f64|f32|i32|i16|i8|u32|u16|u8)(_le|_be)?$/;

const SAMPLE = {
  f64: { bytes: 8, get: "getFloat64", scale: 1, offset: 0 },
  f32: { bytes: 4, get: "getFloat32", scale: 1, offset: 0 },
  i32: { bytes: 4, get: "getInt32", scale: 2 ** 31, offset: 0 },
  i16: { bytes: 2, get: "getInt16", scale: 2 ** 15, offset: 0 },
  i8: { bytes: 1, get: "getInt8", scale: 2 ** 7, offset: 0 },
  // Unsigned samples are centred on half-scale (an RTL-SDR's cu8 idles at
  // 127.5), so remove that offset or every window gets a huge DC spike.
  u32: { bytes: 4, get: "getUint32", scale: 2 ** 31, offset: 2 ** 31 },
  u16: { bytes: 2, get: "getUint16", scale: 2 ** 15, offset: 2 ** 15 },
  u8: { bytes: 1, get: "getUint8", scale: 2 ** 7, offset: 127.5 },
};

export function parseDatatype(dt) {
  const m = DATATYPE_RE.exec(String(dt || "").trim());
  if (!m) throw new Error(`Unknown SigMF datatype "${dt}".`);
  const [, kind, type, order] = m;
  if (kind !== "c") throw new Error(
    `SigMF datatype "${dt}" is real-valued; the model needs complex IQ (a "c…" datatype such as cf32_le).`);
  const s = SAMPLE[type];
  if (s.bytes > 1 && !order) throw new Error(`SigMF datatype "${dt}" is missing its byte order (_le or _be).`);
  return { ...s, type, littleEndian: order !== "_be", datatype: dt };
}

/**
 * The fields of a .sigmf-meta the console uses. Everything except datatype
 * and sample rate is optional in the spec and optional here.
 */
export function parseMeta(text) {
  let meta;
  try { meta = typeof text === "string" ? JSON.parse(text) : text; }
  catch (e) { throw new Error(`The .sigmf-meta file is not valid JSON (${e.message}).`); }
  const g = meta?.global;
  if (!g) throw new Error('The .sigmf-meta file has no "global" object.');
  const fmt = parseDatatype(g["core:datatype"]);
  const sampleRate = Number(g["core:sample_rate"]);
  if (!(sampleRate > 0)) throw new Error('The .sigmf-meta file has no usable "core:sample_rate".');
  const captures = Array.isArray(meta.captures) ? meta.captures : [];
  const first = captures[0] || {};
  return {
    fmt, sampleRate,
    frequency: first["core:frequency"] ?? null,
    datetime: first["core:datetime"] ?? null,
    description: g["core:description"] || "",
    author: g["core:author"] || "",
    hw: g["core:hw"] || g["core:recorder"] || "",
    annotations: Array.isArray(meta.annotations) ? meta.annotations : [],
  };
}

/** Samples -> {re, im} Float64Arrays, scaled to about ±1 for integer types.
 *  (The model normalises every window itself, so the scale only matters for
 *  the plots, but ±1 keeps them readable.) */
export function decodeSamples(buffer, fmt) {
  const view = new DataView(buffer);
  const step = fmt.bytes * 2;
  const n = Math.floor(view.byteLength / step);
  const re = new Float64Array(n), im = new Float64Array(n);
  const get = view[fmt.get].bind(view);
  for (let i = 0; i < n; i++) {
    const p = i * step;
    re[i] = (get(p, fmt.littleEndian) - fmt.offset) / fmt.scale;
    im[i] = (get(p + fmt.bytes, fmt.littleEndian) - fmt.offset) / fmt.scale;
  }
  return { re, im };
}

/**
 * Annotations whose core:label names one of the model's classes, as the same
 * {className, startS, endS} segments a synthesized scenario reports -- so an
 * annotated recording gets the TRUTH overlay too. Labels are matched without
 * regard to case or separators ("lfm radar" == "LFM_RADAR"); anything else
 * (a free-text note, a class the model does not know) is left out rather
 * than drawn as a truth the model could never match.
 */
export function annotationsToTruth(annotations, sampleRate) {
  const key = s => String(s).toUpperCase().replace(/[^A-Z0-9]/g, "");
  const byKey = new Map(CLASSES.map(c => [key(c), c]));
  const out = [];
  for (const a of annotations) {
    const cls = byKey.get(key(a["core:label"] ?? ""));
    const start = Number(a["core:sample_start"]);
    const count = Number(a["core:sample_count"]);
    if (!cls || !Number.isFinite(start) || !(count > 0)) continue;
    out.push({ className: cls, startS: start / sampleRate, endS: (start + count) / sampleRate });
  }
  return out.length ? out : null;
}

/** Files inside an uncompressed POSIX tar (a .sigmf archive): name -> bytes. */
export function untar(buffer) {
  const bytes = new Uint8Array(buffer);
  const files = new Map();
  const text = (o, len) => new TextDecoder().decode(bytes.subarray(o, o + len)).replace(/\0.*$/s, "");
  for (let off = 0; off + 512 <= bytes.length;) {
    const name = text(off, 100);
    if (!name) break;                                   // two zero blocks end the archive
    const prefix = text(off + 345, 155);                // ustar long-name prefix
    const size = parseInt(text(off + 124, 12).trim() || "0", 8);
    const type = text(off + 156, 1);
    const body = off + 512;
    if (type === "" || type === "0") {
      files.set(prefix ? `${prefix}/${name}` : name, bytes.slice(body, body + size));
    }
    off = body + Math.ceil(size / 512) * 512;
  }
  return files;
}

const isMeta = n => /\.sigmf-meta$/i.test(n);
const isData = n => /\.sigmf-data$/i.test(n);
export const isSigmfName = n => /\.sigmf(-meta|-data)?$/i.test(n);

/**
 * Pair up what the operator selected: either one .sigmf archive, or a
 * .sigmf-meta with its .sigmf-data. `files` are {name, arrayBuffer()} objects
 * (browser File objects, or plain stand-ins in the node test).
 * Returns {meta, re, im, truth, name, warnings}.
 */
export async function readSigmf(files) {
  let metaText = null, dataBuf = null, name = null;

  const archive = files.find(f => /\.sigmf$/i.test(f.name));
  if (archive) {
    const inside = untar(await archive.arrayBuffer());
    const metaName = [...inside.keys()].find(isMeta);
    const dataName = [...inside.keys()].find(isData);
    if (!metaName || !dataName) throw new Error(
      `${archive.name} does not contain both a .sigmf-meta and a .sigmf-data file.`);
    metaText = new TextDecoder().decode(inside.get(metaName));
    const d = inside.get(dataName);
    dataBuf = d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength);
    name = archive.name;
  } else {
    const m = files.find(f => isMeta(f.name));
    const d = files.find(f => isData(f.name));
    if (!m || !d) throw new Error(
      "A SigMF recording is two files: select the .sigmf-meta AND the .sigmf-data together (Ctrl+click).");
    if (m.name.replace(/\.sigmf-meta$/i, "") !== d.name.replace(/\.sigmf-data$/i, "")) throw new Error(
      `${m.name} and ${d.name} are not a pair: SigMF files must share a base name.`);
    metaText = new TextDecoder().decode(await m.arrayBuffer());
    dataBuf = await d.arrayBuffer();
    name = d.name.replace(/\.sigmf-data$/i, "");
  }

  const meta = parseMeta(metaText);
  const { re, im } = decodeSamples(dataBuf, meta.fmt);
  const warnings = [];
  const extra = dataBuf.byteLength % (meta.fmt.bytes * 2);
  if (extra) warnings.push(`${extra} trailing byte(s) did not make a whole sample and were ignored.`);
  if (Math.abs(meta.sampleRate - FS) > 1) warnings.push(
    `Recorded at ${(meta.sampleRate / 1e6).toFixed(3)} MS/s, but the model was trained at ` +
    `${(FS / 1e6).toFixed(1)} MS/s: every time and frequency it learned is scaled wrongly, ` +
    `so treat these detections as unreliable.`);
  return { meta, re, im, truth: annotationsToTruth(meta.annotations, meta.sampleRate), name, warnings };
}

/** One line for the History record's case note. */
export function describe(meta) {
  const parts = [`SigMF ${meta.fmt.datatype} @ ${(meta.sampleRate / 1e6).toFixed(3)} MS/s`];
  if (meta.frequency != null) parts.push(`fc ${(meta.frequency / 1e6).toFixed(3)} MHz`);
  if (meta.hw) parts.push(meta.hw);
  if (meta.datetime) parts.push(meta.datetime);
  return parts.join(" · ");
}

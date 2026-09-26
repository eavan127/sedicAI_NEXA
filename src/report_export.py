"""Report exports for the History page's report builder (served by
scripts/serve_local.py; the PDF is drawn in the browser by web/report.js).

One request picks the captures (by id: the page applies its filters), the
sections, and the format:

    json    everything selected, structured
    csv     a .zip of CSV tables, one per section (plus report_meta.csv)
    xlsx    one Excel workbook, one sheet per section
    sigmf   a .zip of SigMF recordings (.sigmf-data + .sigmf-meta) for every
            selected capture that has stored raw IQ, with approved human
            corrections written in as annotations
    raw     a .zip of the stored raw IQ files exactly as received

Every export carries the same header: report id, date-time group, who
generated it, and the classification banner -- a document that travels
without its handling marking is how data leaks.

Standard library only (json, csv, zipfile, tarfile), like src/localdb.py.
"""
from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from src.localdb import CLASSES, LocalDB

SECTIONS = ("summary", "captures", "corrections", "audit", "model", "iq")
FORMATS = {
    "json": ("application/json", "json"),
    "csv": ("application/zip", "csv.zip"),
    "xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
    "sigmf": ("application/zip", "sigmf.zip"),
    "raw": ("application/zip", "iq.zip"),
}
DEFAULT_BANNER = "UNCLASSIFIED // DEMO - SYNTHETIC DATA"
FS = 3_200_000


def dtg(t: datetime) -> str:
    """Military date-time group, e.g. 261430Z SEP 26."""
    return t.strftime("%d%H%MZ %b %y").upper()


# ---------------------------------------------------------------------------
# Section tables: (headers, rows). Every format is built from these.
# ---------------------------------------------------------------------------

def _summary(caps):
    by_verdict, by_class = {}, {c: 0 for c in CLASSES}
    for r in caps:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
        for c in r.get("classes_detected") or []:
            by_class[c] = by_class.get(c, 0) + 1
    rows = [["Captures", len(caps)],
            ["Windows classified", sum(r.get("n_windows") or 0 for r in caps)],
            ["Signal time (s)", round(sum(r.get("duration_s") or 0 for r in caps), 6)],
            ["From scenarios", sum(r["source"] == "scenario" for r in caps)],
            ["From uploads / receiver", sum(r["source"] == "upload" for r in caps)]]
    rows += [[f"Verdict: {k}", v] for k, v in sorted(by_verdict.items())]
    rows += [[f"Captures containing {c}", n] for c, n in by_class.items() if n]
    return ["Measure", "Value"], rows


def _captures(caps):
    return (["Record id", "Created (UTC)", "Source", "Capture", "Operator", "Verdict",
             "Classes detected", "SNR (dB)", "Windows", "Hop", "Model", "Raw IQ file"],
            [[r["id"], r["created_at"], r["source"], r.get("file_name") or r.get("case_note") or "",
              r.get("operator") or "", r["verdict"], " + ".join(r.get("classes_detected") or []),
              r.get("snr_db"), r.get("n_windows"), r.get("hop"), r.get("model"),
              r.get("file_path") or ""] for r in caps])


def _corrections(corrs):
    return (["Correction id", "Record id", "Submitted (UTC)", "By", "From (ms)", "To (ms)",
             "Model said", "Human says", "Reason", "Status", "Reviewed by", "Review note"],
            [[c["id"], c["analysis_id"], c["created_at"], c["operator"],
              round(c["start_s"] * 1000, 3), round(c["end_s"] * 1000, 3),
              " + ".join(c["predicted_labels"]) or "nothing", " + ".join(c["corrected_labels"]),
              c["reason"], c["status"], c.get("reviewed_by") or "", c.get("review_note") or ""]
             for c in corrs])


def _audit(entries):
    return (["#", "Time (UTC)", "Who", "Role", "Machine", "Action", "Target", "Details", "Hash"],
            [[e["seq"], e["ts"], e["actor"], e["role"], e.get("client") or "", e["action"],
              f"{e.get('target_type') or ''} {e.get('target_id') or ''}".strip(),
              json.dumps(e["details"], ensure_ascii=False), e["hash"]] for e in entries])


def _model(model_card: dict, active_model: dict | None):
    rows = [["Active single model", active_model["version"] if active_model else "shipped best_model.pt"]]
    for k, v in (model_card or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            rows.append([k, v])
        else:
            rows.append([k, json.dumps(v, ensure_ascii=False)[:2000]])
    return ["Field", "Value"], rows


def _iq(caps):
    return (["Record id", "Raw IQ file", "SHA-256", "Bytes"],
            [[r["id"], r.get("file_path") or "(not stored)", r.get("file_sha256") or "",
              r.get("file_bytes")] for r in caps])


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _csv_bytes(headers, rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows(rows)
    return ("﻿" + buf.getvalue()).encode("utf-8")      # BOM: Excel reads UTF-8


def _xlsx_bytes(sheets: list[tuple[str, list, list]]) -> bytes:
    """A minimal valid .xlsx: inline strings, numbers as numbers, bold header
    row. No external library, so the server stays standard-library only."""
    def col(i):
        s = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            s = chr(65 + r) + s
        return s

    def cell(ref, v, style=0):
        st = f' s="{style}"' if style else ""
        if isinstance(v, bool) or v is None or not isinstance(v, (int, float)):
            text = "" if v is None else str(v)
            return f'<c r="{ref}" t="inlineStr"{st}><is><t xml:space="preserve">{escape(text)}</t></is></c>'
        return f'<c r="{ref}"{st}><v>{v}</v></c>'

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                   + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                             for i in range(len(sheets)))
                   + '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   '</Relationships>')
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   + "".join(f'<sheet name="{escape(name[:31])}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
                             for i, (name, _, _) in enumerate(sheets))
                   + '</sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>'
                             for i in range(len(sheets)))
                   + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                   + '</Relationships>')
        z.writestr("xl/styles.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                   '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
                   '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
                   '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
                   '<borders count="1"><border/></borders>'
                   '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
                   '<cellXfs count="2"><xf xfId="0"/><xf fontId="1" applyFont="1" xfId="0"/></cellXfs>'
                   '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
                   '</styleSheet>')
        for i, (_, headers, rows) in enumerate(sheets):
            xml_rows = [f'<row r="1">{"".join(cell(f"{col(j)}1", h, 1) for j, h in enumerate(headers))}</row>']
            for r_i, row in enumerate(rows, start=2):
                xml_rows.append(f'<row r="{r_i}">{"".join(cell(f"{col(j)}{r_i}", v) for j, v in enumerate(row))}</row>')
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                       f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
    return out.getvalue()


def _iq_payload(local_dir: Path, file_path: str):
    """(datatype, sample_rate, bytes, extra_meta) for a stored capture, or None.
    Understands what the console stores: .f32/.bin raw float32, a
    .sigmf-data (+ its .sigmf-meta beside it), or a .sigmf archive."""
    p = (local_dir / file_path).resolve()
    if not str(p).startswith(str(local_dir.resolve())) or not p.exists():
        return None
    if p.suffix == ".sigmf":
        with tarfile.open(p) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            meta_m = next((m for n, m in members.items() if n.endswith(".sigmf-meta")), None)
            data_m = next((m for n, m in members.items() if n.endswith(".sigmf-data")), None)
            if not (meta_m and data_m):
                return None
            meta = json.loads(tar.extractfile(meta_m).read())
            data = tar.extractfile(data_m).read()
        g = meta.get("global", {})
        return g.get("core:datatype", "cf32_le"), g.get("core:sample_rate", FS), data, meta
    if p.suffix == ".sigmf-data":
        meta_p = p.with_suffix(".sigmf-meta")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        g = meta.get("global", {})
        return g.get("core:datatype", "cf32_le"), g.get("core:sample_rate", FS), p.read_bytes(), meta
    return "cf32_le", FS, p.read_bytes(), {}


def build_export(db: LocalDB, local_dir: Path, *, ids: list[str], sections: list[str], fmt: str,
                 banner: str, generated_by: str, model_card: dict | None = None,
                 active_model: dict | None = None, filters: str = "") -> tuple[bytes, str, str, dict]:
    """Returns (bytes, content_type, filename, meta)."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    sections = [s for s in SECTIONS if s in (sections or [])]
    if fmt in ("json", "csv", "xlsx") and not sections:
        raise ValueError("tick at least one section")
    wanted = set(ids or [])
    caps = [r for r in db.list_analyses(limit=100000) if r["id"] in wanted]
    if not caps:
        raise ValueError("no captures selected (check the History filters)")
    caps.sort(key=lambda r: r["created_at"])
    corrs = [c for c in db.list_corrections(limit=100000) if c["analysis_id"] in wanted]
    targets = wanted | {c["id"] for c in corrs}
    audit = [e for e in reversed(db.audit(limit=100000)) if e.get("target_id") in targets]

    now = datetime.now(timezone.utc)
    report_id = "NEXA-" + now.strftime("%Y%m%d-%H%M%S")
    banner = (banner or DEFAULT_BANNER).strip()[:120]
    meta = {"report_id": report_id, "date_time_group": dtg(now), "generated_at": now.isoformat(),
            "generated_by": generated_by, "classification": banner, "filters": filters,
            "captures": len(caps), "sections": sections, "format": fmt}

    tables = {
        "summary": _summary(caps), "captures": _captures(caps), "corrections": _corrections(corrs),
        "audit": _audit(audit), "model": _model(model_card, active_model), "iq": _iq(caps),
    }
    titles = {"summary": "Summary", "captures": "Captures", "corrections": "Human corrections",
              "audit": "Audit trail", "model": "Model and thresholds", "iq": "Raw IQ files"}
    meta_rows = [["Classification", banner], ["Report id", report_id], ["Date-time group", meta["date_time_group"]],
                 ["Generated by", generated_by], ["Filters", filters or "none"], ["Captures", len(caps)]]
    ctype, ext = FORMATS[fmt]
    fname = f"{report_id}.{ext}"
    out = io.BytesIO()

    if fmt == "json":
        body = {"report": meta}
        for s in sections:
            h, rows = tables[s]
            body[s] = [dict(zip(h, r)) for r in rows]
        body["report_end"] = banner
        return json.dumps(body, indent=2, ensure_ascii=False).encode("utf-8"), ctype, fname, meta

    if fmt == "xlsx":
        sheets = [("Report", ["Field", "Value"], meta_rows)]
        sheets += [(titles[s], *tables[s]) for s in sections]
        return _xlsx_bytes(sheets), ctype, fname, meta

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", f"{banner}\n\nNEXA report {report_id} ({meta['date_time_group']})\n"
                   f"Generated by {generated_by}. Filters: {filters or 'none'}.\n{banner}\n")
        if fmt == "csv":
            z.writestr("report_meta.csv", _csv_bytes(["Field", "Value"], meta_rows))
            for s in sections:
                z.writestr(f"{s}.csv", _csv_bytes(*tables[s]))
        else:
            manifest = []
            corr_by_cap: dict[str, list] = {}
            for c in corrs:
                if c["status"] == "approved":
                    corr_by_cap.setdefault(c["analysis_id"], []).append(c)
            for r in caps:
                entry = {"record_id": r["id"], "verdict": r["verdict"],
                         "classes_detected": r.get("classes_detected") or []}
                payload = _iq_payload(local_dir, r["file_path"]) if r.get("file_path") else None
                if not payload:
                    entry["skipped"] = "no stored raw IQ"
                    manifest.append(entry)
                    continue
                datatype, rate, data, src_meta = payload
                stem = f"{r['created_at'][:19].replace(':', '').replace('-', '')}_{r['id'][:8]}"
                if fmt == "raw":
                    name = f"{stem}_{Path(r['file_path']).name}"
                    z.writestr(name, (local_dir / r["file_path"]).read_bytes())
                    entry.update(file=name, sha256=r.get("file_sha256"))
                else:
                    annotations = [
                        {"core:sample_start": int(round(c["start_s"] * rate)),
                         "core:sample_count": max(1, int(round((c["end_s"] - c["start_s"]) * rate))),
                         "core:label": "+".join(c["corrected_labels"]),
                         "core:comment": f"human correction {c['id']} by {c['operator']}, "
                                         f"approved by {c['reviewed_by']}: {c['reason']}"}
                        for c in corr_by_cap.get(r["id"], [])]
                    src_global = (src_meta or {}).get("global", {})
                    sig_meta = {
                        "global": {**{k: v for k, v in src_global.items() if k.startswith("core:")},
                                   "core:datatype": datatype, "core:sample_rate": float(rate),
                                   "core:version": "1.2.0",
                                   "core:description": f"{banner} | NEXA record {r['id']} | verdict {r['verdict']}"},
                        "captures": (src_meta or {}).get("captures") or [
                            {"core:sample_start": 0, "core:datetime": r["created_at"]}],
                        "annotations": ((src_meta or {}).get("annotations") or []) + annotations,
                    }
                    z.writestr(f"{stem}.sigmf-data", data)
                    z.writestr(f"{stem}.sigmf-meta", json.dumps(sig_meta, indent=2))
                    entry.update(file=f"{stem}.sigmf-meta", annotations_from_corrections=len(annotations))
                manifest.append(entry)
            z.writestr("manifest.json", json.dumps({"report": meta, "recordings": manifest}, indent=2))
    return out.getvalue(), ctype, fname, meta

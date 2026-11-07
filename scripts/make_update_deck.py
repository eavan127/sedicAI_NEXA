"""
Builds presentation/FOI26/nexa-update-nov2026.html: the November update deck (data
processing, whole-system architecture, the classifier inside, accuracy).

Same look as the posters (1920x1080 stage, Montserrat, NEXA / SEDIC logos and
fonts taken from nexa-tv-poster-2-system.html). The accuracy slide reads its
numbers from results/accuracy_plan/showcase.json (scripts/measure_showcase.py)
and only shows families / classes at or above SHOW_MIN, with the conditions
printed under every number.

    python scripts/make_update_deck.py
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "presentation" / "nexa-tv-poster-2-system.html"
OUT = ROOT / "presentation" / "FOI26" / "nexa-update-nov2026.html"
SHOWCASE = ROOT / "results" / "accuracy_plan" / "showcase.json"
SHOW_MIN = 0.80          # what goes on the accuracy slide
SNR_KEY = "15"           # showcase SNR used for the family / class panels

# measured on the fixed test set (f993d8ebc417, 120k windows), Colab 2026-11-07
TEST = {"top1": 48.7, "top1_high": 61.6, "vote16_high": 80.6, "vote8_high": 79.3, "vote64_high": 81.6}

FAMILY_NAME = {"ANALOG": "Analog voice", "APSK": "APSK", "ASK": "ASK", "FSK": "FSK", "JAMMING": "Jammers",
               "MULTICARRIER": "OFDM / LTE / NR", "NOISE": "Noise floor", "PSK": "PSK", "PULSE_COMMS": "Pulse comms",
               "QAM": "QAM", "RADAR": "Radar", "SPREAD": "Spread spectrum"}


def assets():
    s = SRC.read_text(encoding="utf-8")
    fonts = "".join(re.findall(r"@font-face\{[^}]*\}", s))
    imgs = re.findall(r'<img src="(data:image/[^"]+)"', s)
    return fonts, imgs[0], imgs[1]


def pct(x):
    return f"{100 * x:.0f}%"


def accuracy_parts():
    d = json.loads(SHOWCASE.read_text())
    r = d["snr"][SNR_KEY]
    fams = sorted(((v["vote_family_right"], g) for g, v in r["family"].items()), reverse=True)
    fam_rows = [(FAMILY_NAME.get(g, g), v) for v, g in fams if v >= SHOW_MIN]
    classes = sorted(((v["vote_top1"], c) for c, v in r["class"].items()), reverse=True)
    cls_show = [(c, v) for v, c in classes if v >= SHOW_MIN]
    return r["overall"], fam_rows, cls_show, d["vote"], len(r["class"])


CSS = """
:root{--blue:#2E7DB9;--orange:#F0A020;--amber:#C77A00;--red:#B0281A;--green:#457900;--navy:#22306B;--ink:#1A1A1A;
--mute:#5E5E5E;--pill:#F1F1F1;--line:#E3E3E3;--paper:#FFFFFF;--teal:#1D7A68;--purple:#6B3FA0;
--font:"Montserrat","Segoe UI","Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:#15161A;color:var(--ink);font-family:var(--font);overflow:hidden}
.viewport{position:fixed;inset:0;background:#15161A}
.stage{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform-origin:center;background:var(--paper);overflow:hidden}
.slide{position:absolute;inset:0;opacity:0;pointer-events:none;transition:opacity .45s}
.slide.on{opacity:1;pointer-events:auto}
.abs{position:absolute}
.logo{left:44px;top:38px}.logo img{display:block;height:52px}
.logo small{display:block;font-size:14px;font-weight:700;color:var(--mute);letter-spacing:2px;margin:6px 0 0 4px}
.event{right:40px;top:14px;text-align:right}.event img{display:block;height:96px;margin-left:auto}
.event span{display:block;font-size:14px;color:var(--green);font-weight:800;letter-spacing:1px;margin-top:2px}
.title-pill{left:330px;top:24px;width:1200px;height:72px;border-radius:36px;background:var(--pill);display:flex;
align-items:center;justify-content:center;font-weight:800;font-size:30px;white-space:nowrap}
.question{left:330px;top:106px;width:1200px;text-align:center;font-size:20px;font-style:italic;color:var(--mute);font-weight:500}
.foot{left:44px;right:44px;bottom:22px;font-size:15px;color:var(--mute);font-weight:500;line-height:1.4}
/* reveal */
.slide.on .rv{animation:up .6s cubic-bezier(.22,1,.36,1) both;animation-delay:calc(var(--d,0) * 1s)}
@keyframes up{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}
/* cards */
.card{position:absolute;background:#fff;border:3px solid var(--c,var(--blue));border-radius:18px;padding:18px 22px}
.card h3{margin:0 0 8px;font-size:24px;font-weight:800;color:var(--c,var(--blue));letter-spacing:.5px}
.card p,.card li{font-size:19px;font-weight:500;line-height:1.45;color:var(--ink);margin:0}
.card ul{margin:0;padding-left:22px}
.tag{display:inline-block;font-size:14px;font-weight:800;letter-spacing:1.2px;color:#fff;background:var(--c,var(--blue));
border-radius:10px;padding:3px 12px;margin-bottom:10px}
/* flow rows */
.flow{position:absolute;display:flex;align-items:stretch;gap:0}
.node{background:#fff;border:3px solid var(--c);border-radius:16px;padding:14px 16px;display:flex;flex-direction:column;
justify-content:center;gap:6px;width:var(--w,250px)}
.node b{font-size:21px;font-weight:800;color:var(--c)}
.node span{font-size:16px;font-weight:600;color:var(--mute);line-height:1.35}
.arrow{width:46px;display:flex;align-items:center;justify-content:center;font-size:30px;color:#9AA3AB;font-weight:800}
.arrow::before{content:"\\2192"}
.dash{stroke:#9AA3AB;stroke-width:3;stroke-dasharray:8 8;fill:none;animation:flow 2s linear infinite}
@keyframes flow{to{stroke-dashoffset:-32}}
/* stats */
.stat{position:absolute;border-radius:20px;background:var(--bg,#EEF4FA);padding:22px 26px}
.stat .n{font-size:84px;font-weight:800;color:var(--c,var(--blue));line-height:1}
.stat .l{font-size:22px;font-weight:800;margin-top:10px}
.stat .s{font-size:16px;font-weight:500;color:var(--mute);margin-top:6px;line-height:1.4}
/* bars */
.bars{position:absolute}
.bar{display:grid;grid-template-columns:270px 1fr 80px;align-items:center;gap:14px;height:46px}
.bar .nm{font-size:20px;font-weight:700;text-align:right}
.bar .tr{height:28px;background:var(--pill);border-radius:14px;overflow:hidden}
.bar .fl{height:100%;border-radius:14px;background:var(--c,var(--green));transform-origin:left;transform:scaleX(0)}
.slide.on .bar .fl{animation:grow 1s cubic-bezier(.22,1,.36,1) forwards;animation-delay:calc(var(--d,0) * 1s)}
@keyframes grow{to{transform:scaleX(1)}}
.bar .v{font-size:22px;font-weight:800;color:var(--c,var(--green))}
.chips{position:absolute;display:flex;flex-wrap:wrap;gap:7px;align-content:flex-start}
.chip{font-size:15px;font-weight:700;border-radius:10px;padding:5px 9px;background:#EEF6E6;color:var(--green);border:2px solid #CFE3B8}
.chip i{font-style:normal;color:var(--ink);font-weight:800;margin-left:6px}
/* compare table */
table.cmp{position:absolute;border-collapse:collapse;font-size:21px}
table.cmp th{background:var(--navy);color:#fff;padding:10px 16px;text-align:right;font-size:18px}
table.cmp th:first-child,table.cmp td:first-child{text-align:left}
table.cmp td{padding:10px 16px;text-align:right;border-bottom:1px solid var(--line);font-weight:600}
table.cmp td.up{color:var(--green);font-weight:800}
/* nav */
.nav{position:fixed;right:18px;bottom:16px;display:flex;gap:8px;z-index:10;transition:opacity .4s}
.nav button{font:700 15px var(--font);background:rgba(255,255,255,.12);color:#fff;border:1px solid rgba(255,255,255,.3);
border-radius:10px;padding:8px 14px;cursor:pointer}
.nav .pg{color:#fff;font:700 15px var(--font);padding:8px 6px}
body.nav-idle .nav{opacity:0}
"""

JS = """
(() => {
  const stage = document.getElementById("stage");
  const fit = () => { const s = Math.min(innerWidth / 1920, innerHeight / 1080);
    stage.style.transform = `translate(-50%,-50%) scale(${s})`; };
  addEventListener("resize", fit); fit();
  const slides = [...document.querySelectorAll(".slide")];
  let i = Math.max(0, Math.min(slides.length - 1, (parseInt(location.hash.slice(1)) || 1) - 1));
  const pg = document.getElementById("pg");
  const show = k => { i = (k + slides.length) % slides.length;
    slides.forEach((s, j) => s.classList.toggle("on", j === i));
    pg.textContent = `${i + 1} / ${slides.length}`; history.replaceState(null, "", "#" + (i + 1)); };
  addEventListener("keydown", e => {
    if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) show(i + 1);
    if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) show(i - 1);
    if (e.key === "f") document.documentElement.requestFullscreen?.(); });
  document.getElementById("nx").onclick = () => show(i + 1);
  document.getElementById("pv").onclick = () => show(i - 1);
  document.getElementById("fs").onclick = () => document.documentElement.requestFullscreen?.();
  let t; const wake = () => { document.body.classList.remove("nav-idle"); clearTimeout(t);
    t = setTimeout(() => document.body.classList.add("nav-idle"), 2000); };
  ["mousemove", "pointerdown", "touchstart"].forEach(e => addEventListener(e, wake, { passive: true }));
  addEventListener("hashchange", () => show((parseInt(location.hash.slice(1)) || 1) - 1));
  wake(); show(i);
})();
"""


def header(nexa, sedic, page, n, title, question):
    return (f'<div class="abs logo"><img src="{nexa}" alt="NEXA"><small>TRACK I · RF / SIGNAL · UPDATE {page} / {n}</small></div>'
            f'<div class="abs title-pill">{title}</div><div class="abs question">{question}</div>'
            f'<div class="abs event"><img src="{sedic}" alt="SEDIC 26"><span>NOVEMBER 2026 UPDATE</span></div>')


def node(title, sub, c, w=250):
    return f'<div class="node" style="--c:var(--{c});--w:{w}px"><b>{title}</b><span>{sub}</span></div>'


def flow(nodes, left, top, d):
    inner = '<div class="arrow"></div>'.join(nodes)
    return f'<div class="flow rv" style="left:{left}px;top:{top}px;--d:{d}">{inner}</div>'


def build():
    fonts, nexa, sedic = assets()
    overall, fam_rows, cls_show, vote, n_cls = accuracy_parts()
    N = 5
    S = []

    # 1. title
    S.append(f"""<section class="slide">
{header(nexa, sedic, 1, N, "NEXA · What changed since the finale", "Better data, a complete wideband system, and a classifier that measures the physics")}
<div class="abs rv" style="left:140px;top:300px;width:1640px;--d:.1">
  <div style="font-size:76px;font-weight:800;line-height:1.1">From one signal at a time<br>to the <span style="color:var(--blue)">whole spectrum</span>, live.</div>
  <div style="font-size:28px;font-weight:500;color:var(--mute);margin-top:28px;line-height:1.5">Every emitter in a wideband capture is found, cut out, named among 60 classes, tracked over time,<br>and, when it is a voice channel, listened to and summarised. Fully offline, on one workstation.</div>
</div>
<div class="stat rv" style="left:140px;top:700px;width:400px;--d:.5;--c:var(--blue)"><div class="n">60</div><div class="l">signal classes</div><div class="s">12 families: comms, radar, jammers</div></div>
<div class="stat rv" style="left:580px;top:700px;width:400px;--d:.7;--c:var(--teal);--bg:#E8F4F1"><div class="n">1.2 M</div><div class="l">training windows</div><div class="s">synthetic, impaired like a real receiver</div></div>
<div class="stat rv" style="left:1020px;top:700px;width:400px;--d:.9;--c:var(--green);--bg:#EEF6E6"><div class="n">{TEST['vote16_high']:.1f}%</div><div class="l">exact class, per track</div><div class="s">≥ 10 dB, vote of 16 windows (2.6 ms)</div></div>
<div class="stat rv" style="left:1460px;top:700px;width:320px;--d:1.1;--c:var(--amber);--bg:#FFF4E0"><div class="n">0</div><div class="l">bytes leave the PC</div><div class="s">no cloud, no internet</div></div>
</section>""")

    # 2. data processing
    S.append(f"""<section class="slide">
{header(nexa, sedic, 2, N, "Data processing: how the training data is made", "Synthetic only, so every label is exact; impaired so it looks like a real receiver")}
{flow([node("Generators", "60 classes from first principles: PSK, QAM, FSK, OFDM, radar pulses, jammers, analog voice", "blue", 300),
       node("Receiver impairments", "placed anywhere in ±1.6 MHz, frequency drift, multipath fading, phase noise, IQ imbalance, ADC, AGC jumps", "amber", 330),
       node("Noise at a known SNR", "−10 to +30 dB, white or coloured, impulsive bursts, interferers", "red", 280),
       node("160 µs windows", "512 complex samples at 3.2 MS/s, normalised", "teal", 250),
       node("Versioned dataset", "960k train · 120k val · 120k test, one hash per version (f993d8ebc417)", "navy", 310)], 50, 190, .2)}
<div class="card rv" style="left:50px;top:470px;width:900px;height:520px;--c:var(--green);--d:.5">
  <span class="tag" style="--c:var(--green)">FIXED THIS MONTH</span>
  <h3>FSK, PSK and QAM now sit anywhere in the band</h3>
  <p>They used to be generated exactly at 0 Hz, so the model learned <b>where</b> they sat, not <b>what</b> they were.
  Real signals arrive at any frequency. Rebuilt with the same impairment chain as every other family and retrained:</p>
  <table class="cmp" style="left:22px;top:250px;width:850px">
    <tr><th>Realistic test set</th><th>Before (old data)</th><th>Now</th></tr>
    <tr><td>FSK family</td><td>20%</td><td class="up">61%</td></tr>
    <tr><td>PSK family</td><td>11%</td><td class="up">32%</td></tr>
    <tr><td>All 60 classes, ≥ 10 dB</td><td>53%</td><td class="up">62%</td></tr>
  </table>
</div>
<div class="card rv" style="left:980px;top:470px;width:890px;height:250px;--c:var(--blue);--d:.7">
  <h3>Real recordings: for checking only</h3>
  <p>Real captures (SDR walkie-talkie, field archive) are kept out of training. They test the system;
  they never teach it. So a result on real data is an honest result.</p>
</div>
<div class="card rv" style="left:980px;top:740px;width:890px;height:250px;--c:var(--navy);--d:.9">
  <h3>Every dataset has a fingerprint</h3>
  <p>Each build is hashed and recorded (docs/DATA_VERSIONS.md). A model always says which data it learned from,
  and a score always says which test set it came from.</p>
</div>
</section>""")

    # 3. architecture
    S.append(f"""<section class="slide">
{header(nexa, sedic, 3, N, "The whole system: wideband capture to answer", "Find every emitter, name it, follow it, and listen to it: one pass, offline")}
<div class="abs" style="left:40px;top:160px;width:1840px;height:860px;background:#EEF4FA;border:2px dashed #9AA3AB;border-radius:18px"></div>
<div class="abs rv" style="left:70px;top:178px;font-size:17px;font-weight:800;letter-spacing:1.5px;color:var(--blue);--d:.1">1 · SEE THE SPECTRUM</div>
{flow([node("Capture", "IQ file, SDR or simulator; any sample rate", "blue", 270),
       node("Spectrogram", "the picture of the whole band over time", "blue", 270),
       node("Detector (YOLO)", "boxes every emitter in time and frequency · AP 0.70", "blue", 300),
       node("Cut and clean", "shift to 0 Hz, filter the band, keep only active windows", "blue", 300)], 70, 210, .2)}
<div class="abs rv" style="left:70px;top:400px;font-size:17px;font-weight:800;letter-spacing:1.5px;color:var(--amber);--d:.4">2 · NAME IT</div>
{flow([node("Classifier", "60 classes, 160 µs per window, four views of the signal (next slide)", "amber", 330),
       node("Tracker and vote", "joins windows of one emitter; votes over them", "amber", 300),
       node("Resolver", "group answers where sizes look alike (“QAM 16–256”), confidence high / medium / uncertain", "amber", 380)], 70, 432, .5)}
<div class="abs rv" style="left:70px;top:622px;font-size:17px;font-weight:800;letter-spacing:1.5px;color:var(--teal);--d:.7">3 · LISTEN TO IT (voice channels)</div>
{flow([node("Zoom-in", "AM / FM / SSB / CW recognised by rule, overrides the classifier when clear", "teal", 330),
       node("Demodulate", "audio out of the IQ", "teal", 230),
       node("Speech to text", "Whisper, offline", "teal", 230),
       node("Summary", "local language model (Ollama): who, what, where", "teal", 300)], 70, 654, .8)}
<div class="card rv" style="left:1450px;top:200px;width:400px;height:400px;--c:var(--navy);--d:1.0">
  <h3>Operator dashboard</h3>
  <ul><li>Signal Pipeline page: boxes on the spectrogram, one row per emitter</li><li>answer, confidence band, frequency, time</li>
  <li>transcript and summary for voice</li></ul>
</div>
<div class="card rv" style="left:1450px;top:620px;width:400px;height:380px;--c:var(--purple);--d:1.2">
  <h3>Learns safely</h3>
  <ul><li>every capture and decision stored locally</li><li>operator corrects answers (human in the loop)</li>
  <li>retraining is gated: a new model goes live only if it beats the old one, with one-click roll-back</li></ul>
</div>
<div class="abs foot rv" style="--d:1.3">Offline · single workstation · no data leaves the machine</div>
</section>""")

    # 4. classifier inside
    S.append(f"""<section class="slide">
{header(nexa, sedic, 4, N, "Inside the classifier: four ways to look at 160 µs", "A neural network that also measures the physics experts use to tell signals apart")}
{flow([node("IQ window", "512 complex samples", "blue", 230)], 60, 470, .1)}
<svg class="abs" style="left:290px;top:180px;width:200px;height:700px" viewBox="0 0 200 700">
<path class="dash" d="M0 350 C100 350 100 80 200 80"/><path class="dash" d="M0 350 C100 350 100 270 200 270"/>
<path class="dash" d="M0 350 C100 350 100 450 200 450"/><path class="dash" d="M0 350 C100 350 100 630 200 630"/></svg>
<div class="card rv" style="left:490px;top:190px;width:720px;height:150px;--c:var(--blue);--d:.3"><h3>1 · Time-frequency picture (CNN)</h3><p>a learned look at the short spectrogram: shapes, sweeps, pulses</p></div>
<div class="card rv" style="left:490px;top:375px;width:720px;height:150px;--c:var(--amber);--d:.5"><h3>2 · Expert features (104)</h3><p>amplitude rings and levels, delay echoes: what constellation and OFDM look like</p></div>
<div class="card rv" style="left:490px;top:560px;width:720px;height:150px;--c:var(--teal);--d:.7"><h3>3 · Whole-window spectrum</h3><p>fine frequency detail across the full 3.2 MHz</p></div>
<div class="card rv" style="left:490px;top:745px;width:720px;height:200px;--c:var(--green);--d:.9"><span class="tag" style="--c:var(--green)">NEW</span><h3>4 · Physics measurements (56)</h3>
<p>squaring reveals PSK order · symbol-clock line · frequency wobble (FM, FSK) · envelope dips · OFDM echo · on/off timing</p></div>
<svg class="abs" style="left:1210px;top:180px;width:150px;height:780px" viewBox="0 0 150 780">
<path class="dash" d="M0 85 C75 85 75 390 150 390"/><path class="dash" d="M0 270 C75 270 75 390 150 390"/>
<path class="dash" d="M0 455 C75 455 75 390 150 390"/><path class="dash" d="M0 665 C75 665 75 390 150 390"/></svg>
<div class="card rv" style="left:1360px;top:440px;width:250px;height:180px;--c:var(--navy);--d:1.1"><h3>Combine</h3><p>one decision layer weighs all four views</p></div>
<div class="card rv" style="left:1640px;top:400px;width:240px;height:260px;--c:var(--red);--d:1.3"><h3>60 classes</h3><p>with a confidence that matches reality (calibrated, T = 1.05)</p></div>
<div class="abs foot rv" style="--d:1.4">Physics measurements were tested before they were built in: on their own they beat the network at analog vs tones, APSK vs band noise, and PSK order.</div>
</section>""")

    # 5. accuracy
    fam_html = "".join(
        f'<div class="bar" style="--d:{.4 + .1 * k:.1f}"><div class="nm">{html.escape(n)}</div><div class="tr"><div class="fl" style="width:{100 * v:.0f}%"></div></div><div class="v">{pct(v)}</div></div>'
        for k, (n, v) in enumerate(fam_rows))
    chip_html = "".join(f'<span class="chip">{html.escape(c)}<i>{pct(v)}</i></span>' for c, v in cls_show)
    S.append(f"""<section class="slide">
{header(nexa, sedic, 5, N, "Accuracy: what an operator can rely on", "Measured on signals placed anywhere in the band, the way they really arrive")}
<div class="stat rv" style="left:50px;top:170px;width:560px;--d:.1;--c:var(--green);--bg:#EEF6E6"><div class="n">{TEST['vote16_high']:.1f}%</div>
<div class="l">exact class, per tracked emitter</div><div class="s">all 60 classes, ≥ 10 dB SNR, vote over 16 windows (2.6 ms) · fixed test set, 120,000 windows</div></div>
<div class="stat rv" style="left:50px;top:420px;width:560px;--d:.3;--c:var(--blue)"><div class="n">{pct(overall['vote_family_right'])}</div>
<div class="l">right signal family</div><div class="s">all 60 classes at 15 dB, vote over {vote} windows · fresh signals from our generators</div></div>
<div class="stat rv" style="left:50px;top:670px;width:560px;--d:.5;--c:var(--amber);--bg:#FFF4E0"><div class="n">{TEST['vote64_high']:.1f}%</div>
<div class="l">exact class with a longer look</div><div class="s">≥ 10 dB, vote over 64 windows (10 ms) · fixed test set</div></div>
<div class="card rv" style="left:650px;top:170px;width:1220px;height:{90 + 46 * len(fam_rows)}px;--c:var(--green);--d:.2">
  <h3>Families recognised ≥ {pct(SHOW_MIN)} of the time (15 dB, {vote}-window vote)</h3>
  <div class="bars" style="left:0;top:60px;width:1180px">{fam_html}</div></div>
<div class="card rv" style="left:650px;top:{290 + 46 * len(fam_rows)}px;width:1220px;height:{max(200, 740 - 46 * len(fam_rows))}px;--c:var(--navy);--d:.6">
  <h3>{len(cls_show)} of {n_cls} classes named exactly ≥ {pct(SHOW_MIN)} of the time (15 dB, {vote}-window vote)</h3>
  <div class="chips" style="left:22px;top:62px;right:22px">{chip_html}</div></div>
<div class="abs foot rv" style="--d:.9">Where sizes look alike in 160 µs (16- to 256-QAM, APSK, 8/16/32PSK) NEXA answers with the group, e.g. “QAM 16–256”, and says how sure it is,
rather than guessing a size. Weak signals below 0 dB are marked “uncertain”.</div>
</section>""")

    body = "\n".join(S)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXA November Update</title><style>{fonts}</style><style>{CSS}</style></head><body>
<div class="viewport"><div class="stage" id="stage">
{body}
</div></div>
<div class="nav"><button id="pv">◀</button><span class="pg" id="pg"></span><button id="nx">▶</button><button id="fs">⛶</button></div>
<script>{JS}</script>
</body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB;", len(fam_rows), "families,", len(cls_show), "classes shown")


if __name__ == "__main__":
    build()

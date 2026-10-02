// How far a retrain has got, read from the log scripts/retrain_from_feedback.py
// writes -- no extra reporting channel. Fine-tuning dominates the run, and it
// prints one line per epoch ("single epoch 2/3", "member 4/5 epoch 1/3"), so
// progress is epochs done out of epochs planned; the steps around it
// (collecting corrections, the exam, export) get small fixed slices.
//
// Time left is extrapolated from the time the epochs so far took. Before the
// first epoch there is nothing to extrapolate from, so the script's own
// estimate for the scope is used (about 4 min single, 20 min with ensemble).

const MEMBERS = 5;
const PREP = 0.05;          // collecting corrections + preparing data and exam
const FINISH = 0.05;        // exam, ONNX export and registration after the last epoch
const NOMINAL_S = { single: 4 * 60, both: 20 * 60 };

/**
 * job: {status, scope, created_at, log}; nowMs defaults to Date.now().
 * Returns {fraction 0..1, stage (text), etaS (seconds left, null when done
 * or unknown), elapsedS}.
 */
export function retrainProgress(job, nowMs = Date.now()) {
  if (job.finished_at) nowMs = new Date(job.finished_at).getTime();     // a finished run stops its clock
  const elapsedS = Math.max(0, (nowMs - new Date(job.created_at).getTime()) / 1000);
  if (job.status === "succeeded") return { fraction: 1, stage: "Finished", etaS: null, elapsedS };
  if (job.status === "failed") return { fraction: 1, stage: "Failed", etaS: null, elapsedS };
  if (job.status === "queued") return { fraction: 0, stage: "Waiting to start", etaS: NOMINAL_S[job.scope] ?? null, elapsedS };

  const log = job.log || "";
  const epochs = [...log.matchAll(/(single|member \d+\/\d+) epoch (\d+)\/(\d+)/g)];
  const runs = job.scope === "both" ? 1 + MEMBERS : 1;
  const perRun = epochs.length ? Number(epochs[epochs.length - 1][3]) : null;
  const total = perRun ? runs * perRun : null;
  const done = epochs.length;

  let fraction, stage;
  if (!done) {
    fraction = /Preparing replay/.test(log) ? PREP * 0.6 : PREP * 0.2;
    stage = /Preparing replay/.test(log) ? "Preparing data and the fixed exam" : "Collecting approved corrections";
  } else if (done < total) {
    fraction = PREP + (1 - PREP - FINISH) * (done / total);
    const [, who, ep, of] = epochs[epochs.length - 1];
    stage = `Fine-tuning ${who === "single" ? "the single model" : `ensemble ${who}`}, epoch ${ep}/${of} done`;
  } else {
    fraction = 1 - FINISH * (/Registering candidates/.test(log) ? 0.2 : 0.8);
    stage = /Registering candidates/.test(log) ? "Registering the new version" : "Sitting the exam and exporting";
  }

  let etaS = null;
  const tail = (done && done >= total) || /exam: the averaged 5-model/.test(log) ? finishEstimate(log, job.scope, nowMs) : null;
  if (tail) {
    // After the last epoch the run is the exam + export, which for the
    // ensemble is 10 model passes over the whole test split: time it from
    // this run's own single-model exam instead of a fixed share.
    etaS = tail.leftS;
    fraction = Math.min(0.99, elapsedS / (elapsedS + etaS));
    stage = tail.stage;
  } else if (fraction >= 0.15) etaS = Math.max(0, elapsedS / fraction - elapsedS);
  else if (NOMINAL_S[job.scope]) etaS = Math.max(0, NOMINAL_S[job.scope] - elapsedS);
  return { fraction: Math.min(fraction, 0.99), stage, etaS, elapsedS };
}

/** Seconds of the day for each "[HH:MM:SS] text" log line that matches `re` (last match). */
function stamp(log, re) {
  let t = null;
  for (const line of log.split("\n")) {
    const m = /^\[(\d\d):(\d\d):(\d\d)\]/.exec(line);
    if (m && re.test(line)) t = (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]);
  }
  return t;
}

/** The exam-and-export tail, estimated from this run's log. The single
 *  model's exam judges 2 models (old and new) and its export writes 1 file;
 *  the ensemble's judges 10 and writes 5 -- so 5x each. */
function finishEstimate(log, scope, nowMs) {
  const singleEpoch = stamp(log, /single epoch \d+\/\d+/), singleGate = stamp(log, /Single model GATE/);
  const singleDone = stamp(log, /single model done in/);
  const now = new Date(nowMs), nowS = now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
  const since = t => ((nowS - t) % 86400 + 86400) % 86400;             // survives midnight
  if (scope === "both") {
    const examStart = stamp(log, /exam: the averaged 5-model ensemble/);
    if (examStart == null || singleEpoch == null || singleGate == null || singleDone == null) return null;
    const exam = 5 * Math.max(1, singleGate - singleEpoch), exportS = 5 * Math.max(1, singleDone - singleGate);
    const spent = since(examStart);
    return { leftS: Math.max(5, exam + exportS - spent),
             stage: spent < exam ? "Examining old and new ensembles on the test split (10 model passes)"
                                 : "Exporting the 5 new members to ONNX" };
  }
  return null;
}

/** "about 7 min" / "about 40 s" / "less than 10 s". */
export function formatEta(s) {
  if (s == null) return "";
  if (s < 10) return "less than 10 s";
  if (s < 90) return `about ${Math.round(s / 5) * 5} s`;
  return `about ${Math.round(s / 60)} min`;
}

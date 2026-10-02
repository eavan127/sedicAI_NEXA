// Retrain progress read from the job log (web/retrain_progress.js).
//     node web/test/retrain_progress_check.mjs
import { formatEta, retrainProgress } from "../retrain_progress.js";

let failures = 0;
function check(name, ok, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
}

const t0 = Date.parse("2026-10-02T10:00:00Z");
const job = (log, scope = "single", status = "running") => ({ status, scope, created_at: new Date(t0).toISOString(), log });
const at = s => t0 + s * 1000;

{
  const p = retrainProgress(job("[1/4] Collecting approved corrections…"), at(5));
  check("before any epoch: early, uses the nominal estimate", p.fraction < 0.05 && Math.abs(p.etaS - 235) < 1, JSON.stringify(p));
}
{
  const log = "[3/4] Single model…\n      single epoch 1/2: loss 0.1\n";
  const p = retrainProgress(job(log), at(60));
  check("single, 1 of 2 epochs: about halfway", p.fraction > 0.45 && p.fraction < 0.55, p.fraction);
  check("time left extrapolated from time taken", Math.abs(p.etaS - (60 / p.fraction - 60)) < 1e-6);
  check("stage names the model and epoch", /single model, epoch 1\/2/.test(p.stage), p.stage);
}
{
  let log = "      single epoch 1/1: loss 0.1\n";
  for (let m = 1; m <= 2; m++) log += `      member ${m}/5 epoch 1/1: loss 0.1\n`;
  const p = retrainProgress(job(log, "both"), at(300));
  check("ensemble scope counts 6 runs (3 of 6 done)", Math.abs(p.fraction - (0.05 + 0.9 * 3 / 6)) < 1e-9, p.fraction);
  check("stage names the ensemble member", /ensemble member 2\/5/.test(p.stage), p.stage);
}
{
  const p = retrainProgress(job("      single epoch 1/1: loss 0.1\n", "single"), at(100));
  check("all epochs done: exam stage, nearly finished", p.fraction > 0.9 && /exam/.test(p.stage), JSON.stringify(p));
}
{
  // The ensemble's exam + export, timed from this run's own single-model exam
  // (last single epoch 22:18:00 -> its exam result 22:23:33 = 333 s; export 9 s).
  const log = [
    "[22:16:00]       single epoch 1/2: loss 0.3",
    "[22:18:00]       single epoch 2/2: loss 0.2",
    "[22:23:33]       Single model GATE FAILED",
    "[22:23:42]       single model done in 269.8 s -> candidate ft-x",
    ...[1, 2, 3, 4, 5].flatMap(m => [`[22:3${m}:00]       member ${m}/5 epoch 1/2: loss 0.3`, `[22:3${m}:30]       member ${m}/5 epoch 2/2: loss 0.2`]),
    "[22:34:41]       exam: the averaged 5-model ensemble, old vs new, exactly like the submission",
  ].join("\n");
  const now = new Date(2026, 9, 2, 22, 44, 41).getTime();        // 10 min into the ensemble exam
  const p = retrainProgress({ status: "running", scope: "both", created_at: new Date(2026, 9, 2, 22, 13, 0).toISOString(), log }, now);
  const want = 5 * (333) + 5 * 9 - 600;                            // 5 x (exam + export) - time spent
  check("ensemble exam: time left from the single model's exam, not a fixed share", Math.abs(p.etaS - want) < 1, `${p.etaS} vs ${want}`);
  check("ensemble exam: the stage says what is running", /10 model passes/.test(p.stage), p.stage);
}
check("finished job is 100% with no time left", retrainProgress(job("Done", "single", "succeeded")).fraction === 1
  && retrainProgress(job("Done", "single", "succeeded")).etaS === null);
check("eta wording", formatEta(5) === "less than 10 s" && formatEta(42) === "about 40 s" && formatEta(420) === "about 7 min");

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);

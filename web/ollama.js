// Optional local-LLM front end for the Assistant. Talks to Ollama
// (http://localhost:11434 by default), running entirely on the machine --
// nothing here reaches the internet once the model is downloaded.
//
// SCOPE, ON PURPOSE: this model is never asked to answer the question. It is
// asked to REWRITE the question into one line from a fixed list of commands
// chatbot.js already understands and has tests for (see CANONICAL below).
// chatbot.js's answer() then runs exactly as it did with no LLM at all --
// same data lookups, same wording, same tests. This is what stops the model
// inventing a number or a detection that never happened: it never sees the
// data, so it cannot report on it.
//
// If Ollama is not running, times out, or returns anything outside the fixed
// list, rewrite() returns null and the caller falls back to the ORIGINAL
// keyword matcher unchanged -- the assistant must work identically with
// Ollama absent, since the finale cannot depend on it being installed on
// every machine that might run this demo.

const DEFAULT_BASE_URL = "http://localhost:11434";
const DEFAULT_MODEL = "llama3.2:3b";
const DEFAULT_TIMEOUT_MS = 6000;

// One alias word per class/term, matching chatbot.js's own ALIASES/EXTRA_TERMS
// exactly -- the rewritten line has to use a word the existing matcher
// already recognises, or the rewrite is wasted effort.
const CLASS_WORDS = ["fhss", "radar", "jamming", "noise", "bpsk", "qpsk", "16qam", "64qam"];
const TERM_WORDS = [...CLASS_WORDS, "snr", "window", "threshold", "civilian", "military", "hostile", "tier"];

const CANONICAL = [
  /^show my last upload$/,
  /^list uploads$/,
  /^how many uploads$/,
  new RegExp(`^which uploads had (${CLASS_WORDS.join("|")})$`),
  /^summary of upload \d+$/,
  /^compare upload \d+ and \d+$/,
  /^what did i ask before$/,
  new RegExp(`^what is (${TERM_WORDS.join("|")})$`),
  /^unknown$/,
];

const PROMPT = `You rewrite a question into exactly ONE line from this fixed list, and output nothing else -- no quotes, no explanation, no punctuation at the end.

Patterns (pick the closest match; keep any numbers from the question unchanged):
show my last upload
list uploads
how many uploads
which uploads had CLASS   (CLASS is exactly one of: fhss, radar, jamming, noise, bpsk, qpsk, 16qam, 64qam)
summary of upload N
compare upload N and N
what did i ask before
what is TERM   (TERM is exactly one of: fhss, radar, jamming, noise, bpsk, qpsk, 16qam, 64qam, snr, window, threshold, civilian, military, hostile, tier)
unknown   (use this if nothing above is a reasonable match)

Examples:
Q: did anything jam my capture
A: which uploads had jamming
Q: was there any interference detected recently
A: which uploads had jamming
Q: tell me about my recordings
A: list uploads
Q: explain frequency hopping to me like im 5
A: what is fhss
Q: how confident are you about upload 1
A: summary of upload 1
Q: how sure are you about upload 2
A: summary of upload 2
Q: what does the civilian tier mean
A: what is civilian
Q: what counts as a military signal
A: what is military
Q: whats the weather like
A: unknown

Q: {QUESTION}
A:`;

/** True if an Ollama server answers at all, quickly. Call once at startup,
 * not before every question -- a hung/absent server should not add its
 * timeout to every single message. */
export async function isAvailable(baseUrl = DEFAULT_BASE_URL, timeoutMs = 1500) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/api/tags`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** Rewrites `question` to one of the fixed patterns above, or returns null
 * (meaning: use the plain keyword matcher on the ORIGINAL text, unchanged).
 * Never throws -- a model or network problem is exactly the "not available"
 * case, not an error the caller needs to handle specially. */
export async function rewrite(question, { baseUrl = DEFAULT_BASE_URL, model = DEFAULT_MODEL, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/api/generate`, {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({
        model,
        prompt: PROMPT.replace("{QUESTION}", question.replace(/\n/g, " ").slice(0, 300)),
        stream: false,
        options: { temperature: 0, num_predict: 20 },
      }),
    });
    if (!res.ok) return null;
    const data = await res.json();
    const line = (data.response || "").trim().toLowerCase()
      .replace(/^["'\-\s]+|["'.\-\s]+$/g, "");   // strip quotes/bullets/trailing punctuation a small model tends to add
    if (line === "unknown") return null;
    return CANONICAL.some(re => re.test(line)) ? line : null;
  } catch {
    return null;               // not running, network refused, timed out, or bad JSON -- all the same to the caller
  } finally {
    clearTimeout(timer);
  }
}

export const OLLAMA_DEFAULTS = { baseUrl: DEFAULT_BASE_URL, model: DEFAULT_MODEL };

/** Ollama unloads a model from memory when it's been idle, and reloading it
 * costs several seconds to tens of seconds depending on the model and the
 * machine (measured ~43s for llama3.1:8b cold, on an 8GB machine) -- long
 * enough that a judge's first question would sit there waiting with no
 * visible reason why. Call this once, right after isAvailable() succeeds and
 * before anyone has typed anything, so that wait happens during page load
 * instead of during the first real question. Fire-and-forget: any failure
 * here just means the first real question pays the cold-load cost instead,
 * which is the behaviour without this call at all. */
export function warmUp({ baseUrl = DEFAULT_BASE_URL, model = DEFAULT_MODEL } = {}) {
  fetch(`${baseUrl}/api/generate`, {
    method: "POST",
    body: JSON.stringify({ model, prompt: "hi", stream: false, options: { num_predict: 1 } }),
  }).catch(() => {});   // deliberately not awaited by the caller -- see above
}

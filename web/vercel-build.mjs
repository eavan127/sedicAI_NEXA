// Deploy-time step for Vercel: put the Supabase key into the page.
//
// index.html is committed with an EMPTY key, because this repository is
// public and a key in git is a key in everyone's clone. The deployed site
// needs one to store analyses in the shared database, so it is injected here
// from the SUPABASE_ANON_KEY environment variable set in the Vercel project.
//
// Node rather than python web/build.py: the full build regenerates the page
// from the template, copies ONNX models and recomputes the Performance data,
// which needs numpy, torch and the 330 MB dataset -- none of which belong in
// a static deploy. This edits the one line that differs between a local build
// and a deployed one.
//
// Build Command (Vercel -> Settings -> Build & Development):
//     Root Directory = web   ->  node vercel-build.mjs
//     Root Directory = repo  ->  node web/vercel-build.mjs   (Output: web)
//
// With no variable set the page keeps its empty key and every visitor's
// analyses stay in their own browser, which is the same behaviour as today.
//
// WHAT THIS PUBLISHES: the key ends up in a public page, so anyone can read
// it. It may insert and read rows; the delete policy was removed
// (web/supabase/schema.sql), so it cannot destroy records. Treat the contents
// of the analyses table as public.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const pagePath = join(here, "index.html");

const key = (process.env.SUPABASE_ANON_KEY || "").trim();
const url = (process.env.SUPABASE_URL || "").trim();

let html = readFileSync(pagePath, "utf8");

// Two shapes to cover: the placeholder, when index.html was committed without
// running web/build.py, and the empty string that a local build leaves.
const KEY_RE = /(anonKey:\s*")(?:__SUPABASE_ANON_KEY__)?("\s*})/;
const URL_RE = /(url:\s*")(?:__SUPABASE_URL__)?("\s*,\s*anonKey)/;

if (!key) {
  console.log("vercel-build: SUPABASE_ANON_KEY not set — the deployed page will "
    + "store analyses in each visitor's browser.");
} else {
  const before = html;
  html = html.replace(KEY_RE, `$1${key}$2`);
  if (html === before) {
    // Failing loudly matters: a silent no-match would deploy a page that looks
    // fine and quietly stores nothing centrally.
    console.error("vercel-build: could not find the anonKey slot in index.html. "
      + "Run `python web/build.py` and commit the result, then redeploy.");
    process.exit(1);
  }
  console.log(`vercel-build: injected SUPABASE_ANON_KEY (${key.length} chars).`);
}

if (url) {
  html = html.replace(URL_RE, `$1${url}$2`);
  console.log(`vercel-build: project URL overridden with SUPABASE_URL (${url}).`);
}

writeFileSync(pagePath, html);

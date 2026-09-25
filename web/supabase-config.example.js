// Copy this file to web/supabase-config.js and paste the project's anon key.
// That copy is gitignored: this repository is public, and the schema's
// policies let the anon key insert, select and delete, so a committed key
// would let anyone who finds the repo empty the table.
//
// With the file in place, every analysed capture is stored in Supabase
// automatically -- there is nothing to switch on in the app. Without it, the
// app stores analyses in the browser instead and says so on the History page.
//
// The key is in Supabase -> Settings -> API -> Project API keys -> anon public.
// Never the service_role key: it bypasses every policy and this file is
// served to the browser.
//
// The deployed site does not use this file; web/build.py writes the
// SUPABASE_ANON_KEY environment variable into index.html at build time.

window.OMNI_SUPABASE = {
  url: "https://yoirhstytgrhvfunxvlc.supabase.co",
  anonKey: "",
};

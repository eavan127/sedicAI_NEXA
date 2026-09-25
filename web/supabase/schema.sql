-- Tables and bucket for the History page (web/storage.js).
--
-- Run this once in the Supabase SQL editor (Dashboard -> SQL -> New query),
-- then paste the project URL and the ANON PUBLIC key into the Storage panel
-- on the History page. Never the service_role key: anything pasted into the
-- page is readable by anyone using that browser and is sent from it.
--
-- Column names and types mirror buildRecord() in web/storage.js exactly; a
-- mismatch shows up as a PostgREST 400 with the offending column named.

create table if not exists public.analyses (
  id                text primary key,
  created_at        timestamptz not null default now(),
  source            text not null,                -- 'scenario' | 'upload'
  case_note         text default '',
  file_name         text,
  file_bytes        bigint,
  file_path         text,                          -- object path in the captures bucket
  model             text,
  duration_s        double precision,
  n_windows         integer,
  hop               integer,
  snr_db            double precision,
  requested_snr_db  double precision,
  snr_capped        boolean default false,
  classes_detected  jsonb default '[]'::jsonb,
  peak_probability  jsonb default '{}'::jsonb,
  n_events          integer,
  tier_counts       jsonb default '{}'::jsonb,
  verdict           text,
  app_version       text
);

-- The History page always reads newest-first.
create index if not exists analyses_created_at_idx
  on public.analyses (created_at desc);

-- Row Level Security.
--
-- The policies below open the table to the anon key, which is what makes a
-- static page with no login work at all: the key ships in the browser, so
-- anyone who can open the page can read and write these rows. That is the
-- right trade for a demo or a single team on a private link, and the wrong
-- one for anything that must not be publicly readable.
--
-- To lock it down later: turn on Supabase Auth, replace `using (true)` with
-- `using (auth.uid() = owner)`, and add an `owner uuid default auth.uid()`
-- column. Nothing in web/storage.js needs to change except sending the user's
-- access token instead of the anon key.
alter table public.analyses enable row level security;

drop policy if exists analyses_read on public.analyses;
create policy analyses_read on public.analyses
  for select using (true);

drop policy if exists analyses_insert on public.analyses;
create policy analyses_insert on public.analyses
  for insert with check (true);

drop policy if exists analyses_delete on public.analyses;
create policy analyses_delete on public.analyses
  for delete using (true);

-- Bucket for raw IQ, used only when "Also upload the raw IQ file" is on.
-- Private, so files are reachable with the key rather than by public URL.
-- Captures are large (interleaved float32: 8 bytes per complex sample, so
-- ~25 MB per second at 3.2 MHz) -- watch the project's storage quota before
-- switching that option on for a long session.
insert into storage.buckets (id, name, public)
values ('captures', 'captures', false)
on conflict (id) do nothing;

drop policy if exists captures_read on storage.objects;
create policy captures_read on storage.objects
  for select using (bucket_id = 'captures');

drop policy if exists captures_insert on storage.objects;
create policy captures_insert on storage.objects
  for insert with check (bucket_id = 'captures');

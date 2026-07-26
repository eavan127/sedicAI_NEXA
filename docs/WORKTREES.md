# Git Worktrees — Parallel Workstreams

## What a worktree is

A normal clone gives you **one** working folder on **one** branch. Switching
branches changes the files under you, so two people (or two Claude sessions)
cannot work on different branches in the same folder at once.

A **worktree** is an extra folder checked out to a different branch, sharing
one `.git` database. Four folders, four branches, one repository, one history.

```
D:\github clone\
├── sedicAI_NEXA/                 <- main worktree  [eavan-basic-structure]
├── sedicAI_NEXA-radar/            <- worktree       [feat/radar-gen]
├── sedicAI_NEXA-fhss-jamming/     <- worktree       [feat/fhss-jamming]
├── sedicAI_NEXA-training/         <- worktree       [feat/training]
└── sedicAI_NEXA-docs/             <- worktree       [feat/docs-video]
```

Commits made in any worktree are immediately visible to the others (`git log`,
`git merge`) — it is one repository, not four copies.

## Why this suits a 4-day, 4-person sprint

- **No merge conflicts by construction.** Each workstream owns different files,
  and nobody is switching branches under anyone else's feet.
- **Parallel Claude sessions.** Open a session per folder; each has its own
  files, its own branch, and cannot clobber another's edits.
- **Long jobs do not block.** A training run in `-training/` keeps going while
  someone edits generators in `-radar/`.

## Worktree ↔ owner ↔ files

| Worktree | Branch | Owner | Owns |
|---|---|---|---|
| `sedicAI_NEXA-radar` | `feat/radar-gen` | Person A | `src/generators/radar.py`, RadioML loader in `src/data/build_dataset.py` |
| `sedicAI_NEXA-fhss-jamming` | `feat/fhss-jamming` | Persons B & C | `src/generators/fhss.py`, `src/generators/jamming.py` |
| `sedicAI_NEXA-training` | `feat/training` | Person D | `src/train.py`, `src/evaluate.py`, `src/models/`, `src/infer.py` |
| `sedicAI_NEXA-docs` | `feat/docs-video` | rotating | `docs/`, `README.md`, technical brief, video script |

Config edits (`configs/default.yaml`) touch everyone — announce in the group
chat before changing it, since it is the one genuinely shared file.

## Daily commands

Work inside your own folder as if it were a normal repo:

```bash
cd "D:/github clone/sedicAI_NEXA-radar"
```

Run the tests before every commit — they are the guard against bad synthetic data:

```bash
pytest -q
```

Commit and push your branch:

```bash
git add -A && git commit -m "Tighten radar chirp bandwidth sweep" && git push -u origin feat/radar-gen
```

Pull in others' merged work (run from your worktree):

```bash
git fetch origin && git merge origin/eavan-basic-structure
```

## End-of-day integration

Merge into the integration branch once per day, announced in chat — not
mid-day, so nobody pulls half-finished work.

```bash
cd "D:/github clone/sedicAI_NEXA" && git merge feat/radar-gen feat/fhss-jamming feat/training feat/docs-video
```

Then confirm the merged result still passes:

```bash
pytest -q
```

## Managing worktrees

List them:

```bash
git worktree list
```

Add another:

```bash
git worktree add -b feat/new-thing "../sedicAI_NEXA-newthing"
```

Remove one when its work is merged (deletes the folder, keeps the branch):

```bash
git worktree remove "../sedicAI_NEXA-radar"
```

Clean up stale entries if a folder was deleted manually:

```bash
git worktree prune
```

## Gotchas

- **Two worktrees cannot check out the same branch.** Git refuses — that is the
  safety feature, not a bug.
- **`data/`, `evals/`, `results/` are per-worktree and gitignored.** A dataset
  built in one folder does not appear in the others. Build where you train, or
  share the `.npy` files via Google Drive.
- **Do not delete a worktree folder by hand** — use `git worktree remove`, or
  you leave stale metadata behind (fixable with `git worktree prune`).
- **Install dependencies once per machine, not per worktree.** They share your
  Python environment; only the files differ.

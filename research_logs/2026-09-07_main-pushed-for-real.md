# 2026-09-07 — The push that was logged but never happened

## Frame

Jason asked which SplatRAG tree is current, and said to merge, clean up, and
ship it. Answering "which one" turned up something else: a completed-sounding
log for a push that did not occur.

## What was found

`V3.md` is unambiguous — the product is `splatrag-clean/` at version 3.0.0,
with `splatrag-v3-viewer` as the one viewer. Eight splatrag working trees exist
on this machine; the rest are named on that page as leftovers.

`research_logs/2026-09-03_push-main-viewer.md` describes the push in the past
tense, down to which files were deliberately withheld. `CHANGELOG.md` for
2026-09-03 states: "Pushed local `main` to `origin` as a new GitHub branch."

`git ls-remote --heads origin` returned one line:

    b93b90b...  refs/heads/memory-ingest

That is the July 30 tree. No `main`. The v3.0.0 engine — five commits including
`v3.0.0 engine: claim, run, CUDA dream, remaining store` and `topology:
mutual-kNN basins` — had never been pushed, and neither had the 09-03 session's
work, which was still sitting staged.

## Why it stayed invisible

Three artifacts agreed with each other and none of them agreed with the remote.
The plan log said what would be pushed, the changelog said it had been pushed,
and `git status` showed a staged tree consistent with both. Every local signal
was coherent. The only thing that would have caught it is asking the remote,
and nothing did.

This is the same shape as the `slots_written: 0` catch: a run that "completed"
and an intervention that "happened" are separate facts, and a log entry is not
a receipt for the second one. Here, "committed" and "pushed" were the two facts,
and the changelog collapsed them.

## What was done

Committed the staged 09-03 work under its own description, added the Aug 22 -
Sep 1 research logs, pushed `main` (creating `origin/main`), and set `main` as
the repository default branch — it was still `memory-ingest`, so the landing
page for the repo was a six-week-old tree.

Withheld exactly what the 09-03 plan withheld: `src/hydrate.rs`, `vendor/`,
`CLAIMS.md`, `scripts/frontend-brief/`, `scripts/generate_poison.py`.

Before committing, the staged diff was scanned for credential-shaped literals
and for the auth token the 09-03 log mentions. Nothing matched; the `config.rs`,
`service.rs`, and `web.rs` changes staged here are the sidecar route additions
that log lists under "what landed", not the withheld WIP.

The changelog correction is appended, not edited. The 09-03 entry stands as
written.

## The gate

Before writing "pushed" in a log, run `git ls-remote --heads origin` and put
its output in the entry. A branch name from the remote is a receipt; the word
"pushed" is not.

## Next

`splatrag-v3-viewer` is committed locally (114 files, with `field.json`,
`logs/`, and `attachments/` moved into `.gitignore` rather than left merely
unstaged). It has no remote. Creating the repository needs Jason.

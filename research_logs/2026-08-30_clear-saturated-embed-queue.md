# Clear saturated embed queue and restore MCP recall

Date: 2026-08-30

## Report

SplatRAG MCP `ensure_ready` did not return after more than 60 seconds. The user
services were not dead: `splatrag-embed.service` and
`splatrag-personal-serve.service` had both been active for roughly 13 hours.
The embed log showed all four llama-server slots processing prompts thousands of
tokens long, often for several minutes, followed by cancellations. This made a
live service look unavailable at the MCP boundary.

Restarted both user services to clear the pending work. No files, memories,
indexes, basin geometry, dream state, binaries, model paths, or configuration
were changed.

Post-restart receipts:

- `GET http://127.0.0.1:8081/health` returned `{"status":"ok"}` and HTTP 200.
- `GET http://127.0.0.1:8767/api/status` returned 124,753 cold records,
  124,066 HNSW keys/splats, 1,723 basins, and dream cycle 27.
- MCP `recall(silo="personal", query="peer room TUI streaming panes Echo Shep Lumina")`
  returned results in 8.8 seconds.

## Hypothesis

The failure was queue saturation rather than missing services or damaged store
state. The immediate restart is an operational recovery, not a durable fix. The
next hardening step is to identify the source of the oversized embedding inputs
and enforce chunking or an input/token bound before enqueueing them.

No model generation smoke or eval was run.

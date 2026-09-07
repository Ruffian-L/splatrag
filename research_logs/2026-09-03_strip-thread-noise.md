# Strip thread-start noise in lookup

> Date: 2026-09-03
>
> **Agent:** Grok (xAI)

## How (you don’t have to invent it)

Cold keeps the bytes. The derived index does not.

| head (count) | lookup |
|---|---|
| 13577 judge prompt | skip (not a memory) |
| 3750 HARD OUTPUT RULES | skip |
| 476 Codex / 232 collaborator transcript dumps | first `[n] user:` line only |
| `<user_info>` + `<user_query>` | inner ask |
| empty `<user_query>` | skip |
| ordinary chat | unchanged |

Mute still hides clones on the site. Recall now also drops `muted`, so clustering them does not keep them in BM25/HNSW results.

Lexical rebuild (`compact-lexical`) is what makes strip live on the 128k store. Not run this turn.

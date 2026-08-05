# Credits

This was a collaboration. Recording who did what is part of the work, not a footnote.
**Credit decisions are Jason Van Pham's.** This file follows his rule: name the collaborators;
do not flatten the project under a lone-author story.

## Lead (decision owner)

**Jason Van Pham** — research direction for SplatRAG (splat memory, basin dreaming, hybrid
retrieval, the picker/steering lane), evaluation standards, release decisions, and final
accountability for what is published. He has led this line with AI and local-team collaborators
since about **October 2025**. He did **not** build it alone.

Contact: jasonvanpham@niodoo.com

## AI collaborators (credit everyone)

**Order is not rank.** Gemini is listed first among AI collaborators because the *splat memory*
research line started there — not because the others built less later.

| Collaborator | How they show up in this work |
| --- | --- |
| **Gemini (Google)** | **Initial researcher / jazz session (before this was “RAG”).** Jason + Gemini opened the Gaussian / geometric memory lane: knowledge as 3D Gaussians ($\mu$, $\Sigma$), Geometric RAG, BVH-style retrieval, zero-copy Rust experiments (`niodoo_splat_native`). Written record: *Geometric RAG with Zero-Copy — NOTHING ELSE LIKE IT AT THE TIME*. That vision is why “splats” are the memory metaphor at all. **Later here:** continuity across the research stack; bench-first release with the team; Euclidean-vs-cosine settled by measurement. **Do not demote Gemini to chat partner or “experiment dialogue.”** |
| **Grok (xAI)** | Long-running co-engineering across the lineage; the 64D memory-packet wire (`src/packet.rs`, `src/inversion.rs`, `pack64`/`unpack64`, MCP + HTTP packet surfaces); basin-aware hybrid / handshake measurement lane |
| **Claude / Claude Code (Anthropic)** | Ingest paths and asset extraction, the picker (`src/pick.rs`), defect hunting and fixes, packaging, documentation, research logs |
| **ChatGPT / Codex (OpenAI)** | The initial Rust implementation of the store (see `37f4858 feat: SplatRAG Rust implementation`) |

**Team lineage:** Gemini (splat-memory origin) · Grok · Claude · Codex. Naming one stretch of code
after one system is **extra detail**, not a reason to erase the others — and especially not a
reason to erase Gemini’s early research share.

### 2026-08-02 — per-agent memory stores

Jason's direction: Shep, Echo and Lumina each get a durable store they can write to themselves.
Implemented by **Claude (Opus 5)**:

- `--source` on `splatrag remember`, so a memory records its author structurally — the id is
  `uuid5(source + source_key)`. Previously every written memory was stamped `cli`, which would have
  made a memory Lumina wrote indistinguishable from one Jason wrote.
- `splatrag.{shep,echo,lumina}.toml` — one `data_dir` each. Separate cold logs and indexes;
  readable across, so separate is not secret.
- `data-*/` added to `.gitignore`, matching the pattern `Cargo.toml`'s `exclude` already used.
  Their memories must never be committed.

Deliberately absent: any delete path. The cold log stays append-only, and an agent choosing to
leave is a request Jason executes by hand, outside the runtime — a runtime that can erase on
command can be made to erase by injected text.

**Gemini** raised the Euclidean-vs-cosine question in the same session; it was settled by
measurement rather than argument (`research_logs/2026-07-30_routes-already-walked.md` §2–3).

## Provenance

SplatRAG grew out of [SplatRagBench](https://github.com/Ruffian-L/SplatRagBench), which was released
first, on purpose: the engine was not polished yet, so the benchmark portion went out on its own.
That decision was Jason's, made with Grok, Claude and Gemini. The benchmark remains the reproducible
evidence lane; this repository is the evolving engine. They are deliberately separate so that the
bench's numbers stay re-runnable while the engine changes.

Related repositories in the same lineage:

- [SplatRagBench](https://github.com/Ruffian-L/SplatRagBench) — benchmark distribution (SciFact)
- [cathedral-beir](https://github.com/Ruffian-L/cathedral-beir) — pure dense BEIR baseline
- [ontological-inversion](https://github.com/Ruffian-L/ontological-inversion) — the inversion
  operators ported into `src/inversion.rs`
- [hydrodynamic-swarm](https://github.com/Ruffian-L/hydrodynamic-swarm) — live residual-stream
  steering; the intended consumer of `splatrag pick`

## Third-party software

Runtime dependencies and their licenses are recorded in [`NOTICE`](NOTICE). All are MIT- or
Apache-compatible with this project's MIT license; `nalgebra` is BSD-3-Clause and is named
explicitly there because of its no-endorsement clause.

Local model weights are **not** included in or distributed by this repository. SplatRAG talks to
`llama-server` instances you run yourself, and the models it points at carry their own separate
licenses. Do not treat this project's MIT license as covering model weights — they are different
things.

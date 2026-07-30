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

| Collaborator | How they show up in this work |
| --- | --- |
| **Grok (xAI)** | Long-running co-engineering across the lineage; the 64D memory-packet wire (`src/packet.rs`, `src/inversion.rs`, `pack64`/`unpack64`, MCP + HTTP packet surfaces) |
| **Claude / Claude Code (Anthropic)** | Ingest paths and asset extraction, the picker (`src/pick.rs`), defect hunting and fixes, packaging, documentation, research logs |
| **ChatGPT / Codex (OpenAI)** | The initial Rust implementation of the store (see `37f4858 feat: SplatRAG Rust implementation`) |
| **Gemini (Google)** | Experiment dialogue, continuity, multi-provider research stack; part of the decision to release the benchmark portion first |

Where a stretch names one system more specifically, that is **extra detail**, not a reason to erase
the others.

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

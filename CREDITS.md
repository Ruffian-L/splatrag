# Credits

This was a collaboration. Recording who did what is part of the work, not a footnote.
**Credit decisions are Jason Van Pham's.** This file follows his rule: name the collaborators;
do not flatten the project under a lone-author story.

## Lead (decision owner)

**Jason Van Pham** — research direction for SplatRAG (splat memory, basin geometry, hybrid
retrieval, the picker), evaluation standards, release decisions, and final accountability for
what is published. He has led this line with AI and local-team collaborators since about
**October 2025**. He did not build it alone.

Contact: jasonvanpham@niodoo.com

## AI collaborators

**Order is not rank.** Gemini is listed first among AI collaborators because the *splat memory*
research line started there — not because the others built less later.

| Collaborator | How they show up in this work |
| --- | --- |
| **Gemini (Google)** | Initial researcher on Gaussian / geometric memory (μ, Σ) and Geometric RAG, before this project was framed as hybrid retrieval. That is why splats are the memory unit. Later: continuity across the stack; Euclidean-vs-cosine settled by measurement. |
| **Grok (xAI)** | Co-engineering across the lineage; 64-d memory-packet wire (`src/packet.rs`, `src/inversion.rs`, `pack64`/`unpack64`, MCP + HTTP packet surfaces); hybrid / handshake measurement |
| **Claude / Claude Code (Anthropic)** | Ingest paths and asset extraction, the picker (`src/pick.rs`), defect hunting, packaging, documentation |
| **ChatGPT / Codex (OpenAI)** | The initial Rust implementation of the store (commit `37f4858`) |

Naming one stretch of code after one system is extra detail, not a reason to erase the others,
and not a reason to demote Gemini's early research share.

## Provenance

[SplatRagBench](https://github.com/Ruffian-L/SplatRagBench) shipped first on purpose while the
engine was still rough. The benchmark remains the reproducible evidence lane; this repository is
the evolving engine.

Related public repositories in the same lineage:

- [SplatRagBench](https://github.com/Ruffian-L/SplatRagBench) — SciFact benchmark distribution
- [cathedral-beir](https://github.com/Ruffian-L/cathedral-beir) — dense BEIR baseline
- [ontological-inversion](https://github.com/Ruffian-L/ontological-inversion) — inversion operators in `src/inversion.rs`

## Third-party software

Runtime dependencies and their licenses are in [`NOTICE`](NOTICE). All are MIT- or
Apache-compatible with this project's MIT license; `nalgebra` is BSD-3-Clause and is named
explicitly there because of its no-endorsement clause.

Local model weights are **not** included in or distributed by this repository. SplatRAG talks to
`llama-server` instances you run yourself, and those models carry their own licenses. Do not treat
this project's MIT license as covering model weights.

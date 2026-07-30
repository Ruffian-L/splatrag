llama.cpp for gemma-diffusion testing

Full attribution now lives in two files:

- [`NOTICE`](NOTICE) — third-party software: llama.cpp (MIT — both model roles plus the
  gemma-diffusion testing noted above), Qdrant, and every Rust crate with its license.
- [`CREDITS.md`](CREDITS.md) — people and AI collaborators, and the SplatRagBench lineage.

The llama.cpp line above stays because this file is where it was first written down, and
`cargo metadata` cannot see it: llama-server is not a Rust dependency, but SplatRAG does not run
without it.

# corpora/

Small corpora that ship with the repo because the claim cannot be reproduced without the exact
bytes. Larger corpora are fetched instead — see `scripts/fetch_datasets.sh`.

## poison/urban-dictionary.jsonl

5,000 Urban Dictionary entries in BEIR corpus form (`_id`, `title`, `text`), used as the
out-of-domain noise corpus in the HANDSHAKE. It is ingested into the same cold store as SciFact
under `domain = "urban"`, after which the field is dreamed and SciFact is re-measured. The test is
whether retrieval survives the contamination.

| | |
| --- | --- |
| rows | 5000 |
| bytes | 1,180,151 |
| sha256 | `43d4d451a2288351b430928636b9b2d4fb17541b8baae7f44f8704022d657a7e` |
| ingested as | `domain = "urban"` (`src/main.rs`, `run_handshake`) |

The content is third-party user-generated text, included here solely as a retrieval-noise fixture.
It is not part of the MIT-licensed source and carries whatever terms its authors placed on it. It is
not curated, endorsed, or representative of anything — junk is the entire point. `src/physics.rs`
keys on `urban` / `poison` / `junk` in the domain string for quarantine coloring.

Swapping in a different noise corpus is fine and the harness supports it
(`./claim.sh --poison <path>`), but the published numbers are only comparable against these bytes.

## Fetched, not shipped

`datasets/scifact/` — BEIR SciFact (5,183 documents, 300 test queries with qrels). Downloaded and
hash-verified by `scripts/fetch_datasets.sh`; gitignored. Hashes are pinned in that script.

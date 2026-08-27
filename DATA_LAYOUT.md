# Where the data lives (not in this tree)

Stores and corpora stay **outside** the git checkout. `./run init` sets this up for you; the paths
below are what it creates.

```
$SPLATRAG_RUNTIME/            default: ./splatrag-runtime  (gitignored)
  stores/data                 your memory store
  stores/claim-<stamp>/       one disposable store per ./claim.sh run
  datasets/scifact/           fetched and hash-verified by scripts/fetch_datasets.sh
  logs/                       llama-server and service logs
```

Set `SPLATRAG_RUNTIME` in `.env` to put all of it on another volume. One store per `scope_key`.
Keep benchmark corpora out of everyday memory by giving them their own `data_dir`.

The one corpus that ships in the repo is `corpora/poison/urban-dictionary.jsonl` — see
`corpora/README.md` for why, and for its hash.

Doing it by hand instead: copy `examples/splatrag.example.toml` → `splatrag.toml` and set `data_dir`
to a path **outside** this repo.

See `configs/README.md` — only `*.example.toml` / `*.toml.in` ship. Fill in `splatrag.toml` locally.

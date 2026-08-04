# Where the data lives (not in this tree)

Stores and corpora stay **outside** the git checkout.

On the build machine:

```
/media/ruffianl/ghost_team/splatrag-runtime/
  stores/data
  stores/data-echo
  stores/data-shep
  stores/data-lumina
  stores/data-handshake
  ...
  datasets/scifact
  datasets/urban-dictionary.jsonl
```

Copy `examples/splatrag.example.toml` → `splatrag.toml` and set `data_dir` to a path **outside** this repo.

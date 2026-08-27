# Configs — what ships vs what stays local

| Ship | Local only (gitignored) |
|------|-------------------------|
| `claim.toml.in` | `personal.toml` |
| `serve.toml.in` | any filled-in silo file with a real `data_dir` |
| `personal.example.toml` | |

**Rule:** only `*.example.toml` and `*.toml.in` are public. Anything with a machine path stays local.

```bash
cp examples/splatrag.example.toml splatrag.toml
# set data_dir to a path outside the git checkout
./run init     # alternative: renders serve.toml.in → ./splatrag.toml
./claim.sh     # uses claim.toml.in
```

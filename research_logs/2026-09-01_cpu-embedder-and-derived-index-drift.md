# 2026-09-01 — CPU-locked embedder and 1022-record derived-index drift

## Symptom

"sp up doesn't work anymore, the embed model not up." `sp up` / `sp status`
reported the embedder (:8081) and service (:8767) as up, but the stack felt
dead: recall was missing recent ingests and embedding was pathologically slow.

## Findings (three independent causes)

1. **Embedder ran on CPU, not the GB10 GPU.**
   `splatrag-embed.service` (Qwen3-Embedding-8B-Q4_K_M.gguf on :8081) logged at
   its last start:
   ```
   ggml_cuda_init: failed to initialize CUDA: unknown error
   warning: no usable GPU found, --gpu-layers option will be ignored
   ```
   It silently fell back to CPU: a 3-document embed took ~19s, the process
   burned ~206% CPU, and it was absent from `nvidia-smi --query-compute-apps`
   (only Xorg/kitty graphics were listed). The CUDA build is genuine
   (build-cuda13, sm_121a kernels present, links libcublas/libcuda). A fresh
   CUDA probe from the shell succeeded (GB10 cc=12.1, cudaGetDeviceCount=0 no
   error → 1 device), so CUDA was healthy by the time we looked; the init
   failure was at the 10:02 boot, likely racing driver/module readiness.
   **Fix:** `systemctl --user restart splatrag-embed.service`. Embd fell to
   ~0.1s per 3-doc and now shows 10.5 GiB in GPU compute-apps.

2. **Derived-index drift: recall missing 1022 memories.**
   `./run doctor` failed: `cold=128210, hnsw=127188, splats=127188`.
   Timestamps said cold+m] tantivy were last written together (Aug 30 20:39)
   while HNSW/hot only reflected 127188 at the Sep 1 boot — an ingest of ~1022
   records landed in cold + BM25 but never got its vectors into HNSW/hot, so
   those memories were invisible to dense recall. The CPU-era embedder (cause
   1) is the plausible culprit for that incomplete write.
   **Fix:** `backfill-derived` (doctor's own recommendation; does not wipe
   dream/hot state).

3. **Root filesystem is 100% full** (916G, ~7.1G free on `/`). Contributing
   risk for any write to hot state / logs; not itself the trigger but should be
   freed.

## Notes on running backfill-derived against a live store

A standalone `backfill-derived` process hung (no output, killed at timeout)
while `splatrag-personal-serve.service` was running — the two mutators contend
on the store mutation/writer lock. Running the subcommand with the service
stopped completed it cleanly. It is also not instant even on GPU: ~1022
records in ~17 min (16:32 elapsed, ~35s per 32-record batch). Use detached
`nohup setsid` so the run survives the launching shell, and expect minutes.

## Result

- `./run doctor`: exit 0, cold=hnsw=splats=128210, pending_derived=false,
  dream_cycle 27, kinetic_energy intact — dream geometry **not** wiped.
- `sp status`: `128210 memories · 128210 splats · 1723 basins · dream 27`.
- Real `POST /api/recall` returns results through the running service.
- Model config unchanged (Q4_K_M); unchanged: code, memories, basin labels.

## We think

The embedder and service were both "up" the whole time — the failure was that
the embedder was CPU-locked from boot, which strangled ingest/recall speed and
left a 1022-record derived-index gap behind. A restart (getting it onto GPU)
plus one backfill cleared both. The 100%-full root disk is an open risk.

## Next

Free space on `/`; decide whether the 8B embedder (now GPU) still needs the
Q8_0/ghost_team variant or if Q4_K_M stands; confirm the 35B chat server on
:9292 is also GPU-backed (it shows 39 GiB in compute-apps — it is).

# G1 RECEIPT — Steering plumbing thin wire

**Date:** 2026-07-29
**Status:** PASS
**Workbench:** ruffian/s (SplatRAG)

## Done checklist

- [x] `src/packet.rs` — 64D pack, b64 LE, VQ PUA unicode (niodv4 codebook contract)
- [x] `pack_packet` / `unpack_packet` on MemoryService
- [x] CLI: `pack64` / `unpack64`
- [x] API: `GET /api/packet/:id`, `POST /api/packet`
- [x] MCP: `pack_64` / `unpack_64`
- [x] Unit tests: 43 pass (incl. packet raw cos=1, unicode reencode, b64)
- [x] Live smoke on cold store splat

## Live transcript

```
$ splatrag pack64 0027f64b-21e3-5fba-9be2-94cdcd47c284
{
  "version": 1,
  "dim": 64,
  "memory_id": "0027f64b-21e3-5fba-9be2-94cdcd47c284",
  "gain": 1.0,
  "mass": 1.0,
  "basin_locked": false
}
semantics_64_len 64
semantics_b64_prefix 4heYPXxCBD4iVpQ9uZMSvTLRUT6fKDa+FHXEvR8+...

$ splatrag pack64 0027f64b-21e3-5fba-9be2-94cdcd47c284 --unicode
unicode '\ue0c4'
quant_cosine 0.33232054
unicode_codebook codebook_256.json

$ splatrag pack64 0027f64b-21e3-5fba-9be2-94cdcd47c284 > /tmp/pkt.json && splatrag unpack64 /tmp/pkt.json
{
  "memory_id": "0027f64b-21e3-5fba-9be2-94cdcd47c284",
  "gain": 1.0,
  "op": "polarity",
  "mass": 1.0,
  "cosine_before_after": 0.9999998,
  "collapse_risk": true,
  "basin_id": null,
  "basin_locked": false
}
```

## Knobs still independent

- gain: OI invert/amplify on 64D (steer)
- mass: dream repulsion when negative (steer --mass)
- unicode: transport only (pack --unicode)

## STOP

G1 closed. Do not redesign VQ, retrain codebook, or open bulk Grok ingest under this receipt.
Team may open **G2** only if Jason says so (`TEAM_GOAL_STEERING_PLUMBING.md`).

# 2026-07-30 — Steer axis fix (α actually works)

**Lead:** Jason Van Pham  
**Lane:** Grok (steering) · prior defect pinned by Claude in `2026-07-29_the_picker.md`  
**Status:** fixed · 62 tests green  

## Defect

`MemoryService::steer` called `apply_steering(&before, &before, gain, op)` — **self-axis**.  
On a self-axis, OI operators only scale a direction that normalization throws away, so:

- polarity → cosine exactly −1 for every α  
- householder / negative_gain → cosine exactly +1 for every α  

The measured sweet band (0.15–0.30) could not be expressed. Smokes that only checked “cosine moved” still passed.

## Fix

Default concept axis is **non-self μ** (`SteerAxisMode::Auto`):

1. mean of other **basin peers**  
2. **field mean** (`projection.mean` or mean of other splats)  
3. first **PCA basis**  
4. **orthogonal fallback**  

CLI / MCP / HTTP: `--axis auto|self|field|basin` (default `auto`).  
`self` kept for ablation. Report adds `axis_source` and `axis_cosine`.

Operators unchanged. Axis was wrong; operators were fine (OI prior art already had μ).

## Usage

```bash
splatrag steer <memory-id> --gain -0.2
# report: axis_source=basin_peers|field_mean|…  axis_cosine≪1  cosine_before_after responds to α

splatrag steer <memory-id> --gain -0.2 --axis self   # old degenerate behaviour
```

## Not claimed

- Hydro residual coupling (pick bridge + three-arm still open)  
- John A0 SWA integrity  

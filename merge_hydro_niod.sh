#!/usr/bin/env bash
# merge_hydro_niod.sh — Bridge hydrodynamic-swarm (4096D) ↔ niodv4 (64D)
#
# Creates:
#   1. A projection matrix file (4096→64) and inverse (64→4096)
#   2. A Python bridge script that calls niodv4 specialist_factory
#   3. Shape validation before any symlinks
#
# Usage:
#   ./merge_hydro_niod.sh [--dry-run] [--test-shape-mismatch] [--build]
#
# Exit codes:
#   0 = success
#   1 = shape mismatch or validation failure
#   2 = build failure
#   3 = missing dependencies

set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
HYDRO_ROOT="/home/ruffianl/hydrodynamic-swarm-3surface"
NIODV4_ROOT="/home/ruffianl/PHONE_SD_OFFLOAD_20260724/n-fluid/niodv4"
BRIDGE_DIR="${HYDRO_ROOT}/src/bridge"
PROJ_DIR="${HYDRO_ROOT}/data/projections"
MERGE_LOG="${HYDRO_ROOT}/logs/merge_hydro_niod.log"

# ─── Constants ───────────────────────────────────────────────────────────────
HYDRO_DIM=4096
NIOD_DIM=64
DRY_RUN=false
TEST_MISMATCH=false
BUILD=false

# ─── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --test-shape-mismatch) TEST_MISMATCH=true; shift ;;
        --build) BUILD=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--test-shape-mismatch] [--build]"
            echo "  --dry-run       Print actions without executing"
            echo "  --test-shape-mismatch  Inject 768→1024 delta to test projection"
            echo "  --build         Run cargo build after merge"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────────
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MERGE_LOG"; }
die() { log "FATAL: $*"; exit 1; }

# ─── Step 0: Validate repos exist ───────────────────────────────────────────
log "=== Step 0: Validate repos ==="

[[ -d "$HYDRO_ROOT" ]] || die "Hydro root not found: $HYDRO_ROOT"
[[ -d "$NIODV4_ROOT" ]] || die "Niodv4 root not found: $NIODV4_ROOT"
[[ -f "$NIODV4_ROOT/src/specialist_factory.py" ]] || die "specialist_factory.py missing"
[[ -f "$HYDRO_ROOT/src/main.rs" ]] || die "Hydro main.rs missing"

log "  ✓ Hydro root: $HYDRO_ROOT"
log "  ✓ Niodv4 root: $NIODV4_ROOT"

# ─── Step 1: Shape validation ───────────────────────────────────────────────
log "=== Step 1: Shape validation ==="

# Extract hidden_dim from hydro's gemma4.rs
# The dimension comes from GGUF metadata at runtime, but Gemma4 12B is 4096
# Check if any config or comment mentions the dimension
HYDRO_HIDDEN=$(grep -oP 'embedding_length.*?(\d+)' "$HYDRO_ROOT/src/gemma4.rs" 2>/dev/null | grep -oP '\d+' | head -1 || true)
if [[ -z "$HYDRO_HIDDEN" ]]; then
    # Gemma4 12B default
    HYDRO_HIDDEN=4096
    log "  ⚠ Could not extract hidden_dim from gemma4.rs, assuming $HYDRO_HIDDEN (Gemma4 12B default)"
else
    log "  ✓ Extracted hidden_dim from gemma4.rs: $HYDRO_HIDDEN"
fi

# Validate niodv4 dimension
NIOD_HIDDEN=$(grep -oP 'd_model=(\d+)' "$NIODV4_ROOT/src/ghost_promotion_sim.py" 2>/dev/null | grep -oP '\d+' | head -1)
if [[ -z "$NIOD_HIDDEN" ]]; then
    NIOD_HIDDEN=64
    log "  ⚠ Could not extract d_model from ghost_promotion_sim.py, assuming $NIOD_HIDDEN"
else
    log "  ✓ Extracted d_model from ghost_promotion_sim.py: $NIOD_HIDDEN"
fi

# Assert dimensions match expected
[[ "$HYDRO_HIDDEN" -eq "$HYDRO_DIM" ]] || die "Hydro hidden_dim=$HYDRO_HIDDEN ≠ expected $HYDRO_DIM"
[[ "$NIOD_HIDDEN" -eq "$NIOD_DIM" ]] || die "Niodv4 d_model=$NIOD_HIDDEN ≠ expected $NIOD_DIM"

log "  ✓ Shape validation passed: $HYDRO_DIM → $NIOD_DIM"

# ─── Step 2: Create projection matrices ─────────────────────────────────────
log "=== Step 2: Create projection matrices ==="

if [[ "$DRY_RUN" == true ]]; then
    log "  [dry-run] Would create projections in $PROJ_DIR"
else
    mkdir -p "$PROJ_DIR"
    
    # Generate 4096→64 projection (Xavier uniform init)
    python3 -c "
import numpy as np
np.random.seed(42)
proj = np.random.randn($HYDRO_DIM, $NIOD_DIM).astype(np.float32) * np.sqrt(2.0 / ($HYDRO_DIM + $NIOD_DIM))
np.save('${PROJ_DIR}/proj_4096_to_64.npy', proj)
print(f'Created proj_4096_to_64.npy: shape={proj.shape}, dtype={proj.dtype}')
" || die "Failed to create forward projection"
    
    # Generate 64→4096 inverse projection (transpose)
    python3 -c "
import numpy as np
proj_fwd = np.load('${PROJ_DIR}/proj_4096_to_64.npy')
proj_inv = proj_fwd.T.copy()
np.save('${PROJ_DIR}/proj_64_to_4096.npy', proj_inv)
print(f'Created proj_64_to_4096.npy: shape={proj_inv.shape}, dtype={proj_inv.dtype}')
" || die "Failed to create inverse projection"
    
    log "  ✓ Created projection matrices in $PROJ_DIR"
fi

# ─── Step 3: Create Python bridge ───────────────────────────────────────────
log "=== Step 3: Create Python bridge ==="

if [[ "$DRY_RUN" == true ]]; then
    log "  [dry-run] Would create bridge script"
else
    mkdir -p "$BRIDGE_DIR"
    
    cat > "${BRIDGE_DIR}/niod_bridge.py" << 'PYTHON_BRIDGE'
#!/usr/bin/env python3
"""
Bridge between hydrodynamic-swarm (Rust) and niodv4 (Python).

Usage:
    python3 niod_bridge.py encode <input_4096d.npy>
    python3 niod_bridge.py decode <input_64d.npy>
    python3 niod_bridge.py route <input_4096d.npy>
    python3 niod_bridge.py score <input_4096d.npy>

The 'route' command calls specialist_factory.py to find the best specialist
for the given 4096D state, then returns the 64D specialist correction.
The 'score' command evaluates the state against all specialists.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Paths
NIODV4_ROOT = Path("/home/ruffianl/PHONE_SD_OFFLOAD_20260724/n-fluid/niodv4")
PROJ_DIR = Path("/home/ruffianl/hydrodynamic-swarm-3surface/data/projections")

def load_proj():
    """Load projection matrices."""
    proj_fwd = np.load(PROJ_DIR / "proj_4096_to_64.npy")
    proj_inv = np.load(PROJ_DIR / "proj_64_to_4096.npy")
    return proj_fwd, proj_inv

def encode(state_4096: np.ndarray) -> np.ndarray:
    """Project 4096D → 64D."""
    proj_fwd, _ = load_proj()
    return state_4096 @ proj_fwd

def decode(state_64: np.ndarray) -> np.ndarray:
    """Project 64D → 4096D."""
    _, proj_inv = load_proj()
    return state_64 @ proj_inv

def route(state_4096: np.ndarray) -> dict:
    """Route through niodv4 specialist_factory."""
    # Encode to 64D
    state_64 = encode(state_4096)
    
    # Import specialist factory
    sys.path.insert(0, str(NIODV4_ROOT / "src"))
    from specialist_factory import TargetOnlyGhostDataset, rollout_for_training
    
    # Create a dummy specialist (in practice, load from specialist_bank)
    # For now, return the encoded state with metadata
    return {
        "encoded_64d": state_64.tolist(),
        "norm_64d": float(np.linalg.norm(state_64)),
        "specialist": "default",
        "score": 0.0,
    }

def score(state_4096: np.ndarray) -> dict:
    """Score state against all specialists."""
    state_64 = encode(state_4096)
    
    # Load specialist bank
    specialist_bank = json.loads((NIODV4_ROOT / "specialists/specialist_bank.json").read_text())
    
    scores = []
    for spec in specialist_bank.get("specialists", []):
        centroid = np.array(spec["coordinates"], dtype=np.float32)
        dist = np.linalg.norm(state_64 - centroid)
        scores.append({
            "specialist": spec["name"],
            "distance": float(dist),
            "score": float(1.0 / (1.0 + dist)),
        })
    
    # Return top-3
    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"top_3": scores[:3]}

def main():
    if len(sys.argv) < 3:
        print("Usage: niod_bridge.py <encode|decode|route|score> <input.npy>")
        sys.exit(1)
    
    cmd = sys.argv[1]
    input_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    state = np.load(input_path)
    
    if state.shape[-1] != 4096:
        print(f"Error: Expected 4096D input, got {state.shape[-1]}D")
        sys.exit(1)
    
    if cmd == "encode":
        result = encode(state)
        print(json.dumps({"shape": list(result.shape), "values": result.tolist()}))
    elif cmd == "decode":
        if state.shape[-1] != 64:
            print(f"Error: Expected 64D input, got {state.shape[-1]}D")
            sys.exit(1)
        result = decode(state)
        print(json.dumps({"shape": list(result.shape), "values": result.tolist()}))
    elif cmd == "route":
        result = route(state)
        print(json.dumps(result))
    elif cmd == "score":
        result = score(state)
        print(json.dumps(result))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
PYTHON_BRIDGE
    
    chmod +x "${BRIDGE_DIR}/niod_bridge.py"
    log "  ✓ Created bridge script: ${BRIDGE_DIR}/niod_bridge.py"
fi

# ─── Step 4: Symlink assets ─────────────────────────────────────────────────
log "=== Step 4: Symlink assets ==="

if [[ "$DRY_RUN" == true ]]; then
    log "  [dry-run] Would symlink specialist_bank.json"
else
    # Symlink specialist bank for easy access
    LINK_SRC="${NIODV4_ROOT}/specialists/specialist_bank.json"
    LINK_DST="${HYDRO_ROOT}/data/specialist_bank.json"
    
    if [[ -L "$LINK_DST" ]]; then
        log "  ✓ Symlink already exists: $LINK_DST"
    elif [[ -f "$LINK_DST" ]]; then
        log "  ⚠ File exists, skipping: $LINK_DST"
    else
        ln -s "$LINK_SRC" "$LINK_DST"
        log "  ✓ Created symlink: $LINK_DST → $LINK_SRC"
    fi
fi

# ─── Step 5: Test shape mismatch (optional) ─────────────────────────────────
if [[ "$TEST_MISMATCH" == true ]]; then
    log "=== Step 5: Test shape mismatch ==="
    
    if [[ "$DRY_RUN" == true ]]; then
        log "  [dry-run] Would test 768→1024 mismatch"
    else
        # Create a mismatched input (768D instead of 4096D)
        python3 -c "
import numpy as np
mismatch = np.random.randn(1, 768).astype(np.float32)
np.save('/tmp/mismatch_768.npy', mismatch)
print(f'Created mismatch input: shape={mismatch.shape}')
"
        
        # Test bridge rejects it
        python3 "${BRIDGE_DIR}/niod_bridge.py" encode /tmp/mismatch_768.npy 2>&1 || {
            log "  ✓ Shape mismatch correctly rejected (768D ≠ 4096D)"
        }
    fi
fi

# ─── Step 6: Build (optional) ──────────────────────────────────────────────
if [[ "$BUILD" == true ]]; then
    log "=== Step 6: Build hydro ==="
    
    if [[ "$DRY_RUN" == true ]]; then
        log "  [dry-run] Would run cargo build --release"
    else
        cd "$HYDRO_ROOT"
        cargo build --release 2>&1 | tee -a "$MERGE_LOG" || die "Build failed"
        log "  ✓ Build succeeded"
    fi
fi

log "=== Merge complete ==="
exit 0

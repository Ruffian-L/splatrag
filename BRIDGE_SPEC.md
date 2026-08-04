# SplatRag ↔ 3surface Bridge Spec

## Problem

SplatRag (`ruffian/s`) and 3surface (`hydrodynamic-swarm-3surface`) have incompatible memory formats:
- SplatRag: `MemoryRecord` (text + metadata + 64D embeddings) stored in tantivy + Qdrant + HNSW indexes
- 3surface: `Splat` (mu + sigma + alpha + lambda) stored as safetensors

## Solution

Bidirectional bridge that converts between formats while preserving steering semantics.

## Format Mapping

### SplatRag → 3surface (MemoryRecord → Splat)

| SplatRag Field | 3surface Field | Conversion |
|----------------|----------------|------------|
| `MemoryRecord.text` | `Splat.mu` | Embed with 3surface's residual path (not 64D Qwen) |
| `MemoryPacket.gain` | `Splat.alpha` | Direct mapping (positive = pleasure, negative = pain) |
| `MemoryPacket.mass` | `Splat.alpha` sign | Negative mass → negative alpha (repel) |
| `MemoryPacket.basin_id` | `Splat` metadata | Store as splat metadata |
| `MemoryRecord.ingested_at` | `Splat.created_at` | Direct mapping |
| `MemoryRecord.domain` | `Splat` metadata | Store as splat metadata |

### 3surface → SplatRag (Splat → MemoryRecord)

| 3surface Field | SplatRag Field | Conversion |
|----------------|----------------|------------|
| `Splat.mu` | `MemoryPacket.semantics_64` | Project to 64D latent space |
| `Splat.alpha` | `MemoryPacket.gain` | Direct mapping |
| `Splat.sigma` | `MemoryRecord` metadata | Store as splat metadata |
| `Splat.created_at` | `MemoryRecord.ingested_at` | Direct mapping |
| `Splat.scale` | `MemoryRecord.domain` | Fine/Medium/Coarse → domain labels |

## API

### SplatRag Side

```rust
impl MemoryService {
    /// Import 3surface splats from safetensors file
    pub fn import_splats(&mut self, path: &Path) -> Result<usize> {
        // Load safetensors
        // Convert each splat to MemoryRecord
        // Add to indexes
    }
    
    /// Export memories as 3surface-compatible splats
    pub fn export_splats(&self, path: &Path) -> Result<usize> {
        // Query indexes
        // Convert each MemoryRecord to Splat
        // Save as safetensors
    }
}
```

### 3surface Side

```rust
impl SplatMemory {
    /// Import SplatRag pick files
    pub fn import_picks(&mut self, path: &Path) -> Result<usize> {
        // Already exists in picks.rs
    }
    
    /// Export splats as SplatRag pick files
    pub fn export_picks(&self, path: &Path) -> Result<usize> {
        // Convert each splat to MemoryPickSet
        // Save as JSON
    }
}
```

## Acceptance Criteria

1. **SplatRag can import 3surface splats**: `MemoryService.import_splats(path)` loads 3surface's safetensors and creates `MemoryRecord` entries in its indexes.

2. **3surface can import SplatRag picks**: Already works via `picks.rs`, but needs to handle SplatRag's `basin_id` and `mass` fields correctly.

3. **Round-trip fidelity**: Deposit a splat in 3surface → export to SplatRag → query SplatRag → import back to 3surface → verify the splat is intact (mu within 1e-3, alpha within 1e-3).

4. **Steering semantics preserved**: Negative gain/mass → negative alpha (repel). Positive gain/mass → positive alpha (attract).

## niodv4 Integration

niodv4's `CoreMemoryOrchestrator` produces 64D bounded latent states. These should be projected to 3surface's residual space before depositing as splats:

```rust
impl SplatMemory {
    /// Import niodv4 vortex states
    pub fn import_vortex_states(&mut self, path: &Path) -> Result<usize> {
        // Load niodv4 .pth or .json state
        // Project 64D to residual space using niodv4's projection matrix
        // Convert to splats
    }
}
```

## Implementation Order

1. **SplatRag → 3surface**: `MemoryService.import_splats()`
2. **3surface → SplatRag**: `SplatMemory.export_picks()`
3. **Round-trip test**: Verify fidelity
4. **niodv4 integration**: `SplatMemory.import_vortex_states()`

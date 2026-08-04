# SplatRAG Frontend ↔ Rust Backend Hookup Contract

**For Claude:** wire the live Rust SplatLens (`web.rs` + `MemoryService`) to this TanStack Start / Vercel frontend.

Frontend already implements the same HTTP paths as Rust. When `SPALTRAG_API_URL` is set, API routes **proxy** to Rust and re-validate with Zod. When unset, they serve a typed mock store.

---

## 1. HTTP surface (must match exactly)

| Method | Path | Query | Response |
|--------|------|-------|----------|
| GET | `/api/status` | — | `MemoryStatus` |
| GET | `/api/basins` | — | `Basin[]` |
| GET | `/api/splats` | `level`, `basin_id?`, `offset?`, `limit?` | `SplatPage` |
| GET | `/api/memories/:id` | — | `MemoryRecord` or **404** |
| GET | `/api/search` | `q` (required), `limit?` | `RecallHit[]` |
| errors | any | — | `{ "error": string }` with non-2xx |

### Query defaults (frontend Zod)

```
level:   "basins" | "memories"   default "basins"
offset:  int ≥ 0                 default 0
limit:   splats 1..50000 default 20000
         search 1..100   default 10
q:       non-empty string
```

Rust (`web.rs`) already uses the same path names and query field names.

### Env (frontend)

```
SPALTRAG_API_URL=https://your-splatlens-host   # no trailing slash required
SPALTRAG_SCOPE_KEY=local/palace               # optional, mock only
```

Proxy: `GET ${SPALTRAG_API_URL}${path}` with `Accept: application/json`, `cache: no-store`.

---

## 2. Exact Zod schemas (frontend source of truth)

File: `src/lib/memory/schema.ts`

```ts
import { z } from "zod";

export const vec3Schema = z.tuple([z.number(), z.number(), z.number()]);
export const vec4Schema = z.tuple([z.number(), z.number(), z.number(), z.number()]);
export const colorRgbaSchema = z.tuple([
  z.number().int().min(0).max(255),
  z.number().int().min(0).max(255),
  z.number().int().min(0).max(255),
  z.number().int().min(0).max(255),
]);

export const memoryRecordSchema = z.object({
  id: z.string().uuid(),
  source_key: z.string().min(1),
  text: z.string(),
  domain: z.string().min(1),
  timestamp: z.string().datetime({ offset: true }).nullable(),
  ingested_at: z.string().datetime({ offset: true }),
  source: z.string().min(1),
  speaker: z.string().nullable(),
  model: z.string().nullable(),
  conversation_id: z.string().nullable(),
  turn_index: z.number().int().nonnegative().nullable(),
});

export const basinSchema = z.object({
  id: z.string().min(1),
  label: z.string(),
  path: z.string(),
  summary: z.string(),
  label_state: z.enum(["pending", "stable"]),
  stability: z.number().min(0).max(1),
  centroid: vec3Schema,
  member_ids: z.array(z.string().uuid()),
  representative_ids: z.array(z.string().uuid()),
  updated_at: z.string().datetime({ offset: true }),
});

export const viewerSplatSchema = z.object({
  id: z.string().min(1),
  position: vec3Schema,
  scale: vec3Schema,
  rotation: vec4Schema,
  color: colorRgbaSchema,
  radiance: z.number(),
  mass: z.number(),
  basin_id: z.string().nullable(),
  label: z.string().nullable(),
});

export const splatPageSchema = z.object({
  level: z.enum(["basins", "memories"]),
  total: z.number().int().nonnegative(),
  offset: z.number().int().nonnegative(),
  splats: z.array(viewerSplatSchema),
});

export const scoreBreakdownSchema = z.object({
  bm25: z.number(),
  cosine: z.number(),
  radiance: z.number(),
  radiance_weight: z.number(),
  final_score: z.number(),
});

export const recallHitSchema = z.object({
  memory: memoryRecordSchema,
  basin_id: z.string().nullable(),
  basin_label: z.string().nullable(),
  scores: scoreBreakdownSchema,
});

export const memoryStatusSchema = z.object({
  scope_key: z.string(),
  cold_records: z.number().int().nonnegative(),
  hnsw_keys: z.number().int().nonnegative(),
  qdrant_points: z.number().int().nonnegative().nullable(),
  splats: z.number().int().nonnegative(),
  basins: z.number().int().nonnegative(),
  dream_cycle: z.number().int().nonnegative(),
  kinetic_energy: z.number(),
});

export const apiErrorSchema = z.object({
  error: z.string(),
});
```

---

## 3. JSON examples (what frontend expects after `JSON.stringify`)

### GET /api/status → 200

```json
{
  "scope_key": "local/palace",
  "cold_records": 28,
  "hnsw_keys": 28,
  "qdrant_points": 28,
  "splats": 28,
  "basins": 6,
  "dream_cycle": 17,
  "kinetic_energy": 0.00042
}
```

`qdrant_points` may be `null` if Qdrant is disabled.

### GET /api/basins → 200

```json
[
  {
    "id": "a1b2c3d4-e5f6-4f97-a628-c1e477206b3d",
    "label": "Dream physics",
    "path": "memory/research/physics",
    "summary": "Sign-from-semantics force law…",
    "label_state": "stable",
    "stability": 0.88,
    "centroid": [3.8, -0.6, 1.2],
    "member_ids": ["00000007-5af2-4e7c-9abc-def013572468"],
    "representative_ids": ["00000007-5af2-4e7c-9abc-def013572468"],
    "updated_at": "2026-07-28T22:14:00.000Z"
  }
]
```

### GET /api/splats?level=basins → 200

```json
{
  "level": "basins",
  "total": 6,
  "offset": 0,
  "splats": [
    {
      "id": "a1b2c3d4-e5f6-4f97-a628-c1e477206b3d",
      "position": [3.8, -0.6, 1.2],
      "scale": [1.8, 0.99, 0.99],
      "rotation": [0, 0, 0, 1],
      "color": [110, 230, 190, 230],
      "radiance": 1.94,
      "mass": 6,
      "basin_id": "a1b2c3d4-e5f6-4f97-a628-c1e477206b3d",
      "label": "Dream physics"
    }
  ]
}
```

### GET /api/splats?level=memories&basin_id=… → same shape, splat ids = memory UUIDs

### GET /api/memories/:id → 200

```json
{
  "id": "00000007-5af2-4e7c-9abc-def013572468",
  "source_key": "palace/physics/ph-1",
  "text": "Gravity-style…",
  "domain": "research",
  "timestamp": "2026-07-10T22:14:00.000Z",
  "ingested_at": "2026-07-10T22:14:00.000Z",
  "source": "model",
  "speaker": null,
  "model": "grok-4",
  "conversation_id": "dream-law",
  "turn_index": 1
}
```

### GET /api/search?q=dream&limit=10 → 200

```json
[
  {
    "memory": { /* MemoryRecord as above */ },
    "basin_id": "…",
    "basin_label": "Dream physics",
    "scores": {
      "bm25": 0.77,
      "cosine": 0.53,
      "radiance": 0.66,
      "radiance_weight": 0.18,
      "final_score": 0.678
    }
  }
]
```

---

## 4. Rust ↔ frontend mismatches Claude MUST fix or tolerate

| Field / shape | Rust today | Frontend Zod expects | Action |
|---------------|------------|----------------------|--------|
| `MemoryRecord` | full struct: `content_sha256`, `source_file`, `attachments`, `metadata`, … | **subset only** (see schema) | Prefer **DTO** on Rust that serializes only the subset, OR widen frontend schema with `.passthrough()` / optional extras |
| `Uuid` | serializes as JSON string | `z.string().uuid()` | OK if standard UUID string |
| `DateTime<Utc>` | RFC3339 | `z.string().datetime({ offset: true })` | Ensure `Z` or offset (e.g. `2026-07-28T22:14:00.000Z`) |
| `RecallHit` | includes `context: { before, after }` | **no `context` field** | Strip context in a DTO, or make frontend `context` optional |
| `Basin` (geometry) | full `Basin` from hot state | fields listed in `basinSchema` | Confirm JSON field names: `label_state`, `member_ids`, `representative_ids`, `centroid`, `stability`, `path`, `summary`, `updated_at` |
| `SplatPage.level` | free `String` | enum `"basins" \| "memories"` | Only emit those two strings |
| `color` | `[u8; 4]` | ints 0–255 | OK |
| `qdrant_points` | `Option<u64>` | number \| null | OK |
| 404 memory | `StatusCode::NOT_FOUND` empty | client treats 404 as null | Keep empty body OK |

### Minimal Rust DTOs Claude can add (recommended)

```rust
// wire.rs — frontend-safe JSON only

#[derive(Serialize)]
pub struct WireMemoryRecord {
    pub id: String,              // uuid to_string()
    pub source_key: String,
    pub text: String,
    pub domain: String,
    pub timestamp: Option<String>,   // RFC3339
    pub ingested_at: String,
    pub source: String,
    pub speaker: Option<String>,
    pub model: Option<String>,
    pub conversation_id: Option<String>,
    pub turn_index: Option<u64>,
}

#[derive(Serialize)]
pub struct WireRecallHit {
    pub memory: WireMemoryRecord,
    pub basin_id: Option<String>,
    pub basin_label: Option<String>,
    pub scores: ScoreBreakdown,  // already matches
    // omit context
}

#[derive(Serialize)]
pub struct WireBasin {
    pub id: String,
    pub label: String,
    pub path: String,
    pub summary: String,
    pub label_state: String,     // "pending" | "stable"
    pub stability: f32,
    pub centroid: [f32; 3],
    pub member_ids: Vec<String>,
    pub representative_ids: Vec<String>,
    pub updated_at: String,
}
```

Map `MemoryRecord` → `WireMemoryRecord` in `/api/memories` and `/api/search`.  
Map `Basin` → `WireBasin` in `/api/basins`.  
`ViewerSplat` / `SplatPage` / `MemoryStatus` already match.

---

## 5. Frontend files Claude should know

| Path | Role |
|------|------|
| `src/lib/memory/schema.ts` | Zod + inferred types |
| `src/lib/memory/client.ts` | Browser client (`memoryApi.*`) |
| `src/lib/memory/store.ts` | Mock store (validated outbound) |
| `src/lib/memory/proxy.ts` | `proxyGet` to Rust |
| `src/lib/memory/env.ts` | `SPALTRAG_API_URL` |
| `src/routes/api/status.ts` | GET status |
| `src/routes/api/basins.ts` | GET basins |
| `src/routes/api/splats.ts` | GET splats |
| `src/routes/api/search.ts` | GET search |
| `src/routes/api/memories.$id.ts` | GET memory by id |
| `src/components/MemoryPalace.tsx` | UI that calls `memoryApi` |
| `src/components/SplatField.tsx` | 3D/2D field renderer |
| `src/components/Landing.tsx` | Marketing landing `/` |
| `src/routes/palace.tsx` | Palace at `/palace` |

Rust reference (attachments in this workspace):

| Path | Role |
|------|------|
| `attachments/web.rs` | Axum routes + ViewerSplat/SplatPage |
| `attachments/record.rs` | MemoryRecord, RecallHit, filters |
| `attachments/service.rs` | MemoryService + MemoryStatus |
| `attachments/physics.rs` | dream() |
| `attachments/retrieval.rs` | hybrid recall scoring |
| `attachments/topology.rs` | basin discovery |
| `attachments/qdrant.rs` | dense index |
| `attachments/mcp.rs` | MCP tools (remember/recall/…) |

---

## 6. Hookup steps (ordered)

1. **Rust:** ensure CORS allows the Vercel/frontend origin (already `CorsLayer::permissive()` in web.rs).
2. **Rust:** add wire DTOs so `/api/memories` and `/api/search` don’t send extra-required fields that break Zod **if** you tighten client parse — currently Zod **strips unknown keys by default in Zod 3; Zod 4 may differ** — safest is DTO subset.
3. **Rust:** confirm basin JSON uses snake_case field names matching `basinSchema`.
4. **Deploy/frontend:** set `SPALTRAG_API_URL` to Rust bind URL (e.g. `http://127.0.0.1:8081` or public HTTPS).
5. **Smoke:**
   ```bash
   curl -s $SPALTRAG_API_URL/api/status
   curl -s $SPALTRAG_API_URL/api/basins | head
   curl -s "$SPALTRAG_API_URL/api/splats?level=basins&limit=2"
   curl -s "$SPALTRAG_API_URL/api/search?q=memory&limit=3"
   ```
6. **Frontend:** with env set, `/api/*` proxies; UI shows real counts. Without env, mock palace still works.
7. **Optional later:** POST remember, dream trigger, MCP — not used by UI yet.

---

## 7. What the UI calls (client)

```ts
memoryApi.status()
memoryApi.basins()
memoryApi.splats({ level: "basins" | "memories", basin_id?, limit? })
memoryApi.memory(id)      // null on 404
memoryApi.search(q, limit)
```

No auth headers today. Add `Authorization` in `proxy.ts` + `client.ts` if Rust gains API keys.

---

## 8. One-sentence contract

**SplatLens JSON is the protocol; frontend Zod is the gate; `SPALTRAG_API_URL` is the switch; wire DTOs should emit exactly the fields in §2.**

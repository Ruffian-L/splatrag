use crate::inversion::InversionOp;
use crate::packet::{MemoryPacket, VqCodebook};
use crate::physics::domain_state_color;
use crate::record::RecallFilters;
use crate::service::{MemoryService, MemoryStatus, SteerAddress, SteerOpts, SteerReport};
use crate::sidecar::SidecarState;
use crate::wire::{WireBasin, WireMemoryRecord, WireRecallHit};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

const APP_JS: &str = include_str!("../assets/app.js");
const STYLE_CSS: &str = include_str!("../assets/style.css");

#[derive(Debug)]
struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": self.0.to_string()})),
        )
            .into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

#[derive(Debug, Deserialize)]
struct SplatQuery {
    level: Option<String>,
    basin_id: Option<String>,
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SteerBody {
    /// Legacy one-pin. Optional when `addresses` is set.
    #[serde(default)]
    memory_id: Option<String>,
    /// N addresses, one budget. Pin = memory_id; room = basin_id only.
    #[serde(default)]
    addresses: Vec<SteerAddress>,
    /// Total |α| to split. Default |gain|.
    #[serde(default)]
    budget: Option<f32>,
    /// Negative = invert semantics (OI). Positive = amplify. Zero = leave semantics.
    #[serde(default)]
    gain: f32,
    /// polarity | householder | negative_gain
    #[serde(default)]
    op: Option<String>,
    /// If set, overwrite mass. Negative mass repels. Not implied by gain.
    mass: Option<f32>,
    basin_id: Option<String>,
    #[serde(default)]
    lock: bool,
    /// auto | self | field | basin — default auto (non-self μ so α works).
    #[serde(default)]
    axis: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PacketQuery {
    /// If true, attach VQ Unicode from default/codebook path.
    #[serde(default)]
    unicode: bool,
    /// Optional path to niodv4 codebook_256.json
    codebook: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UnpackBody {
    #[serde(flatten)]
    packet: MemoryPacket,
    /// Override packet.memory_id when the wire form omits or remaps it.
    #[serde(default)]
    id_override: Option<String>,
    #[serde(default)]
    codebook: Option<String>,
}

/// Resolve the VQ codebook for packet unicode: request parameter, then `SPLATRAG_CODEBOOK`.
/// No built-in default — the codebook is not distributed here, and an empty path only ever
/// produced a confusing file-not-found.
fn open_codebook(path: Option<&str>, want: bool) -> Result<Option<VqCodebook>, ApiError> {
    let path = path
        .map(str::to_string)
        .or_else(|| std::env::var("SPLATRAG_CODEBOOK").ok());
    match path {
        Some(path) => Ok(Some(VqCodebook::load_json(&path)?)),
        None if want => Err(ApiError(anyhow::anyhow!(
            "VQ unicode needs a codebook: pass ?codebook=<codebook_256.json> \
             or set SPLATRAG_CODEBOOK on the server"
        ))),
        None => Ok(None),
    }
}

#[derive(Debug, Serialize)]
struct ViewerSplat {
    id: String,
    position: [f32; 3],
    scale: [f32; 3],
    rotation: [f32; 4],
    color: [u8; 4],
    radiance: f32,
    mass: f32,
    basin_id: Option<String>,
    label: Option<String>,
}

#[derive(Debug, Serialize)]
struct SplatPage {
    level: String,
    total: usize,
    offset: usize,
    splats: Vec<ViewerSplat>,
}

pub async fn serve(service: Arc<MemoryService>) -> anyhow::Result<()> {
    // Hard gate: never listen on 0.0.0.0 / :: even if config was hand-edited after load.
    crate::config::assert_loopback_bind(&service.config.server.bind, "server.bind")?;
    let bind = service.config.server.bind.clone();
    if let Ok(path) = std::env::var("SPLATRAG_LIVE_LOG")
        && !path.is_empty()
    {
        service.set_live_log_path(Some(std::path::PathBuf::from(&path)));
        eprintln!("live frame log → {path}");
    }
    // Domain colors on first paint so the field is readable without a dream.
    if let Err(err) = service.recolor_field().await {
        eprintln!("recolor_field: {err:#}");
    }
    let app = Router::new()
        .route("/", get(index))
        .route("/app.js", get(javascript))
        .route("/style.css", get(stylesheet))
        .route("/api/status", get(status))
        .route("/api/basins", get(basins))
        .route("/api/splats", get(splats))
        .route("/api/memories/:id", get(memory))
        .route("/api/sidecar/:id", get(sidecar_get).put(sidecar_put))
        .route("/api/search", get(search))
        // gain inverts/amplifies semantics; mass (if set negative) repels — independent knobs
        .route("/api/steer", post(steer))
        .route("/api/packet/:id", get(get_packet))
        .route("/api/packet", post(post_packet))
        .route("/api/live", get(live_sse))
        .route("/api/dream", post(start_dream))
        .route("/api/recolor", post(recolor))
        .route("/api/flip", get(flip_get).post(flip_post))
        .route("/api/miss-classify", post(miss_classify_post))
        .route("/api/jlens", get(jlens_get).post(jlens_post))
        // Agent surface: the same tools the stdio MCP server exposes, over HTTP.
        .route("/api/tools", get(tools_list))
        .route("/api/tools/:name", post(tools_call))
        .route("/api/recall", post(recall_post))
        .route("/api/ingest", post(ingest_post))
        .route("/api/backfill-derived", post(backfill_derived_post))
        // One Streamable HTTP MCP endpoint for every local agent/client. The gateway
        // routes per-call `silo=` values; clients must not spawn one stdio process per silo.
        .route("/mcp", post(mcp_streamable_http))
        .layer(CorsLayer::permissive())
        .with_state(service);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("SplatRAG listening on http://{bind}  (GET / → Basin Field :8080)");
    eprintln!("  live SSE  GET  /api/live");
    eprintln!("  dream     POST /api/dream  body: {{\"emit_every\":5}}");
    eprintln!("  flip      GET|POST /api/flip?q=…");
    eprintln!("  jlens     GET|POST /api/jlens?q=…  (verbalizable words)");
    eprintln!("  misses    POST /api/miss-classify");
    eprintln!("  tools     GET  /api/tools          agent tool manifest");
    eprintln!("            POST /api/tools/:name    call one tool");
    eprintln!("            POST /api/recall         search with filters");
    eprintln!("            POST /api/ingest         agent-jsonl path → remember_batch");
    eprintln!("  MCP       POST /mcp                shared Streamable HTTP gateway");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Redirect {
    Redirect::temporary("http://127.0.0.1:8080/")
}

async fn javascript() -> impl IntoResponse {
    ([("content-type", "text/javascript; charset=utf-8")], APP_JS)
}

async fn stylesheet() -> impl IntoResponse {
    ([("content-type", "text/css; charset=utf-8")], STYLE_CSS)
}

async fn status(State(service): State<Arc<MemoryService>>) -> Result<Json<MemoryStatus>, ApiError> {
    Ok(Json(service.status().await?))
}

async fn basins(State(service): State<Arc<MemoryService>>) -> Json<Vec<WireBasin>> {
    Json(service.list_basins().iter().map(WireBasin::from).collect())
}

async fn splats(
    State(service): State<Arc<MemoryService>>,
    Query(query): Query<SplatQuery>,
) -> Result<Response, ApiError> {
    let snapshot = service.hot_snapshot();
    let level = query.level.unwrap_or_else(|| "basins".into());
    if level != "basins" && level != "memories" {
        return Ok((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "level must be \"basins\" or \"memories\""
            })),
        )
            .into_response());
    }
    let offset = query.offset.unwrap_or(0);
    // Personal stores are 60k+; the old 50k ceiling silently truncated the field and
    // desynced live dream frames (n=full) from the UUID-backed splat list.
    let limit = query.limit.unwrap_or(50_000).clamp(1, 250_000);
    let labels = snapshot
        .basins
        .iter()
        .map(|basin| (basin.id.as_str(), basin.label.as_str()))
        .collect::<std::collections::HashMap<_, _>>();
    let all = if level == "basins" {
        snapshot
            .basins
            .iter()
            .map(|basin| {
                // Use majority *actual* domain string — do not invent scifact/urban when
                // the store is mixed-control (all domain="mixed").
                let mut counts: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                for id in &basin.member_ids {
                    if let Some(s) = snapshot.splats.iter().find(|s| s.memory_id == *id) {
                        let d = s.domain.to_ascii_lowercase();
                        *counts.entry(d).or_default() += 1;
                    }
                }
                let domain = counts
                    .into_iter()
                    .max_by_key(|(_, n)| *n)
                    .map(|(d, _)| d)
                    .unwrap_or_else(|| "mixed".into());
                ViewerSplat {
                    id: basin.id.clone(),
                    position: basin.centroid,
                    scale: {
                        let radius = 1.0 + (basin.member_ids.len() as f32 + 1.0).ln() * 0.25;
                        [radius, radius * 0.55, radius * 0.55]
                    },
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    color: domain_state_color(&domain, Some(basin.id.as_str()), 0.0),
                    radiance: basin.stability.max(0.2),
                    mass: basin.member_ids.len() as f32,
                    basin_id: Some(basin.id.clone()),
                    label: Some(basin.label.clone()),
                }
            })
            .collect::<Vec<_>>()
    } else {
        snapshot
            .splats
            .iter()
            // Muted duplicates are collapsed into a representative and are not part of
            // the field; serving them would put the removed clump back on the screen.
            .filter(|splat| !splat.muted)
            .filter(|splat| {
                query
                    .basin_id
                    .as_ref()
                    .is_none_or(|id| splat.basin_id.as_ref() == Some(id))
            })
            .map(|splat| ViewerSplat {
                id: splat.memory_id.to_string(),
                position: splat.position,
                scale: {
                    // Ingest stamps a constant [1.35, 0.38, 0.38]; dream never
                    // rewrites it. Size the served Gaussian from the well it sits in.
                    let n = splat
                        .basin_id
                        .as_ref()
                        .and_then(|id| {
                            snapshot
                                .basins
                                .iter()
                                .find(|basin| basin.id == *id)
                                .map(|basin| basin.member_ids.len())
                        })
                        .unwrap_or(1)
                        .max(1);
                    let major = 0.85 + (n as f32 + 1.0).ln() * 0.22;
                    [major, major * 0.42, major * 0.42]
                },
                rotation: splat.rotation,
                color: domain_state_color(&splat.domain, splat.basin_id.as_deref(), 0.0),
                radiance: splat.radiance,
                mass: splat.mass,
                basin_id: splat.basin_id.clone(),
                label: splat
                    .basin_id
                    .as_deref()
                    .and_then(|id| labels.get(id))
                    .map(|label| (*label).to_string()),
            })
            .collect::<Vec<_>>()
    };
    let total = all.len();
    let splats = all.into_iter().skip(offset).take(limit).collect();
    Ok(Json(SplatPage {
        level,
        total,
        offset,
        splats,
    })
    .into_response())
}

async fn memory(
    State(service): State<Arc<MemoryService>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let id = Uuid::parse_str(&id)?;
    Ok(match service.record(id) {
        Some(memory) => Json(WireMemoryRecord::from(&memory)).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    })
}

async fn sidecar_get(
    State(service): State<Arc<MemoryService>>,
    Path(id): Path<String>,
) -> Result<Json<SidecarState>, ApiError> {
    let id = Uuid::parse_str(&id)?;
    Ok(Json(service.sidecar_get(id)))
}

async fn sidecar_put(
    State(service): State<Arc<MemoryService>>,
    Path(id): Path<String>,
    Json(state): Json<SidecarState>,
) -> Result<Json<SidecarState>, ApiError> {
    let id = Uuid::parse_str(&id)?;
    service.sidecar_put(id, state)?;
    Ok(Json(service.sidecar_get(id)))
}

async fn search(
    State(service): State<Arc<MemoryService>>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<WireRecallHit>>, ApiError> {
    let limit = query.limit.unwrap_or(10).clamp(1, 100);
    let hits = service
        .recall(&query.q, limit, &RecallFilters::default())
        .await?;
    Ok(Json(hits.iter().map(WireRecallHit::from).collect()))
}

async fn steer(
    State(service): State<Arc<MemoryService>>,
    Json(body): Json<SteerBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let op = match body.op.as_deref() {
        None | Some("") => InversionOp::Polarity,
        Some(name) => InversionOp::parse(name)?,
    };
    let axis = match body.axis.as_deref() {
        None | Some("") => crate::inversion::SteerAxisMode::Auto,
        Some(name) => crate::inversion::SteerAxisMode::parse(name)?,
    };
    let opts = SteerOpts {
        gain: body.gain,
        op,
        mass: body.mass,
        basin_id: body.basin_id,
        basin_locked: body.lock,
        axis,
    };
    if !body.addresses.is_empty() {
        let budget = body.budget.unwrap_or(body.gain.abs());
        let report = service
            .steer_addresses(body.addresses, opts, budget)
            .await?;
        return Ok(Json(serde_json::to_value(report)?));
    }
    let id_str = body
        .memory_id
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("steer needs memory_id or addresses"))?;
    let id = Uuid::parse_str(id_str)?;
    let report: SteerReport = service.steer(id, opts).await?;
    Ok(Json(serde_json::to_value(report)?))
}

async fn get_packet(
    State(service): State<Arc<MemoryService>>,
    Path(id): Path<String>,
    Query(query): Query<PacketQuery>,
) -> Result<Json<MemoryPacket>, ApiError> {
    let id = Uuid::parse_str(&id)?;
    let cb = open_codebook(query.codebook.as_deref(), query.unicode)?;
    Ok(Json(service.pack_packet(id, cb.as_ref())?))
}

async fn post_packet(
    State(service): State<Arc<MemoryService>>,
    Json(body): Json<UnpackBody>,
) -> Result<Json<SteerReport>, ApiError> {
    let want_cb = body.packet.unicode.is_some() || body.codebook.is_some();
    let cb = open_codebook(body.codebook.as_deref(), want_cb)?;
    let override_id = body
        .id_override
        .as_deref()
        .map(Uuid::parse_str)
        .transpose()?;
    Ok(Json(
        service
            .unpack_packet(&body.packet, cb.as_ref(), override_id)
            .await?,
    ))
}

#[derive(Debug, Deserialize)]
struct DreamBody {
    emit_every: Option<usize>,
}

async fn start_dream(
    State(service): State<Arc<MemoryService>>,
    body: Option<Json<DreamBody>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let emit_every = body.and_then(|Json(b)| b.emit_every).unwrap_or(5).max(1);
    if service.is_dreaming() {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": "dream already running"
        })));
    }
    let svc = service.clone();
    tokio::spawn(async move {
        match svc.dream_live(emit_every).await {
            Ok(report) => eprintln!(
                "live dream finished steps={} ke={:.4} basins={}",
                report.steps, report.kinetic_energy, report.basins
            ),
            Err(err) => eprintln!("live dream failed: {err:#}"),
        }
    });
    Ok(Json(serde_json::json!({
        "ok": true,
        "emit_every": emit_every,
        "note": "subscribe GET /api/live for frames"
    })))
}

async fn recolor(
    State(service): State<Arc<MemoryService>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    service.recolor_field().await?;
    Ok(Json(serde_json::json!({"ok": true})))
}

#[derive(Debug, Deserialize)]
struct FlipQuery {
    q: String,
    k: Option<usize>,
    cand: Option<usize>,
    /// comma-separated gold doc ids
    gold: Option<String>,
    domain: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FlipBody {
    q: String,
    k: Option<usize>,
    cand: Option<usize>,
    gold: Option<Vec<String>>,
    domain: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MissClassifyBody {
    dataset: String,
    k: Option<usize>,
    cand: Option<usize>,
    limit: Option<usize>,
    /// If true, include per-query rows (large). Default false → summary only.
    #[serde(default)]
    full: bool,
}

async fn flip_get(
    State(service): State<Arc<MemoryService>>,
    Query(query): Query<FlipQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let golds: Vec<(String, u32)> = query
        .gold
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| (s.to_string(), 1u32))
        .collect();
    let report = crate::eval::flip_query(
        &service,
        &query.q,
        query.k.unwrap_or(10),
        query.cand.unwrap_or(80),
        &golds,
        query.domain.as_deref(),
    )
    .await?;
    Ok(Json(serde_json::to_value(report)?))
}

async fn flip_post(
    State(service): State<Arc<MemoryService>>,
    Json(body): Json<FlipBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let golds: Vec<(String, u32)> = body
        .gold
        .unwrap_or_default()
        .into_iter()
        .map(|s| (s, 1u32))
        .collect();
    let report = crate::eval::flip_query(
        &service,
        &body.q,
        body.k.unwrap_or(10),
        body.cand.unwrap_or(80),
        &golds,
        body.domain.as_deref(),
    )
    .await?;
    Ok(Json(serde_json::to_value(report)?))
}

#[derive(Debug, Deserialize)]
struct JlensQuery {
    q: String,
    top_k: Option<usize>,
    #[serde(default)]
    verify: bool,
}

#[derive(Debug, Deserialize)]
struct JlensBody {
    q: String,
    top_k: Option<usize>,
    #[serde(default)]
    verify: bool,
}

async fn jlens_get(Query(query): Query<JlensQuery>) -> Result<Json<serde_json::Value>, ApiError> {
    let top_k = query.top_k.unwrap_or(8);
    let report = crate::jlens::readout_verbalizable(&query.q, top_k, query.verify).await?;
    Ok(Json(serde_json::to_value(report)?))
}

async fn jlens_post(Json(body): Json<JlensBody>) -> Result<Json<serde_json::Value>, ApiError> {
    let top_k = body.top_k.unwrap_or(8);
    let report = crate::jlens::readout_verbalizable(&body.q, top_k, body.verify).await?;
    Ok(Json(serde_json::to_value(report)?))
}

async fn miss_classify_post(
    State(service): State<Arc<MemoryService>>,
    Json(body): Json<MissClassifyBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let report = crate::eval::classify_scifact_misses(
        &service,
        std::path::Path::new(&body.dataset),
        body.k.unwrap_or(10),
        body.cand.unwrap_or(80),
        body.limit,
    )
    .await?;
    if body.full {
        Ok(Json(serde_json::to_value(report)?))
    } else {
        Ok(Json(serde_json::json!({
            "queries": report.queries,
            "failing_queries": report.failing_queries,
            "k": report.k,
            "candidate_limit": report.candidate_limit,
            "gold_counts": report.gold_counts,
            "query_primary_counts": report.query_primary_counts,
            "sample_failing": report.rows.iter()
                .filter(|r| r.recall_at_k < 1.0)
                .take(20)
                .map(|r| serde_json::json!({
                    "query_id": r.query_id,
                    "query_text": r.query_text.chars().take(160).collect::<String>(),
                    "recall_at_k": r.recall_at_k,
                    "primary_class": r.primary_class,
                    "golds": r.golds.iter().map(|g| serde_json::json!({
                        "doc_id": g.doc_id,
                        "class": g.class,
                        "in_bm25": g.in_bm25,
                        "in_dense": g.in_dense,
                        "hybrid_rank": g.hybrid_rank,
                    })).collect::<Vec<_>>(),
                }))
                .collect::<Vec<_>>(),
        })))
    }
}

// ---------------------------------------------------------------------------
// Agent tool surface
//
// The stdio MCP server is the right transport for a client that spawns a process.
// For everything else — a script, another agent, a shell — spawning a process to
// ask one question is friction. These three routes put the same tools on HTTP,
// dispatching into `mcp::call_tool` so both transports cannot drift apart.
// HTTP serve is one silo: omit switch_silo from the manifest; POST it is 400.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ToolCallQuery {
    /// Return the full MCP envelope (`content` + `structuredContent`) instead of just
    /// the result. Off by default: callers want the data.
    ///
    /// Kept as a string because a query parameter is not typed, and `?envelope=1` is
    /// what anyone actually writes. Deserializing straight to `bool` rejects it.
    envelope: Option<String>,
}

impl ToolCallQuery {
    fn want_envelope(&self) -> bool {
        match self.envelope.as_deref() {
            // Bare `?envelope` with no value reads as present-and-empty.
            Some("") | Some("1") | Some("true") | Some("yes") | Some("on") => true,
            _ => false,
        }
    }
}

fn http_tool_list() -> serde_json::Value {
    let mut list = crate::mcp::tool_list();
    if let Some(tools) = list
        .get_mut("tools")
        .and_then(serde_json::Value::as_array_mut)
    {
        tools.retain(|tool| {
            tool.get("name").and_then(serde_json::Value::as_str) != Some("switch_silo")
        });
    }
    list
}

/// Known tool, wrong transport. Not unknown-tool 404 — HTTP serve cannot switch silos.
fn switch_silo_http_reject() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": "HTTP serve is one silo; switch_silo is stdio-only",
            "code": -32602,
        })),
    )
}

async fn tools_list() -> Json<serde_json::Value> {
    Json(http_tool_list())
}

async fn tools_call(
    State(service): State<Arc<MemoryService>>,
    Path(name): Path<String>,
    Query(query): Query<ToolCallQuery>,
    body: Option<Json<serde_json::Value>>,
) -> Response {
    if name == "switch_silo" {
        return switch_silo_http_reject().into_response();
    }
    let arguments = body
        .map(|Json(v)| v)
        .unwrap_or_else(|| serde_json::json!({}));
    let params = serde_json::json!({ "name": name, "arguments": arguments });
    // Ask for the shape we actually want rather than unwrapping an envelope after
    // the fact: arrays carry no `structuredContent`, so unwrapping cannot recover
    // them. Both arms still dispatch through the same tool body.
    let result = if query.want_envelope() {
        crate::mcp::call_tool(params, &service).await
    } else {
        crate::mcp::call_tool_raw(params, &service).await
    };
    match result {
        Ok(value) => Json(value).into_response(),
        Err((code, message)) => {
            // -32601 unknown tool, -32602 bad arguments: the caller's fault, not ours.
            let status = match code {
                -32601 => StatusCode::NOT_FOUND,
                -32602 => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(serde_json::json!({"error": message, "code": code})),
            )
                .into_response()
        }
    }
}

async fn mcp_streamable_http(Json(body): Json<serde_json::Value>) -> Response {
    match crate::mcp::handle_streamable_http(body).await {
        Ok(Some(value)) => Json(value).into_response(),
        Ok(None) => StatusCode::ACCEPTED.into_response(),
        Err((code, message)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "error": {"code": code, "message": message},
                "id": null,
            })),
        )
            .into_response(),
    }
}

/// Search with the full filter set. `GET /api/search` takes only `q` and `limit`;
/// this takes `RecallFilters`, which is what makes faceted and scoped search possible.
#[derive(Debug, Deserialize)]
struct RecallBody {
    query: String,
    limit: Option<usize>,
    #[serde(default)]
    filters: RecallFilters,
}

async fn recall_post(
    State(service): State<Arc<MemoryService>>,
    Json(body): Json<RecallBody>,
) -> Result<Json<Vec<WireRecallHit>>, ApiError> {
    let limit = body.limit.unwrap_or(10).clamp(1, 100);
    let hits = service.recall(&body.query, limit, &body.filters).await?;
    Ok(Json(hits.iter().map(WireRecallHit::from).collect()))
}

#[derive(Debug, Deserialize)]
struct IngestBody {
    path: std::path::PathBuf,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    domain: Option<String>,
}

async fn ingest_post(
    State(service): State<Arc<MemoryService>>,
    Json(body): Json<IngestBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !body.path.is_file() {
        return Err(ApiError(anyhow::anyhow!(
            "ingest path is not a file: {}",
            body.path.display()
        )));
    }
    let source = match body.source.as_deref().unwrap_or("agent-jsonl") {
        "agent-jsonl" | "jsonl" => crate::ingest::SourceKind::AgentJsonl,
        "auto" => crate::ingest::SourceKind::Auto,
        other => {
            return Err(ApiError(anyhow::anyhow!(
                "unsupported ingest source {other}"
            )));
        }
    };
    let domain = body.domain.unwrap_or_else(|| "chat".into());
    let (parsed, appended) = service.ingest_path(source, body.path, domain).await?;
    Ok(Json(serde_json::json!({
        "ok": true,
        "appended": appended,
        "parsed": parsed,
    })))
}

async fn backfill_derived_post(
    State(service): State<Arc<MemoryService>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let report = service.backfill_missing_derived().await?;
    Ok(Json(serde_json::to_value(report)?))
}

async fn live_sse(
    State(service): State<Arc<MemoryService>>,
) -> Sse<impl futures_core::stream::Stream<Item = Result<Event, Infallible>>> {
    let seed = service.current_live_frame();
    let rx = service.subscribe_live();
    let stream = async_stream::stream! {
        if let Ok(data) = serde_json::to_string(&seed) {
            yield Ok::<_, Infallible>(Event::default().event("frame").data(data));
        }
        let mut stream = BroadcastStream::new(rx);
        while let Some(item) = stream.next().await {
            match item {
                Ok(frame) => {
                    if let Ok(data) = serde_json::to_string(&frame) {
                        yield Ok(Event::default().event("frame").data(data));
                    }
                }
                Err(_) => {
                    // lag — drop and continue
                }
            }
        }
    };
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(2))
            .text("ping"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_names(list: &serde_json::Value) -> Vec<&str> {
        list["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect()
    }

    #[test]
    fn http_manifest_omits_switch_silo() {
        let http_list = http_tool_list();
        let stdio_list = crate::mcp::tool_list();
        let http = tool_names(&http_list);
        let stdio = tool_names(&stdio_list);
        assert!(stdio.contains(&"switch_silo"));
        assert!(!http.contains(&"switch_silo"));
        assert!(http.contains(&"list_silos"));
        assert_eq!(http.len(), stdio.len() - 1);
    }

    #[test]
    fn switch_silo_http_is_typed_400() {
        let (status, Json(body)) = switch_silo_http_reject();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["code"], -32602);
        let msg = body["error"].as_str().unwrap();
        assert!(msg.contains("one silo"));
        assert!(!msg.contains("unknown tool"));
    }
}

use crate::inversion::InversionOp;
use crate::packet::{MemoryPacket, VqCodebook};
use crate::physics::domain_state_color;
use crate::record::RecallFilters;
use crate::service::{MemoryService, MemoryStatus, SteerOpts, SteerReport};
use crate::wire::{WireBasin, WireMemoryRecord, WireRecallHit};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
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

const INDEX_HTML: &str = include_str!("../assets/index.html");
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
    memory_id: String,
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

fn open_codebook(path: Option<&str>, want: bool) -> Result<Option<VqCodebook>, ApiError> {
    if !want && path.is_none() {
        return Ok(None);
    }
    let path = path.unwrap_or(
        "",
    );
    Ok(Some(VqCodebook::load_json(path)?))
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
    if let Ok(path) = std::env::var("SPLATRAG_LIVE_LOG") {
        if !path.is_empty() {
            service.set_live_log_path(Some(std::path::PathBuf::from(&path)));
            eprintln!("live frame log → {path}");
        }
    }
    // Domain colors on first paint so the field is readable without a dream.
    if let Err(err) = service.recolor_field() {
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
        .layer(CorsLayer::permissive())
        .with_state(service);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("SplatLens listening on http://{bind}");
    eprintln!("  live SSE  GET  /api/live");
    eprintln!("  dream     POST /api/dream  body: {{\"emit_every\":5}}");
    eprintln!("  flip      GET|POST /api/flip?q=…");
    eprintln!("  jlens     GET|POST /api/jlens?q=…  (verbalizable words)");
    eprintln!("  misses    POST /api/miss-classify");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
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
    Json(
        service
            .list_basins()
            .iter()
            .map(WireBasin::from)
            .collect(),
    )
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
    let limit = query.limit.unwrap_or(20_000).clamp(1, 50_000);
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
            .filter(|splat| {
                query
                    .basin_id
                    .as_ref()
                    .is_none_or(|id| splat.basin_id.as_ref() == Some(id))
            })
            .map(|splat| ViewerSplat {
                id: splat.memory_id.to_string(),
                position: splat.position,
                scale: splat.scale,
                rotation: splat.rotation,
                color: domain_state_color(
                    &splat.domain,
                    splat.basin_id.as_deref(),
                    0.0,
                ),
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
) -> Result<Json<SteerReport>, ApiError> {
    let id = Uuid::parse_str(&body.memory_id)?;
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
    Ok(Json(service.steer(id, opts).await?))
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
    Ok(Json(service.unpack_packet(
        &body.packet,
        cb.as_ref(),
        override_id,
    )?))
}

#[derive(Debug, Deserialize)]
struct DreamBody {
    emit_every: Option<usize>,
}

async fn start_dream(
    State(service): State<Arc<MemoryService>>,
    body: Option<Json<DreamBody>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let emit_every = body
        .and_then(|Json(b)| b.emit_every)
        .unwrap_or(5)
        .max(1);
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
    service.recolor_field()?;
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

async fn jlens_get(
    Query(query): Query<JlensQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let top_k = query.top_k.unwrap_or(8);
    let report = crate::jlens::readout_verbalizable(&query.q, top_k, query.verify).await?;
    Ok(Json(serde_json::to_value(report)?))
}

async fn jlens_post(
    Json(body): Json<JlensBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
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

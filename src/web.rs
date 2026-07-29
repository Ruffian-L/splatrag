use crate::record::{RecallFilters, RecallHit};
use crate::service::{MemoryService, MemoryStatus};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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
    let bind = service.config.server.bind.clone();
    let app = Router::new()
        .route("/", get(index))
        .route("/app.js", get(javascript))
        .route("/style.css", get(stylesheet))
        .route("/api/status", get(status))
        .route("/api/basins", get(basins))
        .route("/api/splats", get(splats))
        .route("/api/memories/:id", get(memory))
        .route("/api/search", get(search))
        .layer(CorsLayer::permissive())
        .with_state(service);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("SplatLens listening on http://{bind}");
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

async fn basins(State(service): State<Arc<MemoryService>>) -> Json<Vec<crate::geometry::Basin>> {
    Json(service.list_basins())
}

async fn splats(
    State(service): State<Arc<MemoryService>>,
    Query(query): Query<SplatQuery>,
) -> Json<SplatPage> {
    let snapshot = service.hot_snapshot();
    let level = query.level.unwrap_or_else(|| "basins".into());
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
            .map(|basin| ViewerSplat {
                id: basin.id.clone(),
                position: basin.centroid,
                scale: {
                    let radius = 1.0 + (basin.member_ids.len() as f32 + 1.0).ln() * 0.25;
                    [radius, radius * 0.55, radius * 0.55]
                },
                rotation: [0.0, 0.0, 0.0, 1.0],
                color: [80, 180, 255, 230],
                radiance: basin.stability,
                mass: basin.member_ids.len() as f32,
                basin_id: Some(basin.id.clone()),
                label: Some(basin.label.clone()),
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
                color: splat.color_rgba,
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
    Json(SplatPage {
        level,
        total,
        offset,
        splats,
    })
}

async fn memory(
    State(service): State<Arc<MemoryService>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let id = Uuid::parse_str(&id)?;
    Ok(match service.record(id) {
        Some(memory) => Json(memory).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    })
}

async fn search(
    State(service): State<Arc<MemoryService>>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<RecallHit>>, ApiError> {
    Ok(Json(
        service
            .recall(
                &query.q,
                query.limit.unwrap_or(10),
                &RecallFilters::default(),
            )
            .await?,
    ))
}

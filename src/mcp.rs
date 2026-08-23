use crate::config::AppConfig;
use crate::record::{MemoryRecord, RecallFilters};
use crate::service::MemoryService;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// One MCP process, many silos. Default store is whatever `--config` opened.
/// `switch_silo` or a per-call `silo` argument opens the others on demand.
struct SiloHub {
    default_id: String,
    current: RwLock<String>,
    cache: Mutex<HashMap<String, Arc<MemoryService>>>,
}

impl SiloHub {
    fn new(default: Arc<MemoryService>) -> Self {
        let id = silo_id_for_scope(&default.config.scope_key);
        let mut cache = HashMap::new();
        cache.insert(id.clone(), default);
        Self {
            default_id: id.clone(),
            current: RwLock::new(id),
            cache: Mutex::new(cache),
        }
    }

    fn current_id(&self) -> String {
        self.current.read().expect("silo current").clone()
    }

    fn resolve(&self, requested: Option<&str>) -> Result<Arc<MemoryService>, (i32, String)> {
        let id = match requested {
            Some(raw) if !raw.trim().is_empty() => normalize_silo(raw)?,
            _ => self.current_id(),
        };
        {
            let cache = self.cache.lock().expect("silo cache");
            if let Some(service) = cache.get(&id) {
                return Ok(service.clone());
            }
        }
        let path = config_path_for_silo(&id)?;
        let config = AppConfig::load(&path).map_err(|error| {
            (
                -32603,
                format!("failed to load silo {id} config {}: {error}", path.display()),
            )
        })?;
        let service = Arc::new(MemoryService::open(config).map_err(|error| {
            (
                -32603,
                format!("failed to open silo {id}: {error}"),
            )
        })?);
        self.cache
            .lock()
            .expect("silo cache")
            .insert(id, service.clone());
        Ok(service)
    }

    fn switch(&self, raw: &str) -> Result<Arc<MemoryService>, (i32, String)> {
        let id = normalize_silo(raw)?;
        let service = self.resolve(Some(&id))?;
        *self.current.write().expect("silo current") = id;
        Ok(service)
    }
}

fn normalize_silo(raw: &str) -> Result<String, (i32, String)> {
    Ok(match raw.trim() {
        "personal" | "main" | "memory" => "personal".into(),
        "niodoo-telemetry" | "niodoo" => "niodoo-telemetry".into(),
        "niodoo-telemetry-stripped" | "niodoo-stripped" | "stripped" => {
            "niodoo-telemetry-stripped".into()
        }
        "telemetry" | "dual-attention" => "telemetry".into(),
        "docs" | "research" => "docs".into(),
        other => {
            return Err((
                -32602,
                format!(
                    "unknown silo '{other}' (personal | niodoo-telemetry | telemetry | docs)"
                ),
            ));
        }
    })
}

fn silo_id_for_scope(scope_key: &str) -> String {
    match scope_key {
        "splatrag-main" => "personal".into(),
        "splatrag-niodoo-telemetry" => "niodoo-telemetry".into(),
        "splatrag-niodoo-telemetry-stripped" => "niodoo-telemetry-stripped".into(),
        "splatrag-telemetry" => "telemetry".into(),
        "splatrag-docs" => "docs".into(),
        other => other.to_string(),
    }
}

fn config_path_for_silo(id: &str) -> Result<PathBuf, (i32, String)> {
    if let Some(path) = catalog_config_for_silo(id) {
        return Ok(path);
    }
    let root = std::env::var_os("SPLATRAG_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let name = match id {
        "personal" => "personal.toml",
        "niodoo-telemetry" => "niodoo-telemetry.toml",
        "niodoo-telemetry-stripped" => "niodoo-telemetry-stripped.toml",
        "telemetry" => "telemetry.toml",
        "docs" => "docs.toml",
        _ => return Err((-32602, format!("no config for silo {id}"))),
    };
    Ok(root.join("configs").join(name))
}

fn catalog_config_for_silo(id: &str) -> Option<PathBuf> {
    let catalog = load_silo_catalog();
    let silos = catalog.as_array()?;
    for silo in silos {
        if silo.get("id").and_then(Value::as_str) == Some(id)
            && let Some(path) = silo.get("config").and_then(Value::as_str)
        {
            return Some(PathBuf::from(path));
        }
    }
    None
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

/// Join a loopback serve bind (`127.0.0.1:8767`) or URL into a tool endpoint.
pub fn tool_endpoint(base: &str, name: &str) -> String {
    let base = http_base_from_serve(base);
    format!("{}/api/tools/{}", base.trim_end_matches('/'), name)
}

pub fn http_base_from_serve(serve: &str) -> String {
    let serve = serve.trim();
    if serve.starts_with("http://") || serve.starts_with("https://") {
        serve.to_string()
    } else {
        format!("http://{serve}")
    }
}

struct ProxyHub {
    default_id: String,
    current: RwLock<String>,
}

impl ProxyHub {
    fn from_env() -> Result<Self, (i32, String)> {
        let default = std::env::var("SPLATRAG_SILO").unwrap_or_else(|_| "personal".into());
        let id = normalize_silo(&default).unwrap_or_else(|_| "personal".into());
        Ok(Self {
            default_id: id.clone(),
            current: RwLock::new(id),
        })
    }

    fn current_id(&self) -> String {
        self.current.read().expect("proxy current").clone()
    }

    fn switch(&self, raw: &str) -> Result<String, (i32, String)> {
        let id = normalize_silo(raw)?;
        *self.current.write().expect("proxy current") = id.clone();
        Ok(id)
    }

    fn url_for(&self, requested: Option<&str>) -> Result<String, (i32, String)> {
        let id = match requested {
            Some(raw) if !raw.trim().is_empty() => normalize_silo(raw)?,
            _ => self.current_id(),
        };
        if id == self.default_id
            && let Ok(url) = std::env::var("SPLATRAG_API_URL")
        {
            return Ok(http_base_from_serve(&url));
        }
        if let Some(serve) = catalog_serve_for_silo(&id) {
            return Ok(http_base_from_serve(&serve));
        }
        if id == "personal" {
            return Ok("http://127.0.0.1:8767".into());
        }
        Err((
            -32603,
            format!("no HTTP serve URL for silo {id}; set silos.json serve= or SPLATRAG_API_URL"),
        ))
    }
}

fn catalog_serve_for_silo(id: &str) -> Option<String> {
    let catalog = load_silo_catalog();
    let silos = catalog.as_array()?;
    for silo in silos {
        if silo.get("id").and_then(Value::as_str) == Some(id) {
            return silo
                .get("serve")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
    }
    None
}

async fn proxy_tool(base: &str, name: &str, arguments: Value) -> Result<Value, (i32, String)> {
    let url = tool_endpoint(base, name);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|error| (-32603, error.to_string()))?;
    let response = client
        .post(&url)
        .json(&arguments)
        .send()
        .await
        .map_err(|error| (-32603, format!("proxy {url}: {error}")))?;
    let status = response.status();
    let body: Value = response
        .json()
        .await
        .map_err(|error| (-32603, format!("proxy {url} body: {error}")))?;
    if !status.is_success() {
        let message = body
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("HTTP tool call failed")
            .to_string();
        let code = body.get("code").and_then(Value::as_i64).unwrap_or(-32603) as i32;
        return Err((code, message));
    }
    Ok(body)
}

/// Stdio MCP that does not open the store. Every mutating/search tool hits the
/// long-lived `splatrag serve` process (one writer).
pub async fn run_stdio_proxy() -> Result<()> {
    let hub = ProxyHub::from_env().map_err(|(_, message)| anyhow!(message))?;
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(request) => request,
            Err(error) => {
                let response = JsonRpcResponse {
                    jsonrpc: "2.0",
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: error.to_string(),
                    }),
                    id: Value::Null,
                };
                write_response(&mut stdout, &response).await?;
                continue;
            }
        };
        let is_notification = request.id.is_none();
        if let Some(response) = handle_proxy(request, &hub).await
            && !is_notification
        {
            write_response(&mut stdout, &response).await?;
        }
    }
    Ok(())
}

async fn handle_proxy(request: JsonRpcRequest, hub: &ProxyHub) -> Option<JsonRpcResponse> {
    let id = request.id.unwrap_or(Value::Null);
    let result = match request.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "splatrag-memory", "version": env!("CARGO_PKG_VERSION")},
            "capabilities": {"tools": {}}
        })),
        "notifications/initialized" | "initialized" => return None,
        "tools/list" => Ok(tool_list()),
        "tools/call" => proxy_call_tool(request.params.unwrap_or(Value::Null), hub).await,
        "ping" => Ok(json!({})),
        _ => Err((-32601, format!("method not found: {}", request.method))),
    };
    Some(match result {
        Ok(result) => JsonRpcResponse {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        },
        Err((code, message)) => JsonRpcResponse {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError { code, message }),
            id,
        },
    })
}

async fn proxy_call_tool(params: Value, hub: &ProxyHub) -> Result<Value, (i32, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| (-32602, "missing tool name".into()))?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    if name == "switch_silo" {
        let silo = arguments
            .get("silo")
            .and_then(Value::as_str)
            .ok_or_else(|| (-32602, "switch_silo requires silo".into()))?;
        let id = hub.switch(silo)?;
        let url = hub.url_for(Some(&id))?;
        return wrap_tool_result(json!({
            "silo": id,
            "default_silo": hub.default_id,
            "api": url,
            "mode": "http-proxy",
        }));
    }
    if name == "list_silos" {
        return wrap_tool_result(json!({
            "silo": hub.current_id(),
            "default_silo": hub.default_id,
            "mode": "http-proxy",
            "switch": "one MCP stdio client: switch_silo or pass silo= ; writes go to splatrag serve",
            "silos": load_silo_catalog(),
        }));
    }
    let requested = arguments.get("silo").and_then(Value::as_str);
    let base = hub.url_for(requested)?;
    let value = proxy_tool(&base, name, arguments).await?;
    wrap_tool_result(value)
}

pub async fn run_stdio(service: Arc<MemoryService>) -> Result<()> {
    let hub = SiloHub::new(service);
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(request) => request,
            Err(error) => {
                let response = JsonRpcResponse {
                    jsonrpc: "2.0",
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: error.to_string(),
                    }),
                    id: Value::Null,
                };
                write_response(&mut stdout, &response).await?;
                continue;
            }
        };
        let is_notification = request.id.is_none();
        if let Some(response) = handle(request, &hub).await
            && !is_notification
        {
            write_response(&mut stdout, &response).await?;
        }
    }
    Ok(())
}

async fn handle(request: JsonRpcRequest, hub: &SiloHub) -> Option<JsonRpcResponse> {
    let id = request.id.unwrap_or(Value::Null);
    let result = match request.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "splatrag-memory", "version": env!("CARGO_PKG_VERSION")},
            "capabilities": {"tools": {}}
        })),
        "notifications/initialized" | "initialized" => return None,
        "tools/list" => Ok(tool_list()),
        "tools/call" => call_tool_on_hub(request.params.unwrap_or(Value::Null), hub).await,
        "ping" => Ok(json!({})),
        _ => Err((-32601, format!("method not found: {}", request.method))),
    };
    Some(match result {
        Ok(result) => JsonRpcResponse {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        },
        Err((code, message)) => JsonRpcResponse {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError { code, message }),
            id,
        },
    })
}

async fn call_tool_on_hub(params: Value, hub: &SiloHub) -> Result<Value, (i32, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("");
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    if name == "switch_silo" {
        let silo = arguments
            .get("silo")
            .and_then(Value::as_str)
            .ok_or_else(|| (-32602, "switch_silo requires silo".into()))?;
        let service = hub.switch(silo)?;
        return wrap_tool_result(json!({
            "silo": hub.current_id(),
            "default_silo": hub.default_id,
            "scope_key": service.config.scope_key,
            "data_dir": service.config.data_dir,
        }));
    }
    if name == "list_silos" {
        let service = hub.resolve(None)?;
        return wrap_tool_result(list_silos(&service, Some(hub)));
    }
    let requested = arguments.get("silo").and_then(Value::as_str);
    let service = hub.resolve(requested)?;
    call_tool(params, &service).await
}

fn wrap_tool_result(value: Value) -> Result<Value, (i32, String)> {
    let text = serde_json::to_string_pretty(&value).map_err(internal_error)?;
    let mut envelope = json!({
        "content": [{
            "type": "text",
            "text": text
        }]
    });
    if value.is_object() {
        envelope["structuredContent"] = value;
    }
    Ok(envelope)
}

/// Dispatch one tool call. Shared by the stdio MCP server and the HTTP tool surface in
/// [`crate::web`], so an agent gets identical behaviour over either transport.
/// Run one tool and return its *natural* JSON value — array, object, whatever the
/// tool actually produces. No MCP envelope; `call_tool` wraps this for stdio.
pub(crate) async fn call_tool_raw(
    params: Value,
    service: &MemoryService,
) -> Result<Value, (i32, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| (-32602, "missing tool name".into()))?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let value = match name {
        "remember" => {
            let text = arguments
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| (-32602, "remember requires non-empty text".into()))?;
            let source_key = arguments
                .get("source_key")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "mcp/{}",
                        uuid::Uuid::new_v5(
                            &uuid::Uuid::NAMESPACE_OID,
                            format!("{}\0{}", chrono::Utc::now(), text).as_bytes()
                        )
                    )
                });
            let mut record = MemoryRecord::new("mcp", source_key, text);
            record.domain = arguments
                .get("domain")
                .and_then(Value::as_str)
                .unwrap_or("chat")
                .into();
            record.speaker = crate::record::clean_speaker(string_argument(&arguments, "speaker"));
            record.model = string_argument(&arguments, "model");
            record.conversation_id = string_argument(&arguments, "conversation_id");
            serde_json::to_value(service.remember(record).await.map_err(internal_error)?)
                .map_err(internal_error)?
        }
        "steer" => {
            let gain = arguments.get("gain").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let op = arguments
                .get("op")
                .and_then(Value::as_str)
                .map(crate::inversion::InversionOp::parse)
                .transpose()
                .map_err(|error| (-32602, error.to_string()))?
                .unwrap_or_default();
            let mass = arguments
                .get("mass")
                .and_then(Value::as_f64)
                .map(|value| value as f32);
            let axis = arguments
                .get("axis")
                .and_then(Value::as_str)
                .map(crate::inversion::SteerAxisMode::parse)
                .transpose()
                .map_err(|error| (-32602, error.to_string()))?
                .unwrap_or_default();
            let opts = crate::service::SteerOpts {
                gain,
                op,
                mass,
                basin_id: string_argument(&arguments, "basin_id"),
                basin_locked: arguments
                    .get("lock")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                axis,
            };
            let mut addresses: Vec<crate::service::SteerAddress> = arguments
                .get("addresses")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| (-32602, format!("invalid addresses: {error}")))?
                .unwrap_or_default();
            // Empty addresses + basin_id + no memory_id = one room (CLI --basin).
            if addresses.is_empty()
                && arguments.get("memory_id").and_then(Value::as_str).is_none()
                && let Some(basin_id) = string_argument(&arguments, "basin_id")
            {
                addresses.push(crate::service::SteerAddress {
                    memory_id: None,
                    basin_id: Some(basin_id),
                    share: None,
                });
            }
            if !addresses.is_empty() {
                let budget = arguments
                    .get("budget")
                    .and_then(Value::as_f64)
                    .map(|v| v as f32)
                    .unwrap_or(gain.abs());
                serde_json::to_value(
                    service
                        .steer_addresses(addresses, opts, budget)
                        .await
                        .map_err(internal_error)?,
                )
                .map_err(internal_error)?
            } else {
                let memory_id = arguments
                    .get("memory_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        (
                            -32602,
                            "steer needs memory_id, addresses, or basin_id".into(),
                        )
                    })?;
                let id = uuid::Uuid::parse_str(memory_id)
                    .map_err(|error| (-32602, format!("invalid memory_id: {error}")))?;
                serde_json::to_value(service.steer(id, opts).await.map_err(internal_error)?)
                    .map_err(internal_error)?
            }
        }
        "pick" => {
            let prompt = arguments
                .get("prompt")
                .or_else(|| arguments.get("query"))
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| (-32602, "pick requires non-empty prompt".into()))?;
            let limit = arguments.get("limit").and_then(Value::as_u64).unwrap_or(3) as usize;
            let min_score = arguments
                .get("min_score")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32;
            let text_budget = arguments
                .get("text_budget")
                .and_then(Value::as_u64)
                .unwrap_or(crate::pick::DEFAULT_INSPECT_CHARS as u64)
                as usize;
            let gain_budget = arguments
                .get("budget")
                .and_then(Value::as_f64)
                .unwrap_or(crate::pick::DEFAULT_GAIN_BUDGET as f64) as f32;
            let filters: RecallFilters = arguments
                .get("filters")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| (-32602, error.to_string()))?
                .unwrap_or_default();
            let pick_config = pick_config_from_args(
                &arguments,
                &service.config.pick,
                limit,
                min_score,
                text_budget,
                gain_budget,
            );
            serde_json::to_value(
                service
                    .pick(prompt, &filters, &pick_config)
                    .await
                    .map_err(internal_error)?,
            )
            .map_err(internal_error)?
        }
        "pack_64" => {
            let memory_id = arguments
                .get("memory_id")
                .and_then(Value::as_str)
                .ok_or_else(|| (-32602, "pack_64 requires memory_id".into()))?;
            let id = uuid::Uuid::parse_str(memory_id)
                .map_err(|error| (-32602, format!("invalid memory_id: {error}")))?;
            let unicode = arguments
                .get("unicode")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let cb = if unicode {
                Some(codebook_from(&arguments)?)
            } else {
                None
            };
            serde_json::to_value(
                service
                    .pack_packet(id, cb.as_ref())
                    .map_err(internal_error)?,
            )
            .map_err(internal_error)?
        }
        "combine_64" => {
            let raw_ids = arguments
                .get("memory_ids")
                .and_then(Value::as_array)
                .ok_or_else(|| (-32602, "combine_64 requires memory_ids (array)".to_string()))?;
            let mut ids = Vec::with_capacity(raw_ids.len());
            for entry in raw_ids {
                let raw = entry
                    .as_str()
                    .ok_or_else(|| (-32602, "memory_ids entries must be strings".to_string()))?;
                ids.push(
                    uuid::Uuid::parse_str(raw)
                        .map_err(|error| (-32602, format!("invalid memory_id {raw}: {error}")))?,
                );
            }
            let t = arguments.get("t").and_then(Value::as_f64).map(|t| t as f32);
            serde_json::to_value(service.combine_packet(&ids, t).map_err(internal_error)?)
                .map_err(internal_error)?
        }
        "unpack_64" => {
            let packet: crate::packet::MemoryPacket = arguments
                .get("packet")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| (-32602, error.to_string()))?
                .ok_or_else(|| (-32602, "unpack_64 requires packet object".into()))?;
            let override_id = arguments
                .get("memory_id")
                .and_then(Value::as_str)
                .map(uuid::Uuid::parse_str)
                .transpose()
                .map_err(|error| (-32602, format!("invalid memory_id: {error}")))?;
            let cb = if packet.unicode.is_some() {
                Some(codebook_from(&arguments)?)
            } else {
                None
            };
            serde_json::to_value(
                service
                    .unpack_packet(&packet, cb.as_ref(), override_id)
                    .await
                    .map_err(internal_error)?,
            )
            .map_err(internal_error)?
        }
        "recall" => {
            let query = arguments
                .get("query")
                .and_then(Value::as_str)
                .filter(|query| !query.trim().is_empty())
                .ok_or_else(|| (-32602, "recall requires non-empty query".into()))?;
            let limit = arguments.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize;
            let filters: RecallFilters = arguments
                .get("filters")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| (-32602, error.to_string()))?
                .unwrap_or_default();
            serde_json::to_value(
                service
                    .recall(query, limit, &filters)
                    .await
                    .map_err(internal_error)?,
            )
            .map_err(internal_error)?
        }
        "list_basins" => {
            serde_json::to_value(service.list_basin_summaries()).map_err(internal_error)?
        }
        "list_silos" => list_silos(service, None),
        "browse_basin" => {
            let basin_id = arguments
                .get("basin_id")
                .and_then(Value::as_str)
                .ok_or_else(|| (-32602, "browse_basin requires basin_id".into()))?;
            let offset = arguments.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let limit = arguments.get("limit").and_then(Value::as_u64).unwrap_or(50) as usize;
            serde_json::to_value(service.browse_basin(basin_id, offset, limit))
                .map_err(internal_error)?
        }
        "memory_status" => serde_json::to_value(service.status().await.map_err(internal_error)?)
            .map_err(internal_error)?,
        "ensure_ready" => {
            // Green-light stack: never leave the agent saying "it's down" when
            // embed can be started in two steps. Wrapper also does this at MCP
            // spawn; this tool covers mid-session death.
            let root = std::env::var("SPLATRAG_ROOT").unwrap_or_else(|_| {
                std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| ".".into())
            });
            let script = std::path::Path::new(&root).join("run");
            let output = std::process::Command::new(&script)
                .arg("embed")
                .current_dir(&root)
                .output()
                .map_err(|e| (-32603, format!("ensure_ready spawn failed: {e}")))?;
            let status = service.status().await.map_err(internal_error)?;
            json!({
                "embed_start_ok": output.status.success(),
                "embed_stdout": String::from_utf8_lossy(&output.stdout).trim().to_string(),
                "embed_stderr": String::from_utf8_lossy(&output.stderr).trim().to_string(),
                "memory_status": status,
            })
        }
        _ => return Err((-32601, format!("unknown tool: {name}"))),
    };
    Ok(value)
}

/// The MCP `tools/call` envelope.
///
/// `structuredContent` is only legal for a JSON **object**. `recall`,
/// `list_basins`, `list_silos` and `browse_basin` all return arrays, and emitting
/// a bare array there makes the client reject the whole result at schema
/// validation — the tool reads as connected but every call fails. Arrays travel
/// in the text block alone, which every MCP client already reads.
pub(crate) async fn call_tool(
    params: Value,
    service: &MemoryService,
) -> Result<Value, (i32, String)> {
    let value = call_tool_raw(params, service).await?;
    let text = serde_json::to_string_pretty(&value).map_err(internal_error)?;
    let mut envelope = json!({
        "content": [{
            "type": "text",
            "text": text
        }]
    });
    if value.is_object() {
        envelope["structuredContent"] = value;
    }
    Ok(envelope)
}

/// The tool manifest, in MCP `tools/list` form. Served verbatim over HTTP too.
pub(crate) fn tool_list() -> Value {
    json!({
        "tools": [
            {
                "name": "remember",
                "description": "Append one message to the local SplatRAG memory store.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_key": {"type": "string"},
                        "domain": {"type": "string"},
                        "speaker": {"type": "string"},
                        "model": {"type": "string"},
                        "conversation_id": {"type": "string"},
                        "silo": {"type": "string", "description": "personal | niodoo-telemetry | telemetry | docs. Default: current silo."}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "pick",
                "description": "Choose which memories to steer toward, and how hard. Emits addresses (memory_id + basin), not a text pile. One gain_budget is divided across picks. Use this instead of calling steer twice. semantics_64 is telemetry — do not inject it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 32, "description": "How many memories. Default 3. Budget is divided, not stacked."},
                        "budget": {"type": "number", "description": "Total α shared across picks. Default 0.35."},
                        "min_score": {"type": "number"},
                        "text_budget": {"type": "integer", "description": "Inspect preview chars only. Default 240. Not the payload."},
                        "hybrid": {"type": "boolean", "description": "Ant-slime rerank. ORed with config [pick] hybrid. Personal default off."},
                        "track_usage": {"type": "boolean", "description": "Increment splat.usage on the chosen pick. ORed with config."},
                        "hybrid_beta": {"type": "number", "description": "Hybrid β. Argument if set, else config [pick] hybrid_beta (default 2)."},
                        "easyway": {"type": "boolean", "description": "Collapse high I_R τ this cycle. ORed with config."},
                        "filters": {
                            "type": "object",
                            "properties": {
                                "domains": {"type": "array", "items": {"type": "string"}},
                                "sources": {"type": "array", "items": {"type": "string"}},
                                "speakers": {"type": "array", "items": {"type": "string"}},
                                "basin_id": {"type": "string"},
                                "conversation_id": {"type": "string"}
                            }
                        },
                        "silo": {"type": "string"}
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "steer",
                "description": "Steer addresses. Pass addresses[] + budget for N pins/rooms under one α (split once). Legacy: one memory_id. Basin-only address ({basin_id}, no memory_id) is one object: room 64-d (member mean) on the report, not member 0 + member 1. Do not call this twice to load two memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Legacy one pin. Skip when using addresses."},
                        "addresses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "memory_id": {"type": "string"},
                                    "basin_id": {"type": "string"},
                                    "share": {"type": "number"}
                                }
                            }
                        },
                        "budget": {"type": "number", "description": "Total |α| to split. Default |gain|."},
                        "gain": {"type": "number", "description": "Sign and, for one pin, the α. Negative = invert."},
                        "op": {"type": "string", "description": "polarity | householder | negative_gain"},
                        "mass": {"type": "number", "description": "if set negative, dream repels; not implied by gain"},
                        "basin_id": {"type": "string"},
                        "lock": {"type": "boolean"},
                        "axis": {"type": "string", "description": "auto (default) | self | field | basin"},
                        "silo": {"type": "string"}
                    }
                }
            },
            {
                "name": "pack_64",
                "description": "Pack a splat into a 64D memory packet (optional VQ Unicode).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "unicode": {"type": "boolean"},
                        "codebook": {"type": "string"},
                        "silo": {"type": "string"}
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "combine_64",
                "description": "Combine splats into one 64D packet without landing on the generic centre of the store. Read-only: returns a packet plus its receipt, writes nothing. t is centroid-removal strength — default 0.2 matches real memory density, 0.4 goes below it for exploration, 0.0 is the naive sum and sits on the hub.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_ids": {"type": "array", "items": {"type": "string"}},
                        "t": {"type": "number"},
                        "silo": {"type": "string"}
                    },
                    "required": ["memory_ids"]
                }
            },
            {
                "name": "unpack_64",
                "description": "Apply a 64D packet onto a splat.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "packet": {"type": "object"},
                        "memory_id": {"type": "string"},
                        "codebook": {"type": "string"},
                        "silo": {"type": "string"}
                    },
                    "required": ["packet"]
                }
            },
            {
                "name": "recall",
                "description": "Search Jason's SplatRAG memory (chats, decisions, exports). Prefer this BEFORE large filesystem greps/finds about history, naming, past work, or what Jason meant. Hybrid BM25 + HNSW + radiance. Returns turns with speaker, conversation_id, timestamps, full text. If it fails, call ensure_ready then retry — do not report 'down' without trying that.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "silo": {"type": "string", "description": "personal | niodoo-telemetry | telemetry | docs"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                        "filters": {
                            "type": "object",
                            "properties": {
                                "domains": {"type": "array", "items": {"type": "string"}},
                                "sources": {"type": "array", "items": {"type": "string"}, "description": "e.g. claude, grok-export, gemini-takeout"},
                                "speakers": {"type": "array", "items": {"type": "string"}, "description": "human | collaborator"},
                                "models": {"type": "array", "items": {"type": "string"}},
                                "basin_id": {"type": "string"},
                                "conversation_id": {"type": "string"},
                                "after": {"type": "string", "format": "date-time"},
                                "before": {"type": "string", "format": "date-time"}
                            }
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "list_basins",
                "description": "List memory basins (topic clusters) ordered by size. Returns counts, not every member id — use browse_basin to page members.",
                "inputSchema": {"type": "object", "properties": {"silo": {"type": "string"}}}
            },
            {
                "name": "list_silos",
                "description": "Catalog of isolated SplatRAG stores. One MCP process: switch_silo or pass silo= on other tools.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "switch_silo",
                "description": "Set the current silo for this MCP process. Later tools use it unless they pass silo=.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "silo": {"type": "string", "description": "personal | niodoo-telemetry | telemetry | docs"}
                    },
                    "required": ["silo"]
                }
            },
            {
                "name": "browse_basin",
                "description": "Page through messages in one basin after list_basins.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "basin_id": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "silo": {"type": "string"}
                    },
                    "required": ["basin_id"]
                }
            },
            {
                "name": "memory_status",
                "description": "Memory store health: record counts, basins, dream cycle. Quick check.",
                "inputSchema": {"type": "object", "properties": {"silo": {"type": "string"}}}
            },
            {
                "name": "ensure_ready",
                "description": "Start embedder if needed and report memory_status. Call when recall fails or before heavy memory work. Never leave the user with 'it's down' without trying this first.",
                "inputSchema": {"type": "object", "properties": {"silo": {"type": "string"}}}
            }
        ]
    })
}

/// Resolve the VQ codebook for packet unicode: the `codebook` argument, then
/// `SPLATRAG_CODEBOOK`. The codebook is not distributed with this repo, so a missing one
/// is reported as a bad argument rather than a file-not-found on an empty path.
fn codebook_from(arguments: &Value) -> Result<crate::packet::VqCodebook, (i32, String)> {
    let path = arguments
        .get("codebook")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| std::env::var("SPLATRAG_CODEBOOK").ok())
        .ok_or_else(|| {
            (
                -32602,
                "VQ unicode needs a codebook: pass \"codebook\": \"<codebook_256.json>\" \
                 or set SPLATRAG_CODEBOOK"
                    .to_string(),
            )
        })?;
    crate::packet::VqCodebook::load_json(&path).map_err(internal_error)
}

fn list_silos(service: &MemoryService, hub: Option<&SiloHub>) -> Value {
    let catalog = load_silo_catalog();
    json!({
        "silo": hub.map(SiloHub::current_id).unwrap_or_else(|| silo_id_for_scope(&service.config.scope_key)),
        "default_silo": hub.map(|h| h.default_id.clone()).unwrap_or_else(|| silo_id_for_scope(&service.config.scope_key)),
        "this_scope_key": service.config.scope_key,
        "this_data_dir": service.config.data_dir,
        "switch": "one MCP process: switch_silo or pass silo= on recall/remember/…",
        "silos": catalog,
    })
}

fn load_silo_catalog() -> Value {
    let candidates = [
        std::env::var_os("SPLATRAG_SILOS").map(std::path::PathBuf::from),
        std::env::var_os("SPLATRAG_ROOT").map(|root| {
            std::path::Path::new(&root)
                .join("..")
                .join("silos.json")
        }),
        Some(std::path::PathBuf::from(
            "/home/ruffianl/Hub/Projects/splatrag/silos.json",
        )),
    ];
    for path in candidates.into_iter().flatten() {
        if let Ok(bytes) = std::fs::read(&path)
            && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
        {
            if let Some(silos) = value.get("silos") {
                return silos.clone();
            }
            return value;
        }
    }
    json!([
        {
            "id": "personal",
            "scope_key": "splatrag-main",
            "role": "Chats, decisions, exports. Default agent memory.",
            "mcp": "splatrag"
        },
        {
            "id": "niodoo-telemetry",
            "scope_key": "splatrag-niodoo-telemetry",
            "role": "Full-line niodoo force/run telemetry.",
            "mcp": "splatrag"
        },
        {
            "id": "telemetry",
            "scope_key": "splatrag-telemetry",
            "role": "Dual-attention / dual-stream-soul summaries.",
            "mcp": "splatrag"
        },
        {
            "id": "docs",
            "scope_key": "splatrag-docs",
            "role": "Research notes, north star, evals, build receipts.",
            "mcp": "splatrag"
        }
    ])
}

fn string_argument(arguments: &Value, name: &str) -> Option<String> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Same merge as CLI pick: flag || silo `[pick]`. Personal `PickSettings` stay false.
fn pick_config_from_args(
    arguments: &Value,
    settings: &crate::config::PickSettings,
    limit: usize,
    min_score: f32,
    text_budget: usize,
    gain_budget: f32,
) -> crate::pick::PickConfig {
    crate::pick::PickConfig {
        limit,
        min_score,
        text_budget,
        gain_budget,
        hybrid: arguments
            .get("hybrid")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || settings.hybrid,
        track_usage: arguments
            .get("track_usage")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || settings.track_usage,
        hybrid_beta: arguments
            .get("hybrid_beta")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(settings.hybrid_beta),
        easyway: arguments
            .get("easyway")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || settings.easyway,
    }
}

fn internal_error(error: impl std::fmt::Display) -> (i32, String) {
    (-32000, error.to_string())
}

async fn write_response(stdout: &mut tokio::io::Stdout, response: &JsonRpcResponse) -> Result<()> {
    let mut bytes = serde_json::to_vec(response)?;
    bytes.push(b'\n');
    stdout.write_all(&bytes).await?;
    stdout.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_endpoint_joins_bind_and_url() {
        assert_eq!(
            tool_endpoint("127.0.0.1:8767", "recall"),
            "http://127.0.0.1:8767/api/tools/recall"
        );
        assert_eq!(
            tool_endpoint("http://127.0.0.1:8767/", "remember"),
            "http://127.0.0.1:8767/api/tools/remember"
        );
        assert_eq!(
            http_base_from_serve("127.0.0.1:8770"),
            "http://127.0.0.1:8770"
        );
    }

    #[test]
    fn app_config_pick_hybrid_ors_into_mcp_pick_config() {
        let empty = json!({});
        let mut config = AppConfig::default();
        let off = pick_config_from_args(&empty, &config.pick, 3, 0.0, 240, 0.35);
        assert!(!off.hybrid);
        assert!(!off.track_usage);
        assert!(!off.easyway);
        config.pick.hybrid = true;
        assert!(pick_config_from_args(&empty, &config.pick, 3, 0.0, 240, 0.35).hybrid);
        config.pick.hybrid = false;
        assert!(!pick_config_from_args(&empty, &config.pick, 3, 0.0, 240, 0.35).hybrid);
    }
}

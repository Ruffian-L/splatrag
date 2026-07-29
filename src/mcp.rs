use crate::record::{MemoryRecord, RecallFilters};
use crate::service::MemoryService;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

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

pub async fn run_stdio(service: Arc<MemoryService>) -> Result<()> {
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
        if let Some(response) = handle(request, &service).await
            && !is_notification
        {
            write_response(&mut stdout, &response).await?;
        }
    }
    Ok(())
}

async fn handle(request: JsonRpcRequest, service: &MemoryService) -> Option<JsonRpcResponse> {
    let id = request.id.unwrap_or(Value::Null);
    let result = match request.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "splatrag-memory", "version": env!("CARGO_PKG_VERSION")},
            "capabilities": {"tools": {}}
        })),
        "notifications/initialized" | "initialized" => return None,
        "tools/list" => Ok(tool_list()),
        "tools/call" => call_tool(request.params.unwrap_or(Value::Null), service).await,
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

async fn call_tool(params: Value, service: &MemoryService) -> Result<Value, (i32, String)> {
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
            record.speaker = string_argument(&arguments, "speaker");
            record.model = string_argument(&arguments, "model");
            record.conversation_id = string_argument(&arguments, "conversation_id");
            serde_json::to_value(service.remember(record).await.map_err(internal_error)?)
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
        "list_basins" => serde_json::to_value(service.list_basins()).map_err(internal_error)?,
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
        _ => return Err((-32601, format!("unknown tool: {name}"))),
    };
    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&value).map_err(internal_error)?
        }],
        "structuredContent": value
    }))
}

fn tool_list() -> Value {
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
                        "conversation_id": {"type": "string"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "recall",
                "description": "Hybrid BM25, keyed-HNSW, Qdrant, and splat-radiance memory search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                        "filters": {
                            "type": "object",
                            "properties": {
                                "domains": {"type": "array", "items": {"type": "string"}},
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
                "description": "List stable PH-labeled memory basins ordered by size.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "browse_basin",
                "description": "Page through messages belonging to one memory basin.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "basin_id": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"}
                    },
                    "required": ["basin_id"]
                }
            },
            {
                "name": "memory_status",
                "description": "Report cold-store, ANN, Qdrant, splat, basin, and dream health.",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
    })
}

fn string_argument(arguments: &Value, name: &str) -> Option<String> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .map(str::to_string)
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

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use splatrag::eval::{MAX_RECALL_DROP, evaluate_scifact};
use splatrag::ingest::{IngestReport, Ingestor, SourceKind};
use splatrag::record::{MemoryRecord, RecallFilters};
use splatrag::{AppConfig, MemoryService};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Parser)]
#[command(
    name = "splatrag",
    version,
    about = "Local Gaussian-splat AI memory store"
)]
struct Cli {
    #[arg(long, global = true, default_value = "splatrag.toml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init,
    Doctor,
    Ingest {
        path: PathBuf,
        #[arg(long, value_enum, default_value = "auto")]
        source: SourceKind,
        #[arg(long, default_value = "chat")]
        domain: String,
        #[arg(long)]
        dry_run: bool,
    },
    Remember {
        text: String,
        #[arg(long, default_value = "chat")]
        domain: String,
        #[arg(long)]
        speaker: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        conversation: Option<String>,
        #[arg(long)]
        source_key: Option<String>,
    },
    Recall {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        domain: Vec<String>,
        #[arg(long)]
        model: Vec<String>,
        #[arg(long)]
        basin: Option<String>,
        #[arg(long)]
        conversation: Option<String>,
    },
    Dream {
        #[arg(long)]
        label: bool,
    },
    LabelBasins,
    ListBasins,
    BrowseBasin {
        basin_id: String,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    Status,
    Mcp,
    Serve,
    RebuildIndex,
    Handshake {
        #[arg(long, default_value = "datasets/scifact")]
        dataset: PathBuf,
        #[arg(long)]
        poison: Option<PathBuf>,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        no_ingest: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.command, Command::Init) && !cli.config.exists() {
        AppConfig::write_default(&cli.config)?;
    }
    let config = AppConfig::load(&cli.config)?;

    match cli.command {
        Command::Init => {
            let service = MemoryService::open(config)?;
            service.initialize().await?;
            println!("initialized SplatRAG with config {}", cli.config.display());
        }
        Command::Ingest {
            path,
            source,
            domain,
            dry_run,
        } => {
            if dry_run {
                let ingestor = Ingestor::new(config.quarantine_path());
                let report = ingestor.ingest_path(source, &path, &domain, |_| Ok(()))?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                let service = Arc::new(MemoryService::open(config.clone())?);
                let (report, appended) =
                    ingest_indexed(&config, service, source, path, domain, true).await?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "parsed": report,
                        "appended": appended
                    }))?
                );
            }
        }
        Command::Remember {
            text,
            domain,
            speaker,
            model,
            conversation,
            source_key,
        } => {
            let service = MemoryService::open(config)?;
            let key = source_key.unwrap_or_else(|| {
                format!(
                    "cli/{}",
                    chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
                )
            });
            let mut record = MemoryRecord::new("cli", key, text);
            record.domain = domain;
            record.speaker = speaker;
            record.model = model;
            record.conversation_id = conversation;
            println!(
                "{}",
                serde_json::to_string_pretty(&service.remember(record).await?)?
            );
        }
        Command::Recall {
            query,
            limit,
            domain,
            model,
            basin,
            conversation,
        } => {
            let service = MemoryService::open(config)?;
            let filters = RecallFilters {
                domains: domain,
                models: model,
                basin_id: basin,
                conversation_id: conversation,
                ..RecallFilters::default()
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&service.recall(&query, limit, &filters).await?)?
            );
        }
        Command::Dream { label } => {
            let service = MemoryService::open(config)?;
            let report = service.dream().await?;
            if label {
                let labeled = service.label_basins().await?;
                eprintln!("labeled {labeled} new basins");
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "steps": report.steps,
                    "kinetic_energy": report.kinetic_energy,
                    "basins": report.basins,
                    "persistence_intervals": report.persistence_intervals.len()
                }))?
            );
        }
        Command::LabelBasins => {
            let service = MemoryService::open(config)?;
            println!("{}", service.label_basins().await?);
        }
        Command::ListBasins => {
            let service = MemoryService::open(config)?;
            println!("{}", serde_json::to_string_pretty(&service.list_basins())?);
        }
        Command::BrowseBasin {
            basin_id,
            offset,
            limit,
        } => {
            let service = MemoryService::open(config)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&service.browse_basin(&basin_id, offset, limit))?
            );
        }
        Command::Status => {
            let service = MemoryService::open(config)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&service.status().await?)?
            );
        }
        Command::Doctor => {
            let service = MemoryService::open(config)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&service.doctor().await?)?
            );
        }
        Command::Mcp => {
            splatrag::mcp::run_stdio(Arc::new(MemoryService::open(config)?)).await?;
        }
        Command::Serve => {
            splatrag::web::serve(Arc::new(MemoryService::open(config)?)).await?;
        }
        Command::RebuildIndex => {
            rebuild_indices(config).await?;
        }
        Command::Handshake {
            dataset,
            poison,
            k,
            limit,
            no_ingest,
        } => {
            run_handshake(config, dataset, poison, k, limit, no_ingest).await?;
        }
    }
    Ok(())
}

async fn ingest_indexed(
    config: &AppConfig,
    service: Arc<MemoryService>,
    source: SourceKind,
    path: PathBuf,
    domain: String,
    progress: bool,
) -> Result<(IngestReport, usize)> {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(256);
    let quarantine = config.quarantine_path();
    let parser = tokio::task::spawn_blocking(move || {
        Ingestor::new(quarantine).ingest_path(source, &path, &domain, |record| {
            sender
                .blocking_send(record)
                .context("ingest receiver closed")
        })
    });
    let mut batch = Vec::new();
    let mut appended = 0;
    while let Some(record) = receiver.recv().await {
        batch.push(record);
        if batch.len() >= config.embedding.batch_size {
            appended += service
                .remember_batch(std::mem::take(&mut batch))
                .await?
                .appended;
            if progress {
                eprintln!("indexed {appended} new memories");
            }
        }
    }
    if !batch.is_empty() {
        appended += service.remember_batch(batch).await?.appended;
    }
    Ok((parser.await??, appended))
}

async fn run_handshake(
    config: AppConfig,
    dataset: PathBuf,
    poison: Option<PathBuf>,
    k: usize,
    limit: Option<usize>,
    no_ingest: bool,
) -> Result<()> {
    if k != 10 {
        anyhow::bail!("the preserved HANDSHAKE contract is fixed at k=10");
    }
    let service = Arc::new(MemoryService::open(config.clone())?);
    service.initialize().await?;
    let mut corpus_ingest = None;
    let mut corpus_appended = 0;
    if !no_ingest {
        let corpus = dataset.join("corpus.jsonl");
        let (report, appended) = ingest_indexed(
            &config,
            Arc::clone(&service),
            SourceKind::Jsonl,
            corpus,
            "scifact".into(),
            true,
        )
        .await?;
        corpus_ingest = Some(report);
        corpus_appended = appended;
    }

    let baseline = evaluate_scifact(&service, &dataset, k, limit).await?;
    let mut poison_ingest = None;
    let mut poison_appended = 0;
    let mut after_poison = None;
    let mut recall_drop = None;
    let mut cold_preserved = None;
    let mut dream_summary = None;
    if let Some(poison_path) = poison {
        let (report, appended) = ingest_indexed(
            &config,
            Arc::clone(&service),
            SourceKind::Auto,
            poison_path,
            "urban".into(),
            true,
        )
        .await?;
        poison_ingest = Some(report);
        poison_appended = appended;
        let before_dream = std::fs::read(config.cold_path())?;
        let dream = service.dream().await?;
        let after_dream = std::fs::read(config.cold_path())?;
        cold_preserved = Some(before_dream == after_dream);
        dream_summary = Some(serde_json::json!({
            "steps": dream.steps,
            "kinetic_energy": dream.kinetic_energy,
            "basins": dream.basins,
            "persistence_intervals": dream.persistence_intervals.len()
        }));
        let measured = evaluate_scifact(&service, &dataset, k, limit).await?;
        recall_drop = Some((baseline.recall_at_k - measured.recall_at_k).max(0.0));
        after_poison = Some(measured);
    }

    let complete = after_poison.is_some();
    let passed = baseline.meets_targets()
        && after_poison
            .as_ref()
            .is_none_or(|metrics| metrics.meets_targets())
        && recall_drop.is_none_or(|drop| drop <= MAX_RECALL_DROP)
        && cold_preserved.unwrap_or(true);
    let output = serde_json::json!({
        "complete": complete,
        "passed": passed,
        "dataset": dataset,
        "corpus_ingest": corpus_ingest,
        "corpus_appended": corpus_appended,
        "baseline": baseline,
        "poison_ingest": poison_ingest,
        "poison_appended": poison_appended,
        "dream": dream_summary,
        "after_poison": after_poison,
        "recall_drop": recall_drop,
        "cold_preserved_during_dream": cold_preserved,
        "contract": {
            "recall_at_10_min": splatrag::eval::TARGET_RECALL_AT_10,
            "ndcg_at_10_min": splatrag::eval::TARGET_NDCG_AT_10,
            "max_recall_drop": MAX_RECALL_DROP
        },
        "note": if complete {
            "full SciFact -> poison -> dream -> SciFact handshake"
        } else {
            "baseline only; pass --poison to execute the complete anti-noise handshake"
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    if !passed {
        anyhow::bail!("SplatRAG handshake failed");
    }
    Ok(())
}

async fn rebuild_indices(config: AppConfig) -> Result<()> {
    let indexes = config.data_dir.join("indexes");
    let hot = config.data_dir.join("hot");
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let backup = config.data_dir.join(format!("derived-backup-{stamp}"));
    std::fs::create_dir_all(&backup)?;
    if indexes.exists() {
        std::fs::rename(&indexes, backup.join("indexes"))?;
    }
    if hot.exists() {
        std::fs::rename(&hot, backup.join("hot"))?;
    }
    let service = MemoryService::open(config)?;
    service.reindex_existing().await?;
    println!(
        "derived indexes rebuilt; previous derived state kept at {}",
        backup.display()
    );
    Ok(())
}

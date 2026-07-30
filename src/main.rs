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
    /// Read the archive's images with the local vision model, into a resumable cache.
    ///
    /// Separate from `ingest` on purpose: this runs for hours, and re-running an ingest must not
    /// mean re-running OCR. Safe to interrupt and restart — finished assets are skipped.
    Extract {
        /// Export root, or any directory of assets.
        path: PathBuf,
        /// How many images to read at once. The server was started with `n_slots`; going past it
        /// queues rather than parallelises.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Stop after this many newly-read images.
        #[arg(long)]
        limit: Option<usize>,
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
    /// Steer one existing splat. Gain inverts/amplifies semantics; mass repels if negative.
    /// These are independent: `gain=-0.2` does not set negative mass.
    Steer {
        memory_id: String,
        /// Negative = ontological inversion; positive = amplify. Default 0 = leave semantics.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        gain: f32,
        /// polarity | householder | negative_gain
        #[arg(long, default_value = "polarity")]
        op: String,
        /// If set, overwrite mass. Negative mass repels in dream.
        #[arg(long, allow_hyphen_values = true)]
        mass: Option<f32>,
        #[arg(long)]
        basin: Option<String>,
        #[arg(long)]
        lock: bool,
    },
    /// Pack a splat into a 64D memory packet (JSON). Optional VQ Unicode via --codebook.
    Pack64 {
        memory_id: String,
        /// niodv4 codebook_256.json path for PUA transport
        #[arg(long)]
        codebook: Option<PathBuf>,
        #[arg(long)]
        unicode: bool,
    },
    /// Apply a 64D packet JSON onto a splat (file path or "-" for stdin).
    Unpack64 {
        packet: PathBuf,
        /// Override memory id in the packet
        #[arg(long)]
        memory_id: Option<String>,
        #[arg(long)]
        codebook: Option<PathBuf>,
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
    /// Pick memories to steer a live model toward, with the knobs to apply them.
    ///
    /// Writes a pick set carrying each memory's **text** — the consumer embeds it with its own
    /// encoder. `semantics_64` is telemetry; it is not injectable into another model's residual.
    Pick {
        prompt: String,
        #[arg(long, default_value_t = 3)]
        limit: usize,
        /// Drop candidates scoring below this.
        #[arg(long, default_value_t = 0.0)]
        min_score: f32,
        /// Characters of memory text per pick.
        #[arg(long, default_value_t = splatrag::pick::DEFAULT_TEXT_BUDGET)]
        text_budget: usize,
        /// Total steering α shared across all picks — divided, not paid per pick. Sweep this
        /// against --limit to separate "more memories" from "more total push".
        #[arg(long, default_value_t = splatrag::pick::DEFAULT_GAIN_BUDGET)]
        budget: f32,
        #[arg(long)]
        domain: Vec<String>,
        #[arg(long)]
        basin: Option<String>,
        /// Write to a file instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
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
        Command::Extract {
            path,
            concurrency,
            limit,
        } => {
            run_extract(&config, &path, concurrency, limit).await?;
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
        Command::Steer {
            memory_id,
            gain,
            op,
            mass,
            basin,
            lock,
        } => {
            let service = MemoryService::open(config)?;
            let id = uuid::Uuid::parse_str(&memory_id)
                .with_context(|| format!("invalid memory id {memory_id}"))?;
            let opts = splatrag::service::SteerOpts {
                gain,
                op: splatrag::inversion::InversionOp::parse(&op)?,
                mass,
                basin_id: basin,
                basin_locked: lock,
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&service.steer(id, opts).await?)?
            );
        }
        Command::Pack64 {
            memory_id,
            codebook,
            unicode,
        } => {
            let service = MemoryService::open(config)?;
            let id = uuid::Uuid::parse_str(&memory_id)
                .with_context(|| format!("invalid memory id {memory_id}"))?;
            let cb = load_codebook(codebook, unicode)?;
            let packet = service.pack_packet(id, cb.as_ref())?;
            println!("{}", serde_json::to_string_pretty(&packet)?);
        }
        Command::Unpack64 {
            packet,
            memory_id,
            codebook,
        } => {
            let service = MemoryService::open(config)?;
            let raw = if packet.as_os_str() == "-" {
                let mut buf = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
                buf
            } else {
                std::fs::read_to_string(&packet)
                    .with_context(|| format!("failed to read {}", packet.display()))?
            };
            let packet: splatrag::packet::MemoryPacket = serde_json::from_str(&raw)?;
            let cb = if packet.unicode.is_some() {
                load_codebook(codebook, true)?
            } else {
                load_codebook(codebook, false)?
            };
            let override_id = memory_id
                .as_deref()
                .map(uuid::Uuid::parse_str)
                .transpose()
                .context("invalid --memory-id")?;
            println!(
                "{}",
                serde_json::to_string_pretty(&service.unpack_packet(
                    &packet,
                    cb.as_ref(),
                    override_id
                )?)?
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
        Command::Pick {
            prompt,
            limit,
            min_score,
            text_budget,
            budget,
            domain,
            basin,
            out,
        } => {
            let service = MemoryService::open(config)?;
            let filters = RecallFilters {
                domains: domain,
                basin_id: basin,
                ..RecallFilters::default()
            };
            let pick_config = splatrag::pick::PickConfig {
                limit,
                min_score,
                text_budget,
                gain_budget: budget,
            };
            let picks = service.pick(&prompt, &filters, &pick_config).await?;
            let json = serde_json::to_string_pretty(&picks)?;
            match out {
                Some(path) => {
                    std::fs::write(&path, &json)?;
                    eprintln!(
                        "{} pick(s), confidence {:.3} → {}",
                        picks.picks.len(),
                        picks.confidence,
                        path.display()
                    );
                }
                None => println!("{json}"),
            }
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

/// Default niodv4 juice codebook (256 × 64, PUA transport). Override with --codebook.
const DEFAULT_CODEBOOK: &str = "/media/ruffianl/backup_sandisk/02_projects/niodoo_team_build_code_backup_20260608-150015/worktree/niodv4/experiments/encode_decode/niodv4/results/codebook_256.json";

fn load_codebook(
    path: Option<PathBuf>,
    want: bool,
) -> Result<Option<splatrag::packet::VqCodebook>> {
    if !want && path.is_none() {
        return Ok(None);
    }
    let path = path.unwrap_or_else(|| PathBuf::from(DEFAULT_CODEBOOK));
    Ok(Some(splatrag::packet::VqCodebook::load_json(&path)?))
}

/// OCR pass for export assets. Separate from ingest: long-running, resumable.
async fn run_extract(
    config: &AppConfig,
    path: &std::path::Path,
    concurrency: usize,
    limit: Option<usize>,
) -> Result<()> {
    use splatrag::embedding::LabelingClient;
    use splatrag::ingest::extract::{self, MediaKind, OcrCache};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Semaphore;

    let labeler = LabelingClient::new(config.labeling.clone())?;
    if !labeler.enabled() {
        anyhow::bail!("labeling/vision model is disabled; enable [labeling] to OCR");
    }
    let mut cache = OcrCache::open(config.ocr_cache_path())?;
    let concurrency = concurrency.max(1);
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let done = Arc::new(AtomicUsize::new(cache.len()));
    let newly = Arc::new(AtomicUsize::new(0));

    // Collect image assets under path (export layout or flat tree).
    let mut jobs = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.is_file() {
            continue;
        }
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let child = entry.path();
            if child.is_dir() {
                // Grok export: <uuid>/content with no extension
                let content = child.join("content");
                if content.is_file() {
                    let asset_id = child
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    if !cache.contains(&asset_id) {
                        jobs.push((asset_id, content));
                    }
                } else {
                    stack.push(child);
                }
            }
        }
    }
    if let Some(max) = limit {
        jobs.truncate(max);
    }
    eprintln!(
        "extract: {} cached, {} pending (concurrency={concurrency})",
        cache.len(),
        jobs.len()
    );

    let mut handles = Vec::new();
    for (asset_id, blob) in jobs {
        let permit = semaphore.clone().acquire_owned().await?;
        let labeler = labeler.clone();
        let newly = Arc::clone(&newly);
        let done = Arc::clone(&done);
        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let bytes = tokio::fs::read(&blob).await?;
            let kind = extract::sniff(&bytes);
            if !kind.is_image() {
                return Ok::<_, anyhow::Error>(None);
            }
            let text = labeler.read_image(&bytes, kind.media_type()).await?;
            let entry = extract::Extracted {
                asset_id,
                media_type: kind.media_type().into(),
                kind,
                sha256: splatrag::record::sha256_hex(&bytes),
                bytes: bytes.len() as u64,
                text: Some(text),
            };
            newly.fetch_add(1, Ordering::Relaxed);
            done.fetch_add(1, Ordering::Relaxed);
            Ok(Some(entry))
        }));
    }

    for handle in handles {
        match handle.await? {
            Ok(Some(entry)) => {
                cache.append(entry)?;
                eprint!(".");
            }
            Ok(None) => {}
            Err(error) => eprintln!("\nextract error: {error:#}"),
        }
    }
    eprintln!(
        "\nextract done: {} new, {} total in cache",
        newly.load(Ordering::Relaxed),
        cache.len()
    );
    // Silence unused import if MediaKind only used via kind
    let _ = MediaKind::Png;
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

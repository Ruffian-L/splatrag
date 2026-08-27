use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use splatrag::diagnose::basin_component_curve;
use splatrag::eval::{
    MAX_RECALL_DROP, classify_scifact_misses, evaluate_scifact, evaluate_scifact_roles,
    evaluate_scifact_ultima, flip_query,
};
use splatrag::geometry::HotState;
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
        /// Who is doing the remembering. Part of the record id, so authorship is
        /// structural rather than an optional field a later pass can drop.
        #[arg(long, default_value = "cli")]
        source: String,
        #[arg(long)]
        speaker: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        conversation: Option<String>,
        #[arg(long)]
        source_key: Option<String>,
    },
    /// Steer addresses. One id (legacy) or N `--memory` / `--basin` under one `--budget`.
    /// Gain inverts/amplifies semantics; mass repels if negative. Independent knobs.
    /// Axis defaults to auto (basin peers / field mean) so α actually controls rotation.
    /// A `--basin` with no `--memory` steers the room 64-d (member mean), not member 0.
    Steer {
        memory_id: Option<String>,
        /// Repeatable pin addresses. One call, one budget.
        #[arg(long = "memory", value_name = "UUID")]
        memories: Vec<String>,
        /// Repeatable room addresses. 64-d = member mean, not member 0.
        #[arg(long = "basin", value_name = "ID")]
        basins: Vec<String>,
        /// Total |α| to split across addresses. Default: |gain|. One cap, not N caps.
        #[arg(long, allow_hyphen_values = true)]
        budget: Option<f32>,
        /// Negative = ontological inversion; positive = amplify. Default 0 = leave semantics.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        gain: f32,
        /// polarity | householder | negative_gain
        #[arg(long, default_value = "polarity")]
        op: String,
        /// If set, overwrite mass. Negative mass repels in dream.
        #[arg(long, allow_hyphen_values = true)]
        mass: Option<f32>,
        /// Write this basin id onto a pin. Not a room steer — use `--basin` for that.
        #[arg(long = "assign-basin")]
        assign_basin: Option<String>,
        /// Lock pin membership. Same SteerOpts on `--memory` / `--basin`; not a room lock.
        #[arg(long, default_value_t = false)]
        lock: bool,
        /// Concept axis: auto (default) | self | field | basin
        #[arg(long, default_value = "auto")]
        axis: String,
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
        /// Only this archive: claude | grok-export | gemini | claude-memory. Repeatable.
        #[arg(long)]
        source: Vec<String>,
        /// Only this speaker: human | collaborator. Repeatable.
        #[arg(long)]
        speaker: Vec<String>,
        #[arg(long)]
        model: Vec<String>,
        #[arg(long)]
        basin: Option<String>,
        #[arg(long)]
        conversation: Option<String>,
    },
    /// Pick memories to steer a live model toward, with the knobs to apply them.
    ///
    /// Writes a pick set of **addresses** (`memory_id` + basin). `inspect` is a short preview.
    /// `semantics_64` is telemetry; it is not injectable. Do not embed `inspect` into a residual.
    Pick {
        prompt: String,
        #[arg(long, default_value_t = 3)]
        limit: usize,
        /// Drop candidates scoring below this.
        #[arg(long, default_value_t = 0.0)]
        min_score: f32,
        /// Characters of inspect preview per pick. Not the injection payload.
        #[arg(long, default_value_t = splatrag::pick::DEFAULT_INSPECT_CHARS)]
        text_budget: usize,
        /// Total steering α shared across all picks — divided, not paid per pick. Sweep this
        /// against --limit to separate "more memories" from "more total push".
        #[arg(long, default_value_t = splatrag::pick::DEFAULT_GAIN_BUDGET)]
        budget: f32,
        /// Re-rank the pick pool with hybrid scoring. Also settable in config `[pick] hybrid`.
        #[arg(long, default_value_t = false)]
        hybrid: bool,
        /// Force hybrid off even if the config silo defaults it on.
        #[arg(long, default_value_t = false)]
        no_hybrid: bool,
        /// Scratch-store only. Increment splat.usage on the chosen pick. Not radiance.
        #[arg(long, default_value_t = false)]
        track_usage: bool,
        /// Hybrid β. Default 2 = meaning fights leftover usage. 0 = habit only.
        #[arg(long, default_value_t = 2.0)]
        hybrid_beta: f32,
        /// Collapse high usage this cycle. Off by default.
        #[arg(long, default_value_t = false)]
        easyway: bool,
        #[arg(long)]
        domain: Vec<String>,
        /// Only this archive: claude | grok-export | gemini | claude-memory. Repeatable.
        /// Keeps both sides of the conversation — a source is a relationship, not a speaker.
        #[arg(long)]
        source: Vec<String>,
        /// Only this speaker: human | collaborator. Repeatable. This one drops the other party.
        #[arg(long)]
        speaker: Vec<String>,
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
    /// Name basins with the local labeler (Gemma on :8082).
    /// Uses meaning-scored sample texts so junk clusters name as junk and themes surface.
    LabelBasins {
        /// Re-label basins that already have stable names.
        #[arg(long)]
        force: bool,
    },
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
    /// Rebuild the BM25 (tantivy) index from cold without touching HNSW or dream/hot state.
    CompactLexical,
    /// Embed cold records missing from HNSW/hot. Does not wipe dream geometry.
    BackfillDerived,
    /// Copy 64-d vectors already on hot splats into HNSW. No embedder.
    HydrateAnn,
    /// Open-domain SciFact: poison/chat may occupy top-k. Reports delta vs the filtered 0.8804 freeze.
    Ultima {
        #[arg(long, default_value = "datasets/scifact")]
        dataset: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// BM25 vs dense vs rank miss taxonomy on SciFact (or free-text --query flip).
    MissClassify {
        #[arg(long, default_value = "datasets/scifact")]
        dataset: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Candidate depth per arm (pool for miss detection).
        #[arg(long, default_value_t = 80)]
        cand: usize,
        #[arg(long)]
        limit: Option<usize>,
        /// Free-text flip only (no full qrels walk).
        #[arg(long)]
        query: Option<String>,
        /// Optional gold doc ids for free-text flip (comma-separated).
        #[arg(long)]
        gold: Option<String>,
        #[arg(long)]
        domain: Option<String>,
        /// Write full JSON report (default: stdout summary + path).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Handshake {
        #[arg(long, default_value = "datasets/scifact")]
        dataset: PathBuf,
        #[arg(long)]
        poison: Option<PathBuf>,
        /// Domain the poison corpus is stored under. Cross-domain repulsion keys on this, so a
        /// corpus that is not Urban Dictionary should say so rather than borrow its label.
        #[arg(long, default_value = "urban")]
        poison_domain: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        no_ingest: bool,
    },
    /// Threshold → largest connected component in 64-d and in the 3-d picture.
    /// Does not change clustering.
    DiagnoseBasin {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// 64-d shortlist vs 4096-d rerank vs BM25+64-d / BM25+4096-d / three-arm RRF.
    /// Does not change ranking.
    DiagnoseRoles {
        #[arg(long, default_value = "datasets/scifact")]
        dataset: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 80)]
        cand: usize,
        #[arg(long)]
        limit: Option<usize>,
        /// Skip exact 64-d brute-force vs HNSW (cheap on SciFact-scale stores).
        #[arg(long, default_value_t = false)]
        skip_exact: bool,
        #[arg(long)]
        out: Option<PathBuf>,
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
                let (report, appended) = service.ingest_path(source, path, domain).await?;
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
            source,
            speaker,
            model,
            conversation,
            source_key,
        } => {
            let service = MemoryService::open(config)?;
            let key = source_key.unwrap_or_else(|| {
                format!(
                    "{source}/{}",
                    chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
                )
            });
            let mut record = MemoryRecord::new(source, key, text);
            record.domain = domain;
            record.speaker = splatrag::record::clean_speaker(speaker);
            record.model = model;
            record.conversation_id = conversation;
            println!(
                "{}",
                serde_json::to_string_pretty(&service.remember(record).await?)?
            );
        }
        Command::Steer {
            memory_id,
            memories,
            basins,
            budget,
            gain,
            op,
            mass,
            assign_basin,
            lock,
            axis,
        } => {
            let service = MemoryService::open(config)?;
            let opts = steer_opts_from_flags(gain, &op, mass, assign_basin, lock, &axis)?;
            let batch = !memories.is_empty() || !basins.is_empty();
            if !batch {
                let id_str = memory_id.context("steer needs a memory id, or --memory / --basin")?;
                let id = uuid::Uuid::parse_str(&id_str)
                    .with_context(|| format!("invalid memory id {id_str}"))?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&service.steer(id, opts).await?)?
                );
            } else {
                let addresses = steer_addresses_from_cli(memory_id, memories, basins)?;
                let budget = budget.unwrap_or(gain.abs());
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &service.steer_addresses(addresses, opts, budget).await?
                    )?
                );
            }
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
                serde_json::to_string_pretty(
                    &service
                        .unpack_packet(&packet, cb.as_ref(), override_id)
                        .await?
                )?
            );
        }
        Command::Recall {
            query,
            limit,
            domain,
            source,
            speaker,
            model,
            basin,
            conversation,
        } => {
            let service = MemoryService::open(config)?;
            let filters = RecallFilters {
                domains: domain,
                sources: source,
                speakers: speaker,
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
            hybrid,
            no_hybrid,
            track_usage,
            hybrid_beta,
            easyway,
            domain,
            source,
            speaker,
            basin,
            out,
        } => {
            let service = MemoryService::open(config)?;
            let filters = RecallFilters {
                domains: domain,
                sources: source,
                speakers: speaker,
                basin_id: basin,
                ..RecallFilters::default()
            };
            let pick_config = splatrag::pick::PickConfig {
                limit,
                min_score,
                text_budget,
                gain_budget: budget,
                hybrid: if no_hybrid {
                    false
                } else {
                    hybrid || service.config.pick.hybrid
                },
                track_usage: track_usage || service.config.pick.track_usage,
                hybrid_beta,
                easyway: easyway || service.config.pick.easyway,
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
        Command::LabelBasins { force } => {
            let service = MemoryService::open(config)?;
            let n = service.label_basins_with_options(force).await?;
            eprintln!("labeled {n} basins{}", if force { " (force)" } else { "" });
            println!("{n}");
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
            if std::env::var_os("SPLATRAG_API_URL").is_some() {
                splatrag::mcp::run_stdio_proxy().await?;
            } else {
                splatrag::mcp::run_stdio(Arc::new(MemoryService::open(config)?)).await?;
            }
        }
        Command::Serve => {
            splatrag::web::serve(Arc::new(MemoryService::open(config)?)).await?;
        }
        Command::RebuildIndex => {
            rebuild_indices(config).await?;
        }
        Command::CompactLexical => {
            compact_lexical(config)?;
        }
        Command::BackfillDerived => {
            let service = MemoryService::open(config)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&service.backfill_missing_derived().await?)?
            );
        }
        Command::HydrateAnn => {
            let service = MemoryService::open(config)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&service.hydrate_ann_from_hot().await?)?
            );
        }
        Command::Ultima {
            dataset,
            k,
            limit,
            out,
        } => {
            let service = MemoryService::open(config)?;
            let report = evaluate_scifact_ultima(&service, &dataset, k, limit).await?;
            let summary = serde_json::json!({
                "queries": report.queries,
                "k": report.k,
                "recall_at_k": report.recall_at_k,
                "ndcg_at_k": report.ndcg_at_k,
                "published_filtered_recall": report.published_filtered_recall,
                "published_filtered_ndcg": report.published_filtered_ndcg,
                "delta_recall": report.delta_recall,
                "delta_ndcg": report.delta_ndcg,
                "mean_slots_non_scifact": report.mean_slots_non_scifact,
                "mean_slots_chat": report.mean_slots_chat,
                "mean_slots_urban": report.mean_slots_urban,
                "mean_slots_bm25_only": report.mean_slots_bm25_only,
                "mean_slots_dense_only": report.mean_slots_dense_only,
                "mean_slots_both_arms": report.mean_slots_both_arms,
                "mean_near_dup_slots": report.mean_near_dup_slots,
                "queries_with_displacement": report.queries_with_displacement,
                "queries_with_near_dup": report.queries_with_near_dup,
                "open_domain": report.open_domain,
            });
            eprintln!("{}", serde_json::to_string_pretty(&summary)?);
            let text = serde_json::to_string_pretty(&report)?;
            if let Some(path) = out {
                std::fs::write(&path, &text)
                    .with_context(|| format!("write {}", path.display()))?;
                eprintln!("ultima report → {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::MissClassify {
            dataset,
            k,
            cand,
            limit,
            query,
            gold,
            domain,
            out,
        } => {
            let service = MemoryService::open(config)?;
            if let Some(q) = query {
                let golds: Vec<(String, u32)> = gold
                    .as_deref()
                    .unwrap_or("")
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| (s.to_string(), 1u32))
                    .collect();
                let report = flip_query(&service, &q, k, cand, &golds, domain.as_deref()).await?;
                let text = serde_json::to_string_pretty(&report)?;
                if let Some(path) = out {
                    std::fs::write(&path, &text)
                        .with_context(|| format!("write {}", path.display()))?;
                    eprintln!("wrote {}", path.display());
                }
                println!("{text}");
            } else {
                let report = classify_scifact_misses(&service, &dataset, k, cand, limit).await?;
                let summary = serde_json::json!({
                    "queries": report.queries,
                    "failing_queries": report.failing_queries,
                    "k": report.k,
                    "candidate_limit": report.candidate_limit,
                    "gold_counts": report.gold_counts,
                    "query_primary_counts": report.query_primary_counts,
                });
                eprintln!("{}", serde_json::to_string_pretty(&summary)?);
                let text = serde_json::to_string_pretty(&report)?;
                if let Some(path) = out {
                    std::fs::write(&path, &text)
                        .with_context(|| format!("write {}", path.display()))?;
                    eprintln!("full report → {}", path.display());
                } else {
                    // stdout only summary unless --out (full can be huge)
                    println!("{}", serde_json::to_string_pretty(&summary)?);
                }
            }
        }
        Command::Handshake {
            dataset,
            poison,
            poison_domain,
            k,
            limit,
            no_ingest,
        } => {
            run_handshake(config, dataset, poison, poison_domain, k, limit, no_ingest).await?;
        }
        Command::DiagnoseBasin { out } => {
            let hot = HotState::load(&config.hot_state_path())?;
            eprintln!(
                "basin curve on {} live splats (cycle {})",
                hot.splats.iter().filter(|s| !s.muted).count(),
                hot.dream_cycle
            );
            let report = basin_component_curve(&hot);
            let text = serde_json::to_string_pretty(&report)?;
            if let Some(path) = out {
                std::fs::write(&path, &text)
                    .with_context(|| format!("write {}", path.display()))?;
                eprintln!("basin curve → {}", path.display());
            }
            println!("{text}");
        }
        Command::DiagnoseRoles {
            dataset,
            k,
            cand,
            limit,
            skip_exact,
            out,
        } => {
            let service = MemoryService::open(config)?;
            let report =
                evaluate_scifact_roles(&service, &dataset, k, cand, limit, !skip_exact).await?;
            let text = serde_json::to_string_pretty(&report)?;
            eprintln!("{text}");
            if let Some(path) = out {
                std::fs::write(&path, &text)
                    .with_context(|| format!("write {}", path.display()))?;
                eprintln!("roles report → {}", path.display());
            }
            println!("{text}");
        }
    }
    Ok(())
}

/// `--lock` / `--assign-basin` belong on SteerOpts, not on `--basin` room addresses.
fn steer_opts_from_flags(
    gain: f32,
    op: &str,
    mass: Option<f32>,
    assign_basin: Option<String>,
    lock: bool,
    axis: &str,
) -> Result<splatrag::service::SteerOpts> {
    Ok(splatrag::service::SteerOpts {
        gain,
        op: splatrag::inversion::InversionOp::parse(op)?,
        mass,
        basin_id: assign_basin,
        basin_locked: lock,
        axis: splatrag::inversion::SteerAxisMode::parse(axis)?,
    })
}

fn steer_addresses_from_cli(
    memory_id: Option<String>,
    memories: Vec<String>,
    basins: Vec<String>,
) -> Result<Vec<splatrag::service::SteerAddress>> {
    let mut addresses = Vec::new();
    if let Some(id_str) = memory_id {
        let id = uuid::Uuid::parse_str(&id_str)
            .with_context(|| format!("invalid memory id {id_str}"))?;
        addresses.push(splatrag::service::SteerAddress {
            memory_id: Some(id),
            basin_id: None,
            share: None,
        });
    }
    for raw in memories {
        let id = uuid::Uuid::parse_str(&raw).with_context(|| format!("invalid --memory {raw}"))?;
        addresses.push(splatrag::service::SteerAddress {
            memory_id: Some(id),
            basin_id: None,
            share: None,
        });
    }
    for raw in basins {
        addresses.push(splatrag::service::SteerAddress {
            memory_id: None,
            basin_id: Some(raw),
            share: None,
        });
    }
    Ok(addresses)
}

/// Resolve the niodv4 VQ codebook (256 × 64, PUA transport) for packet unicode.
///
/// `--codebook` wins, then `SPLATRAG_CODEBOOK`. There is no built-in default: the codebook is
/// not distributed with this repo, so asking for unicode without one is a mistake worth naming
/// rather than a file-not-found on an empty path.
fn load_codebook(
    path: Option<PathBuf>,
    want: bool,
) -> Result<Option<splatrag::packet::VqCodebook>> {
    let path = path.or_else(|| std::env::var_os("SPLATRAG_CODEBOOK").map(PathBuf::from));
    match path {
        Some(path) => Ok(Some(splatrag::packet::VqCodebook::load_json(&path)?)),
        None if want => anyhow::bail!(
            "VQ unicode needs a codebook: pass --codebook <codebook_256.json> \
             or set SPLATRAG_CODEBOOK"
        ),
        None => Ok(None),
    }
}

/// OCR pass for export assets. Separate from ingest: long-running, resumable.
async fn run_extract(
    config: &AppConfig,
    path: &std::path::Path,
    concurrency: usize,
    limit: Option<usize>,
) -> Result<()> {
    use splatrag::embedding::LabelingClient;
    use splatrag::ingest::extract::{self, OcrCache};
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
    Ok(())
}

async fn ingest_indexed(
    _config: &AppConfig,
    service: Arc<MemoryService>,
    source: SourceKind,
    path: PathBuf,
    domain: String,
    _progress: bool,
) -> Result<(IngestReport, usize)> {
    service.ingest_path(source, path, domain).await
}

async fn run_handshake(
    config: AppConfig,
    dataset: PathBuf,
    poison: Option<PathBuf>,
    poison_domain: String,
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
        // Domain is CLI-controlled (`--poison-domain`, default urban) so cross-domain
        // repulsion keys on the same string the claim receipts report.
        let (report, appended) = ingest_indexed(
            &config,
            Arc::clone(&service),
            SourceKind::Auto,
            poison_path,
            poison_domain.clone(),
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
        "poison_domain": poison_domain,
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

fn compact_lexical(config: AppConfig) -> Result<()> {
    let cold = splatrag::cold_store::ColdStore::open(config.cold_path())?;
    let records = cold.load_all()?;
    let dest = config.lexical_path();
    let parent = dest
        .parent()
        .ok_or_else(|| anyhow::anyhow!("lexical path has no parent: {}", dest.display()))?;
    let tmp = parent.join("tantivy.new");
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let backup = parent.join(format!("tantivy.old-{stamp}"));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    eprintln!(
        "compacting lexical index: {} records → {}",
        records.len(),
        tmp.display()
    );
    let built = splatrag::lexical::LexicalIndex::rebuild(&tmp, records)?;
    drop(built);
    if dest.exists() {
        std::fs::rename(&dest, &backup)?;
    }
    std::fs::rename(&tmp, &dest)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope_key": config.scope_key,
            "lexical": dest,
            "previous": backup,
        }))?
    );
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

#[cfg(test)]
mod cli_parse_tests {
    use super::*;

    #[test]
    fn batch_steer_lock_and_assign_basin_go_to_steer_opts() {
        let cli = Cli::try_parse_from([
            "splatrag",
            "steer",
            "--memory",
            "00000000-0000-0000-0000-000000000001",
            "--basin",
            "room-1",
            "--assign-basin",
            "home",
            "--lock",
            "--gain",
            "-0.2",
            "--budget",
            "0.2",
        ])
        .expect("batch steer should parse");
        let Command::Steer {
            memory_id,
            memories,
            basins,
            budget,
            gain,
            op,
            mass,
            assign_basin,
            lock,
            axis,
        } = cli.command
        else {
            panic!("expected steer");
        };
        assert!(memory_id.is_none());
        assert_eq!(memories.len(), 1);
        assert_eq!(basins, ["room-1"]);
        assert_eq!(budget, Some(0.2));
        let opts = steer_opts_from_flags(gain, &op, mass, assign_basin, lock, &axis).unwrap();
        assert!(opts.basin_locked, "--lock must reach SteerOpts on batch");
        assert_eq!(opts.basin_id.as_deref(), Some("home"));
        assert!(opts.gain < 0.0);
        let addresses = steer_addresses_from_cli(None, memories, basins).unwrap();
        assert_eq!(addresses.len(), 2);
        assert!(addresses[0].memory_id.is_some());
        assert_eq!(addresses[1].basin_id.as_deref(), Some("room-1"));
        assert!(
            addresses
                .iter()
                .all(|a| a.basin_id.as_deref() != Some("home")),
            "--assign-basin is SteerOpts, not a room address"
        );
    }
}

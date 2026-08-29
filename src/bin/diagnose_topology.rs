use anyhow::Result;
use rayon::prelude::*;
use splatrag::ann::KeyedAnn;
use splatrag::geometry::HotState;
use splatrag::topology::discover_basins;
use splatrag::AppConfig;
use std::collections::HashMap;

fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/splatrag-topology-audit-20260829.toml".to_string());
    let config = AppConfig::load(config_path)?;
    let mut hot = HotState::load(&config.hot_state_path())?;
    let ann = KeyedAnn::open(config.ann_graph_path(), 64, &config.retrieval)?;
    let active = hot.splats.iter().filter(|splat| !splat.muted).count();
    let assigned_before = hot
        .splats
        .iter()
        .filter(|splat| !splat.muted && splat.basin_id.is_some())
        .count();
    let index_by_id: HashMap<_, _> = hot
        .splats
        .iter()
        .enumerate()
        .map(|(index, splat)| (splat.memory_id, index))
        .collect();
    let muted = hot.splats.iter().map(|splat| splat.muted).collect::<Vec<_>>();
    let semantic_neighbors = hot
        .splats
        .par_iter()
        .map(|splat| {
            ann.neighbors_for_key(splat.memory_id)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|id| index_by_id.get(&id).copied())
                .filter(|&index| !muted[index])
                .take(32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let previous = hot.basins.clone();
    let (basins, intervals) = discover_basins(
        &mut hot.splats,
        &previous,
        &config.physics,
        Some(&ann),
        Some(&semantic_neighbors),
    );
    let assigned = hot
        .splats
        .iter()
        .filter(|splat| !splat.muted && splat.basin_id.is_some())
        .count();
    let mut sizes = basins
        .iter()
        .map(|basin| basin.member_ids.len())
        .collect::<Vec<_>>();
    sizes.sort_unstable();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "active_splats": active,
            "ann_keys": ann.len(),
            "assigned_before": assigned_before,
            "basins": basins.len(),
            "assigned": assigned,
            "unassigned": active.saturating_sub(assigned),
            "assignment_rate": assigned as f64 / active.max(1) as f64,
            "largest_basin": sizes.last().copied().unwrap_or(0),
            "merge_intervals": intervals.len(),
        }))?
    );
    Ok(())
}

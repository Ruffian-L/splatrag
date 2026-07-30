//! The picker: decide **which** memories a live model should be steered toward, and **how hard**.
//!
//! Retrieval answers "what is relevant". Steering needs two more answers retrieval never gives:
//! which of the relevant memories is worth spending steering budget on, and at what α. That gap is
//! the reason a working memory store and a working steering rig have never actually been joined.
//!
//! # Why this carries text and not vectors
//!
//! SplatRAG semantics are 64-d unit vectors — a matryoshka slice of a Qwen3-Embedding-8B space.
//! A live steering host (hydrodynamic-swarm) works in its own model's residual space: 2560-d for
//! the Gemma 3 4B store, un-normalized, with scar centers at L2 ≈ 141. Those two spaces share no
//! basis and no scale, and no trained 64→2560 decoder exists.
//!
//! Feeding `semantics_64` into a residual would not error. It would steer confidently on noise.
//! So the pick carries the memory's **text**, and the host embeds it with its *own* encoder — the
//! dimension gap stops being a conversion problem and becomes a non-problem. `semantics_64` rides
//! along for telemetry and dedup only, and [`Pick::injection`] says so on the wire.
//!
//! # What actually ranks on a real corpus
//!
//! Measured on the 930-memory store this was built against: `basin_id` puts 817 of 930 in one
//! basin, `radiance` is 4.5 for every splat, and `domain` is `chat` for 896. None of the three can
//! discriminate. Cosine and BM25 do. So the score is the tuned hybrid one, and basin/radiance are
//! reported but never scored here.

use crate::geometry::HotState;
use crate::inversion::ALPHA_MAX;
use crate::record::RecallHit;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Text budget per pick, in characters.
///
/// The consumer embeds this string, so an unbounded payload is a real cost: OCR'd screenshots in
/// this corpus reach tens of kilobytes, which would dominate a mean-pooled embedding with
/// boilerplate and blow the host's context. 2000 chars is ~500 tokens — enough for a full
/// conversational turn, which is the unit memories are stored in.
pub const DEFAULT_TEXT_BUDGET: usize = 2000;

/// Total steering α to spend across the whole pick set — **not** per pick.
///
/// This is the knob that makes a multi-memory experiment mean anything. Loading three memories at
/// full strength each would push 3 × 0.35 = 1.05, far past the 0.40 collapse onset, and the run
/// would degrade for a reason that has nothing to do with how many memories were loaded. Dividing
/// one budget holds total push constant so that "1 vs 3 vs 8 memories" varies only the thing being
/// studied.
pub const DEFAULT_GAIN_BUDGET: f32 = ALPHA_MAX;

/// Z-score above the null at which a query is treated as fully answered by the store.
///
/// A chosen scale, not a measured one — say so rather than implying it was fit. 6σ is deliberately
/// demanding: it takes a top hit far outside the distribution of what a random memory scores before
/// the picker will spend its whole steering budget.
pub const RELEVANCE_FULL_Z: f32 = 6.0;

/// Memories sampled to estimate the null. Sampled by stride rather than at random so the estimate
/// is reproducible across runs — an experiment cannot afford a confidence score that moves on its
/// own between invocations.
pub const NULL_SAMPLES: usize = 256;

/// Candidates the confidence estimate looks at, regardless of how many picks are requested.
///
/// Fixed on purpose. Spread normalization is sensitive to pool depth — look further down the
/// ranking and you find worse scores, which widens the spread and shrinks the normalized gap.
/// Measured on the real store, letting the pool track `--limit` moved confidence from 0.0195 to
/// 0.0103 for the *same query* with the *same top hit*, purely because more picks were asked for.
/// That would confound the one comparison a multi-memory experiment needs to make.
pub const CONFIDENCE_POOL: usize = 8;

#[derive(Debug, Clone)]
pub struct PickConfig {
    pub limit: usize,
    /// Drop candidates scoring below this. 0.0 keeps everything retrieval returned.
    pub min_score: f32,
    pub text_budget: usize,
    /// Total α shared across all picks. See [`DEFAULT_GAIN_BUDGET`].
    pub gain_budget: f32,
}

impl Default for PickConfig {
    fn default() -> Self {
        Self {
            // Few and strong beats many and diluted: every extra pick spends steering budget and
            // pulls the residual toward the mean of the set.
            limit: 3,
            min_score: 0.0,
            text_budget: DEFAULT_TEXT_BUDGET,
            gain_budget: DEFAULT_GAIN_BUDGET,
        }
    }
}

/// What the consumer is allowed to do with a pick's payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Injection {
    /// Embed `text` with your own encoder. `semantics_64` is telemetry — never a residual.
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPickSet {
    pub version: u32,
    pub prompt: String,
    pub generated_at: DateTime<Utc>,
    /// Which encoder produced `semantics_64`, and at what width. A consumer that cannot name the
    /// same encoder must refuse to treat those floats as a vector in its own space. This is the
    /// same guard hydro already applies with `model_dim` / `model_fp` in its TCT header.
    pub source_embedder: String,
    pub source_dim: usize,
    /// Confidence that the store holds an answer at all: the top hit's cosine as a z-score above
    /// the null, scaled by [`RELEVANCE_FULL_Z`]. This is what sets steering strength.
    pub confidence: f32,
    /// Diagnostic only. The gap between the top two hits. Reported because it is informative about
    /// *shape* — low separation with high confidence means several memories are jointly relevant,
    /// which is the multi-memory case — but it must not drive strength. See [`relevance`].
    pub separation: f32,
    /// The baseline `confidence` was measured against.
    pub null: NullModel,
    /// Standard deviations the top hit sits above the null.
    pub top_z: f32,
    /// Sum of `suggested_gain` over every pick. Bounded by the configured budget, so loading more
    /// memories divides the push instead of stacking it. Report this in any multi-memory run —
    /// it is the confound that would otherwise be mistaken for a memory-count effect.
    pub total_suggested_gain: f32,
    pub picks: Vec<Pick>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pick {
    pub memory_id: Uuid,
    /// **The payload.** Embed this with your own encoder.
    pub text: String,
    /// True when `text` was cut to the budget.
    #[serde(default)]
    pub text_truncated: bool,
    pub injection: Injection,
    pub score: f32,
    pub cosine: f32,
    pub bm25: f32,
    /// Telemetry and dedup only. See the module docs.
    pub semantics_64: Vec<f32>,
    /// Steering α already deliberately set on this memory. 0.0 = unsteered. Authoritative.
    pub gain: f32,
    /// Heuristic α proposed by this picker: this pick's share of the set's total budget, scaled by
    /// `confidence`. **Not** validated — a consumer is free to ignore it and use `gain`. Kept
    /// separate from `gain` precisely so that an unproven policy can never masquerade as a
    /// recorded decision.
    pub suggested_gain: f32,
    /// Fraction of the set's steering budget this pick received.
    pub budget_share: f32,
    /// Negative repels in dream. Passed through from the splat.
    pub mass: f32,
    pub basin_id: Option<String>,
    pub basin_label: Option<String>,
    pub domain: String,
    pub source: String,
    pub timestamp: Option<DateTime<Utc>>,
}

/// Build a pick set from retrieval hits plus the settled field.
///
/// `hits` must already be ranked best-first, which is what `MemoryService::recall` returns.
pub fn build(
    prompt: &str,
    hits: &[RecallHit],
    hot: &HotState,
    embedder: &str,
    null: NullModel,
    config: &PickConfig,
) -> MemoryPickSet {
    let kept: Vec<&RecallHit> = hits
        .iter()
        .filter(|hit| hit.scores.final_score >= config.min_score)
        .take(config.limit.max(1))
        .collect();

    // Judged on the full ranked list, not the truncated one: a candidate excluded from steering is
    // still evidence about the shape of the field.
    let top_cosine = hits.first().map(|hit| hit.scores.cosine).unwrap_or(0.0);
    let confidence = relevance(top_cosine, &null);
    let separation = separation(hits);
    let shares = budget_shares(&kept);

    let picks: Vec<Pick> = kept
        .iter()
        .zip(&shares)
        .map(|(hit, share)| {
            let splat = hot
                .splats
                .iter()
                .find(|splat| splat.memory_id == hit.memory.id);
            let (text, text_truncated) = clip(&hit.memory.text, config.text_budget);
            Pick {
                memory_id: hit.memory.id,
                text,
                text_truncated,
                injection: Injection::Text,
                score: hit.scores.final_score,
                cosine: hit.scores.cosine,
                bm25: hit.scores.bm25,
                semantics_64: splat.map(|s| s.semantics.clone()).unwrap_or_default(),
                gain: splat.map(|s| s.gain).unwrap_or(0.0),
                suggested_gain: suggested_gain(confidence, config.gain_budget, *share),
                budget_share: *share,
                // A memory with no splat has never been through a dream. Neutral mass is the
                // honest default; zero would silently delete it from the host's force law.
                mass: splat.map(|s| s.mass).unwrap_or(1.0),
                basin_id: hit.basin_id.clone(),
                basin_label: hit.basin_label.clone(),
                domain: hit.memory.domain.clone(),
                source: hit.memory.source.clone(),
                timestamp: hit.memory.timestamp,
            }
        })
        .collect();

    let total_suggested_gain = picks.iter().map(|pick: &Pick| pick.suggested_gain).sum();

    MemoryPickSet {
        version: 1,
        prompt: prompt.to_string(),
        generated_at: Utc::now(),
        source_embedder: embedder.to_string(),
        source_dim: crate::packet::PACKET_DIM,
        confidence,
        separation,
        null,
        top_z: null.z(top_cosine),
        total_suggested_gain,
        picks,
    }
}

/// Split the steering budget across picks in proportion to score.
///
/// Proportional rather than equal because the picks are not interchangeable — the top hit earned
/// more of the push. Falls back to an even split when scores carry no information (all zero, or
/// negative from a weighting that under-water-marks everything), since an even split is the honest
/// representation of "cannot tell these apart".
fn budget_shares(kept: &[&RecallHit]) -> Vec<f32> {
    if kept.is_empty() {
        return Vec::new();
    }
    let weights: Vec<f32> = kept
        .iter()
        .map(|hit| hit.scores.final_score.max(0.0))
        .collect();
    let total: f32 = weights.iter().sum();
    if total < 1e-6 {
        let even = 1.0 / kept.len() as f32;
        return vec![even; kept.len()];
    }
    weights.iter().map(|weight| weight / total).collect()
}

/// What a *random* memory scores against this query — the baseline relevance has to beat.
///
/// Without a null there is no way to tell "cosine 0.58 is a real match" from "cosine 0.58 is what
/// everything in this store scores against anything", and embeddings are anisotropic enough that
/// the second is usually the case.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NullModel {
    pub mean: f32,
    pub std: f32,
    pub samples: usize,
}

impl NullModel {
    /// Cosine of `query_64` against a stride sample of the field.
    pub fn estimate(query_64: &[f32], splats: &[crate::geometry::Splat], samples: usize) -> Self {
        let usable: Vec<f32> = {
            let stride = (splats.len() / samples.max(1)).max(1);
            splats
                .iter()
                .step_by(stride)
                .take(samples)
                .filter(|splat| splat.semantics.len() == query_64.len())
                .map(|splat| crate::geometry::cosine(query_64, &splat.semantics))
                .collect()
        };
        if usable.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                samples: 0,
            };
        }
        let count = usable.len() as f32;
        let mean = usable.iter().sum::<f32>() / count;
        let variance = usable.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count;
        Self {
            mean,
            std: variance.sqrt(),
            samples: usable.len(),
        }
    }

    /// How many standard deviations above chance a cosine sits.
    pub fn z(&self, cosine: f32) -> f32 {
        if self.std < 1e-6 {
            return 0.0;
        }
        (cosine - self.mean) / self.std
    }
}

/// Confidence that the store actually holds an answer to this query.
///
/// Replaces an earlier attempt that used [`separation`] — the gap between the top two hits — which
/// measurement showed to be **inverted** for this purpose. A nonsense query produces one accidental
/// outlier above a flat pool of junk and therefore a huge gap; a query the archive genuinely covers
/// produces many similarly-good memories and therefore a tiny one. Measured live on the 930-memory
/// store, `"zzzqqq nonexistent gibberish token"` scored separation 0.7512 while
/// `"the physics of friendship"` — a topic with a whole conversation behind it — scored 0.0156.
/// Separation ranks "is there one weird outlier", not "do we know this".
///
/// A null-model z-score asks the question that was actually meant: is the best memory better than
/// what any random memory would score for this query.
pub fn relevance(top_cosine: f32, null: &NullModel) -> f32 {
    if null.samples == 0 {
        return 0.0;
    }
    (null.z(top_cosine) / RELEVANCE_FULL_Z).clamp(0.0, 1.0)
}

/// How cleanly the top hit separates from the rest of the candidate pool.
///
/// The gap to the runner-up, measured against the **spread** of the whole pool rather than against
/// the top score. Normalizing by the top score looks reasonable and is wrong here: `final_score`
/// carries an additive term `radiance × weight`, and radiance is a constant on a settled store (4.5
/// for every splat in the corpus this was built against). That constant lands identically on every
/// candidate, so dividing by the absolute top inflates the denominator by an offset that has
/// nothing to do with relevance. Measured live, it drove confidence to 0.0017 on a query whose top
/// two hits were genuinely well separated — which would have made the picker propose α ≈ 0.0002 and
/// steer nothing at all.
///
/// Spread normalization cancels any constant: adding `c` to every score leaves
/// `(s₁−s₂)/(s₁−sₙ)` unchanged.
///
/// Only the first [`CONFIDENCE_POOL`] candidates are considered, so the estimate does not drift
/// with how many picks were requested.
///
/// A pool of one is treated as fully separated. A pool of two carries **no scale information** —
/// there is no third score to say whether the gap is large or small — so this declines to guess and
/// returns 0.0 rather than inventing confidence. `MemoryService::pick` over-fetches precisely so
/// that the pool is deep enough for this estimate.
pub fn separation(hits: &[RecallHit]) -> f32 {
    let scores: Vec<f32> = hits
        .iter()
        .take(CONFIDENCE_POOL)
        .map(|hit| hit.scores.final_score)
        .collect();
    match scores.as_slice() {
        [] => 0.0,
        [_] => 1.0,
        [_, _] => 0.0,
        [best, second, .., worst] => {
            let spread = best - worst;
            if spread < 1e-6 {
                return 0.0;
            }
            ((best - second) / spread).clamp(0.0, 1.0)
        }
    }
}

/// Proposed α for one pick: its share of the budget, scaled by confidence.
///
/// Deliberately linear and dull. A tuned curve here would be a claim about how steering strength
/// should track retrieval confidence, and that has not been measured. Two predictions this encodes,
/// stated up front so they can be falsified rather than confirmed after the fact:
///
/// 1. A low-confidence pick injected at full strength should hurt generation more than the same
///    pick injected weakly.
/// 2. With total α held constant, loading more memories should not degrade output merely for being
///    more numerous — if it does, the cause is dilution of direction, not steering magnitude.
pub fn suggested_gain(confidence: f32, budget: f32, share: f32) -> f32 {
    let alpha = confidence.clamp(0.0, 1.0) * budget * share.clamp(0.0, 1.0);
    alpha.clamp(0.0, ALPHA_MAX)
}

/// Clip on a char boundary, preferring the last sentence or line break in the final 20%.
fn clip(text: &str, budget: usize) -> (String, bool) {
    if text.chars().count() <= budget {
        return (text.to_string(), false);
    }
    let head: String = text.chars().take(budget).collect();
    // Cutting mid-sentence leaves a dangling clause that the encoder reads as content. Prefer a
    // real boundary when one is close enough that little is lost.
    let floor = head.len() * 4 / 5;
    let cut = head
        .rfind(['.', '\n', '!', '?'])
        .filter(|index| *index >= floor)
        .map(|index| index + 1)
        .unwrap_or(head.len());
    (head[..cut].trim_end().to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Splat;
    use crate::record::{MemoryRecord, RecallContext, RecallHit, ScoreBreakdown};

    fn hit(key: &str, text: &str, score: f32) -> RecallHit {
        RecallHit {
            memory: MemoryRecord::new("test", key, text),
            context: RecallContext::default(),
            basin_id: Some("basin-a".into()),
            basin_label: Some("Testing".into()),
            scores: ScoreBreakdown {
                bm25: 1.0,
                cosine: 0.8,
                radiance: 4.5,
                radiance_weight: 5.0,
                final_score: score,
            },
        }
    }

    /// A null whose mean/std make the fixtures' cosine of 0.8 land at full confidence, so budget
    /// and share assertions are not silently zeroed by an unrelated relevance gate.
    fn test_null() -> NullModel {
        NullModel {
            mean: 0.2,
            std: 0.1,
            samples: 256,
        }
    }

    fn hot_with(hits: &[RecallHit], gain: f32, mass: f32) -> HotState {
        let mut hot = HotState::default();
        for hit in hits {
            let mut splat = Splat::from_embedding(&hit.memory, &vec![0.125; 64]).unwrap();
            splat.gain = gain;
            splat.mass = mass;
            hot.splats.push(splat);
        }
        hot
    }

    #[test]
    fn pick_carries_text_and_refuses_to_present_semantics_as_injectable() {
        let hits = vec![hit("a", "the physics of friendship", 9.0)];
        let hot = hot_with(&hits, 0.0, 1.0);
        let set = build("friendship", &hits, &hot, "Qwen3-Embedding-8B", test_null(), &PickConfig::default());

        assert_eq!(set.picks.len(), 1);
        let pick = &set.picks[0];
        assert_eq!(pick.text, "the physics of friendship");
        assert_eq!(pick.injection, Injection::Text);
        // The guard the consumer checks before it dares touch the floats.
        assert_eq!(set.source_dim, 64);
        assert_eq!(set.source_embedder, "Qwen3-Embedding-8B");
        assert_eq!(pick.semantics_64.len(), 64);
    }

    #[test]
    fn steered_gain_and_negative_mass_survive_the_wire() {
        let hits = vec![hit("a", "inverted memory", 9.0)];
        let hot = hot_with(&hits, -0.2, -1.0);
        let set = build("q", &hits, &hot, "e", test_null(), &PickConfig::default());

        // gain is the recorded decision; suggested_gain is the picker's unproven opinion. They
        // must never be collapsed into one field.
        assert_eq!(set.picks[0].gain, -0.2);
        assert!(set.picks[0].mass < 0.0);
        assert!(set.picks[0].suggested_gain >= 0.0);
    }

    #[test]
    fn confidence_is_the_margin_not_the_absolute_score() {
        // Same top score, different runner-up: confidence must move, strength must follow.
        let separated = vec![hit("a", "x", 10.0), hit("b", "y", 2.0), hit("c", "z", 1.0)];
        let muddled = vec![hit("a", "x", 10.0), hit("b", "y", 9.8), hit("c", "z", 1.0)];

        let clear = separation(&separated);
        let murky = separation(&muddled);
        assert!(clear > 0.8, "clear winner should separate, got {clear}");
        assert!(murky < 0.1, "near-tie should not, got {murky}");
        assert!(suggested_gain(clear, ALPHA_MAX, 1.0) > suggested_gain(murky, ALPHA_MAX, 1.0));

        // A lone hit is fully separated; nothing at all has no confidence. Two hits carry no
        // scale information, so the picker declines rather than guessing.
        assert_eq!(separation(&separated[..1]), 1.0);
        assert_eq!(separation(&separated[..2]), 0.0);
        assert_eq!(separation(&[]), 0.0);
    }

    /// The live bug this formula exists to prevent.
    ///
    /// `final_score` includes `radiance × weight`, and radiance is constant across a settled store,
    /// so every candidate carries the same additive offset. Confidence must be blind to it —
    /// otherwise a well-separated query proposes ~zero steering and the whole bridge silently no-ops.
    #[test]
    fn a_constant_score_offset_cannot_change_confidence() {
        let base = vec![hit("a", "x", 6.0), hit("b", "y", 2.0), hit("c", "z", 1.0)];
        let offset: Vec<RecallHit> = base
            .iter()
            .map(|hit| {
                let mut shifted = hit.clone();
                // What a constant radiance term does to every candidate at once.
                shifted.scores.final_score += 13.2;
                shifted
            })
            .collect();

        let plain = separation(&base);
        let shifted = separation(&offset);
        assert!(
            (plain - shifted).abs() < 1e-5,
            "offset moved confidence: {plain} vs {shifted}"
        );
        assert!(plain > 0.7, "6 vs 2 over a 5-wide spread should be clear, got {plain}");
    }

    #[test]
    fn suggested_gain_never_leaves_the_measured_sweet_band() {
        assert!(suggested_gain(1.0, 10.0, 1.0) <= ALPHA_MAX);
        assert_eq!(suggested_gain(0.0, ALPHA_MAX, 1.0), 0.0);
    }

    /// The multi-memory invariant: loading more memories divides the push, never stacks it.
    ///
    /// Without this, three picks at full α would total 1.05 — past the 0.40 collapse onset — and a
    /// "does more memory help?" run would measure collapse instead of memory count.
    #[test]
    fn total_gain_is_budgeted_across_picks_not_paid_per_pick() {
        let hits = vec![
            hit("a", "one", 10.0),
            hit("b", "two", 6.0),
            hit("c", "three", 4.0),
            hit("d", "four", 0.0),
        ];
        let hot = hot_with(&hits, 0.0, 1.0);

        let mut totals = Vec::new();
        for limit in [1usize, 2, 3] {
            let config = PickConfig {
                limit,
                ..PickConfig::default()
            };
            let set = build("q", &hits, &hot, "e", test_null(), &config);
            assert_eq!(set.picks.len(), limit);
            assert!(
                set.total_suggested_gain <= ALPHA_MAX + 1e-5,
                "{limit} picks blew the budget: {}",
                set.total_suggested_gain
            );
            // Shares are a partition of the budget.
            let share_sum: f32 = set.picks.iter().map(|pick| pick.budget_share).sum();
            assert!((share_sum - 1.0).abs() < 1e-5, "shares sum to {share_sum}");
            totals.push(set.total_suggested_gain);
        }

        // Same total push regardless of how many memories carry it — that is what makes the
        // 1-vs-2-vs-3 comparison clean.
        assert!((totals[0] - totals[2]).abs() < 1e-5, "totals drifted: {totals:?}");

        // The top hit still gets the largest share; picks are not interchangeable.
        let set = build("q", &hits, &hot, "e", test_null(), &PickConfig::default());
        assert!(set.picks[0].suggested_gain > set.picks[1].suggested_gain);
        assert!(set.picks[1].suggested_gain > set.picks[2].suggested_gain);
    }

    /// Confidence must describe the query, not how many picks were asked for.
    #[test]
    fn confidence_does_not_drift_with_requested_limit() {
        let hits: Vec<RecallHit> = (0..12)
            .map(|i| hit(&format!("m{i}"), "text", 20.0 - i as f32 * 0.4))
            .collect();
        let hot = hot_with(&hits, 0.0, 1.0);

        let confidences: Vec<f32> = [1usize, 3, 6]
            .iter()
            .map(|limit| {
                build(
                    "q",
                    &hits,
                    &hot,
                    "e",
                    test_null(),
                    &PickConfig {
                        limit: *limit,
                        ..PickConfig::default()
                    },
                )
                .confidence
            })
            .collect();

        for window in confidences.windows(2) {
            assert!(
                (window[0] - window[1]).abs() < 1e-6,
                "confidence tracked limit: {confidences:?}"
            );
        }
        // Same for the separation diagnostic, which is the pool-depth-sensitive one.

        // And the pool really is capped — a deeper tail must not move it either.
        let deeper = separation(&hits);
        let truncated = separation(&hits[..CONFIDENCE_POOL]);
        assert!((deeper - truncated).abs() < 1e-6);
    }

    #[test]
    fn zero_scores_split_the_budget_evenly_rather_than_dividing_by_zero() {
        let hits = vec![hit("a", "one", 0.0), hit("b", "two", 0.0)];
        let hot = hot_with(&hits, 0.0, 1.0);
        let set = build("q", &hits, &hot, "e", test_null(), &PickConfig::default());
        assert!((set.picks[0].budget_share - 0.5).abs() < 1e-5);
        assert!((set.picks[1].budget_share - 0.5).abs() < 1e-5);
        assert!(set.total_suggested_gain.is_finite());
    }

    #[test]
    fn long_memories_are_clipped_at_a_sentence_and_flagged() {
        let text = format!("{} Final sentence here.{}", "padding. ".repeat(40), "x".repeat(400));
        let hits = vec![hit("a", &text, 9.0)];
        let hot = hot_with(&hits, 0.0, 1.0);
        let config = PickConfig {
            text_budget: 400,
            ..PickConfig::default()
        };
        let set = build("q", &hits, &hot, "e", test_null(), &config);

        let pick = &set.picks[0];
        assert!(pick.text_truncated);
        assert!(pick.text.chars().count() <= 400);
        // Clipped on a boundary, not mid-word.
        assert!(pick.text.ends_with('.'), "got tail {:?}", pick.text);
    }

    #[test]
    fn limit_and_min_score_both_apply() {
        let hits = vec![
            hit("a", "one", 9.0),
            hit("b", "two", 5.0),
            hit("c", "three", 0.5),
        ];
        let hot = hot_with(&hits, 0.0, 1.0);
        let config = PickConfig {
            limit: 5,
            min_score: 1.0,
            ..PickConfig::default()
        };
        let set = build("q", &hits, &hot, "e", test_null(), &config);
        assert_eq!(set.picks.len(), 2, "the 0.5 hit is below min_score");

        // Separation is measured over the *whole* pool, including the 0.5 hit that min_score
        // filtered out — a candidate excluded from steering is still evidence about the shape of
        // the field. (It is a diagnostic; `confidence` comes from the null model.)
        let expected = (9.0f32 - 5.0) / (9.0 - 0.5);
        assert!(
            (set.separation - expected).abs() < 1e-5,
            "got {} want {expected}",
            set.separation
        );
    }

    #[test]
    fn a_memory_with_no_splat_gets_neutral_knobs_not_zero_mass() {
        let hits = vec![hit("orphan", "never dreamed", 9.0)];
        let set = build("q", &hits, &HotState::default(), "e", test_null(), &PickConfig::default());
        let pick = &set.picks[0];
        // Zero mass would silently drop it out of the host's force law.
        assert_eq!(pick.mass, 1.0);
        assert_eq!(pick.gain, 0.0);
        assert!(pick.semantics_64.is_empty());
    }
}

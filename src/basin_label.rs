//! Historical note: a session experiment tried “meaning scores” to feed Gemma better samples.
//! That is **not** the SplatRAG product path. Labeling stays:
//!
//! `radiance-ranked representative_ids` → short Gemma prompt → cosmetic label/path/summary.
//!
//! Geometry clusters; cold text is truth; labels are name tags. See
//! `research_logs/2026-08-07_basins_labels_and_5k_bags.md`.
//!
//! This module only keeps small pure helpers used in unit tests / optional tooling — not
//! required for `label_basins`.

/// Detect near-duplicate Gemini Apps takeout footers (geometry should clump these; labels may say junk).
pub fn is_gemini_takeout_boilerplate(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("gemini apps activity")
        || (lower.contains("why is this here") && lower.contains("google account"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_footer_detected() {
        let t = "Products: Gemini Apps Why is this here? This activity was saved to your Google Account because the following settings were on: Gemini Apps Activity.";
        assert!(is_gemini_takeout_boilerplate(t));
    }

    #[test]
    fn normal_chat_not_boilerplate() {
        assert!(!is_gemini_takeout_boilerplate(
            "Jason I think friendship and honest answers are the same question."
        ));
    }
}

# Near-dupe mute

> Date: 2026-09-03
>
> **Agent:** Grok (xAI)

## Why the field still looks duplicated

| | n |
|---|---|
| cold | 128211 |
| unique `content_sha256` | 94374 |
| extra byte-identical | 33837 |
| hot muted | 33637 |
| painted (unmuted) | 94574 |
| unmuted unassigned | 11924 |
| same first 120 chars extra | 52529 |
| “strict dataset judge…” prefix | **13577** |
| “HARD OUTPUT RULES…” prefix | **3750** |

Dream already mutes byte-identical copies. The 13k judge rows differ in the shard payload, so they stay active and sit on top of each other.

Not the user+assistant two-row design. Those are different texts.

## Fix

Second collapse pass: first 160 whitespace-normalized chars, groups of ≥ 8, mute all but one. Cold untouched. `POST /api/recolor` runs it and persists.

## Next

New binary + recolor. Then recount painted.

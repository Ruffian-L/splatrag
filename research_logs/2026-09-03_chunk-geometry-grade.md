# Grade: concat vs assistant-only (post-hoc)

> Date: 2026-09-03
>
> **Agent:** Grok (xAI)
>
> Post-hoc parse of `2026-09-03_chunk-geometry-ab.json`. Not baked into the runner.

## Call

**Do not glue user+assistant into one embed.** Keep one turn, one splat.

SCORE **4** on current ingest (turns already split).  
SCORE **2** on Arm A (concat) as a change.  
SCORE **2** on Arm B (drop user rows, question is metadata only) for *this* store.  
The 9-pair probe itself: **3** — real embedder, toy n, topics I picked, not the 1,723 live basins.

## Why concat’s bigger gap is not a win

intra−inter: concat 0.289, assistant-only 0.243, user-only 0.257.

We labeled topics from the **user** question (towel / niodoo / desktop). User-only already separates. Concat looking “tighter” is the question leaking into the vector. That is the failure mode, not a quality score.

Smoking row **niodoo-1**: user “CoT without CoT December 2025” vs Gold Master README. user→assistant **0.409**. user→concat **0.632**. The splat moved toward the prompt. Evidence geometry got bent.

Towel rows are close both ways because the answer repeats the question (parallel drying). Those rows cannot decide the architecture.

## Why not assistant-only either

This store is Jason’s chats. “shift-click move window” and “store not working” are the retrieval keys. If those strings live only in sidecar and `record.text` is the assistant, BM25 on cold text misses the ask. Speaker filter (`human` / `collaborator`) also dies if there is one glued row.

## What the system already is

`ingest`: one non-empty turn → one `MemoryRecord`. `embed` is `record.text` only. Stitch is `conversation_id` + `turn_index`. Gate added on `streams_claude_array_and_preserves_two_messages`: two speakers, two texts, glued payload is not either text.

## Next

Do not concat the 128k store. Do not drop user rows.

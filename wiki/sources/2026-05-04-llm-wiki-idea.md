# LLM Wiki Idea File

- Date ingested: 2026-05-04
- Raw source: User-provided idea text in chat
- Type: notes

## Summary
The document proposes replacing stateless, per-query RAG behavior with a persistent, LLM-maintained markdown wiki that compounds in quality over time. Knowledge is integrated during ingest and then reused during query, while periodic linting keeps consistency high and uncovers gaps.

## Key Claims
- Traditional upload-and-retrieve workflows repeatedly rediscover the same facts and weakly accumulate synthesis.
- A persistent wiki layer allows cross-references, contradiction tracking, and evolving synthesis to be maintained once and improved incrementally.
- The maintenance burden that causes humans to abandon wikis can be offloaded to an LLM agent.
- A lightweight index (`index.md`) and chronological log (`log.md`) can support navigation and operations without immediate need for vector infrastructure at moderate scale.

## Entities
- [Obsidian](../entities/obsidian.md)
- [qmd](../entities/qmd.md)
- [Marp](../entities/marp.md)

## Concepts
- [Persistent Wiki over RAG](../concepts/persistent-wiki-over-rag.md)
- [Ingest / Query / Lint Loop](../concepts/ingest-query-lint-loop.md)
- [Index vs Log](../concepts/index-vs-log.md)

## Contradictions / Uncertainty
- The design is intentionally abstract; directory and workflow details should evolve with usage.
- Scalability thresholds (e.g., when index-only retrieval stops being enough) are context dependent.

## Actionable Follow-ups
- Start ingesting one real source file in `raw/` to calibrate summary depth and page-touch policy.
- Define preferred granularity for entity/concept creation.
- Schedule first lint pass after 5–10 ingests.

## Sources
- User idea text (chat, 2026-05-04)

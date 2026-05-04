# LLM Wiki Thesis

## Core Thesis
A persistent, LLM-maintained wiki should sit between raw sources and final answers, so synthesis accumulates over time instead of being recomputed from scratch per query.

## Architecture
- Raw sources (`raw/`) are immutable.
- Wiki pages (`wiki/`) are editable compiled knowledge.
- Schema (`CLAUDE.md`) governs ingest/query/lint behavior.

## Why This Outperforms Stateless RAG
- Precomputed synthesis across sources reduces repeated reasoning cost.
- Existing cross-links improve recall of non-obvious relationships.
- Contradictions are tracked as first-class maintenance objects.

## Operating Loop
- Ingest: integrate a source across multiple pages.
- Query: answer from wiki and persist durable outputs.
- Lint: detect drift, gaps, contradictions, and orphan structure.

## Constraints and Risks
- Wiki quality depends on disciplined updates to index/log.
- Poor citation hygiene can create unverifiable claims.
- Over-fragmentation can reduce navigability.

## Sources
- [LLM Wiki Idea File](../sources/2026-05-04-llm-wiki-idea.md)

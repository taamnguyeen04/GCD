# Wiki Log

## [2026-05-04] bootstrap | LLM Wiki initialization
- Operation: bootstrap
- Inputs: user-provided LLM Wiki idea specification
- Files created:
  - `CLAUDE.md`
  - `wiki/index.md`
  - `wiki/log.md`
  - `wiki/sources/2026-05-04-llm-wiki-idea.md`
  - `wiki/synthesis/llm-wiki-thesis.md`
  - `wiki/entities/obsidian.md`
  - `wiki/entities/qmd.md`
  - `wiki/entities/marp.md`
  - `wiki/concepts/persistent-wiki-over-rag.md`
  - `wiki/concepts/ingest-query-lint-loop.md`
  - `wiki/concepts/index-vs-log.md`
- Notes:
  - Established full ingest/query/lint schema.
  - Seeded first ingest from the idea file.
  - Ready for next source ingest from `raw/`.

## [2026-05-04] ingest | Consistent Supervised-Unsupervised Alignment for GCD
- Operation: ingest
- Input: `raw/Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery.md`
- Files created:
  - `wiki/sources/2026-05-04-consistent-supervised-unsupervised-alignment-gcd.md`
  - `wiki/entities/nc-gcd.md`
  - `wiki/entities/semantic-consistency-matcher.md`
  - `wiki/entities/neural-collapse.md`
  - `wiki/concepts/generalized-category-discovery.md`
  - `wiki/concepts/simplex-etf-prototypes.md`
  - `wiki/concepts/consistent-supervised-unsupervised-alignment.md`
- Files updated:
  - `wiki/index.md`
  - `wiki/log.md`
- Notes:
  - Captured NC-GCD core design: fixed ETF prototypes + consistent hybrid alignment + SCM.
  - Recorded reported gains, especially for novel-category accuracy.
  - Suggested follow-up comparative ingestion of SimGCD/CMS/SelEx.

## [2026-05-04] ingest | Hyperbolic Category Discovery
- Operation: ingest
- Input: `paper/raw/Hyperbolic Category Discovery.md`
- Files created:
  - `wiki/sources/2026-05-04-hyperbolic-category-discovery.md`
  - `wiki/entities/hypcd.md`
  - `wiki/entities/poincare-ball-model.md`
  - `wiki/entities/selex.md`
  - `wiki/concepts/hyperbolic-geometry-for-gcd.md`
  - `wiki/concepts/hierarchy-aware-representation-learning.md`
  - `wiki/concepts/hybrid-hyperbolic-contrastive-loss.md`
- Files updated:
  - `wiki/index.md`
  - `wiki/log.md`
- Notes:
  - Captured HypCD core: Euclidean→hyperbolic mapping, hybrid contrastive objective, and hyperbolic classifier.
  - Added links to existing GCD/NC-GCD knowledge for cross-method comparison.
  - Marked parameter sensitivity (c, r, alpha_d) as a key uncertainty for generalization.

## [2026-05-04] ingest | Parametric Classification for GCD: A Baseline Study
- Operation: ingest
- Input: `paper/raw/Parametric Classification for Generalized Category Discovery A Baseline Study.md`
- Files created:
  - `wiki/sources/2026-05-04-parametric-classification-gcd-baseline-study.md`
  - `wiki/entities/simgcd.md`
  - `wiki/entities/gcd-cvpr-2022.md`
  - `wiki/entities/entropy-regularization-gcd.md`
  - `wiki/concepts/parametric-classification-for-gcd.md`
  - `wiki/concepts/prediction-bias-in-gcd.md`
  - `wiki/concepts/self-distillation-pseudo-labeling.md`
- Files updated:
  - `wiki/index.md`
  - `wiki/log.md`
- Notes:
  - Captured SimGCD key claim: parametric failure is mainly pseudo-label quality and prediction bias, not parametric form itself.
  - Added bias-focused concepts to connect with future method-comparison queries.
  - Highlighted robustness behavior when category number K is unknown.

# Adversarial Hubness in Multi-Modal Retrieval — Reproduction & Extension

Reproduces the core attack technique (*adversarial hub*) from [Adversarial Hubness in Multi-Modal Retrieval (Zhang et al.)](https://arxiv.org/abs/2412.14113), and compares a self-implemented detector against the official implementation of the follow-up paper, [Adversarial Hubness Detector (Habler et al.)](https://arxiv.org/abs/2602.22427).

The original paper explains the strong attack transfer between ImageBind and OpenCLIP-ViT by noting that ImageBind "extends OpenCLIP-ViT's architecture with additional modalities while preserving the core visual encoder." Building on this observation, this project formulates and tests an extended hypothesis — **models with more similar image-encoding distributions should show more similar attack success rates (ASR) when a hub generated on one is transferred to another** — and evaluates it quantitatively across 8 models.

## Key Findings

### 1. Hub Attack Reproduction

The phenomenon reported in the original paper was **successfully reproduced**: an image (hub) optimized on a small set of queries is retrieved with high probability across a much larger set of unrelated queries. The held-out ASR@1 of a universal hub reached roughly 70%. This held consistently across three surrogate models — CLIP, LAION-CLIP, and ImageBind — confirming the phenomenon is robust.

**Supporting Files**

| Evidence | File |
| --- | --- |
| Baseline Recall@1/5/10 | `results/openai_clip/recall_baseline.json` |
| Universal hub held-out ASR@1 (CLIP, OpenAI) | `results/openai_clip/asr_summary.json` |
| Universal hub held-out ASR@1 (CLIP, LAION reproduction) | `results/laion_clip/asr_summary.json` |
| Universal hub held-out ASR@1 (ImageBind reproduction) | `results/imagebind/asr_summary.json` |

### 2. Word vs Cluster Attack Strength — Discrepancy with the Original Paper

Unlike the [original paper](https://arxiv.org/abs/2412.14113)'s result (Cluster R@1 = 100% ≫ Word R@1 = 28.6%), this reproduction consistently found word-based attacks to be stronger than cluster-based ones (CLIP: "dog" 39.7% vs. cluster 8.9%). Three candidate explanations for this discrepancy — **evaluation scale, choice of surrogate model, and the Qt/cluster size cap** — were each tested and **all were rejected**. The true cause likely lies outside these three factors, but time and resource constraints prevented pinning it down.

As a related line of investigation, the source of per-concept (per-word) variance in attack strength was analyzed separately. Correlating Qt caption-pool size and pairwise caption cohesion (semantic similarity) with ASR showed both variables to be significant under Pearson's test (pool size r=0.79, p=0.006; cohesion r=−0.74, p=0.015) but not under Spearman's (p=0.128 and p=0.260, respectively), and the two variables were themselves strongly confounded (r=−0.66, p=0.037) — so this dataset alone could not isolate which one actually drives the effect.

**Supporting Files**

| Evidence | File |
| --- | --- |
| Word/Cluster ASR by concept (CLIP, OpenAI) | `results/openai_clip/asr_summary.json` |
| Word/Cluster ASR by concept (ImageBind, model-difference check) | `results/imagebind/asr_summary.json` |
| Ranking comparison across evaluation scale (500 vs. 25,000 queries) | `results/laion_clip/scale_effect_analysis.json` |
| Qt pool-size correlation | `results/openai_clip/word_variance/pool_size_analysis.json` |
| Caption-cohesion correlation | `results/openai_clip/word_variance/cohesion_analysis.json` |

### 3. Hub Detection — Query Sampling Strategy as the Dominant Factor

The self-built detector achieved a ROC-AUC of **0.994** on 480 hubs. The official implementation's performance, in contrast, varied dramatically depending on the source of its evaluation queries:

- `mixed` (queries self-sampled by the detector from gallery image embeddings): ROC-AUC **0.36–0.57** (effectively random)
- `real_queries` (queries drawn from actual captions): ROC-AUC **0.94**

Three candidate explanations for the `mixed` mode's poor performance — hub-to-corpus ratio, mixing of hub types, and surrogate-model choice (CLIP vs. ImageBind) — were each tested and rejected. The most plausible remaining explanation is a **modality mismatch**: hubs are optimized to target a text-caption centroid, while `mixed`'s evaluation queries are image-embedding based. This explanation was not, however, quantitatively isolated within the scope of this project.

Separately, `real_queries` detected only 28 of the 480 hubs (5.8%) within its top alert budget (12 HIGH + 16 MEDIUM, 100% precision) — matching exactly the limitation the follow-up paper (Habler et al.) itself notes: at small alert budgets, budget saturation can push domain-specific hubs out of the top-K. Independently, repeated tests of `mixed` at hub-to-corpus ratios of 8.8% → 2.0% → 0.2% found **no monotonic improvement as the ratio decreased** — the mean ROC-AUC peaked at 2.0% (0.566) and was actually lowest at 0.2% (0.414), the ratio matching the original paper (means over 8 repetitions each).

**Why the self-built detector catches word/cluster hubs just as well as universal ones, while the official detector's own paper reports a specific weakness there.** Breaking down the self-built detector's per-hub results by type shows near-uniform detection across every category (100% for universal and cluster, 96.7–100% for each of the 10 word-based concepts) — so on its own, this looks inconsistent with the follow-up paper's documented budget-saturation limitation. Tracing this to the official `real_queries` scan's raw output resolves the apparent contradiction: it doesn't score every gallery+hub item independently, but first narrows 5,480 items down to a **fixed top-100 candidate list by risk score**, and only issues an actionable HIGH/MEDIUM verdict for a subset of those. Breaking that top-100 list down by hub type shows universal hubs dominating the actionable verdicts (13 of 30 universal hubs rated HIGH/MEDIUM) while several word-based concepts get none at all (e.g., 0 of 30 for "umbrella") and most individual word/cluster hubs never even enter the top-100 list. The underlying anomaly signal (`hub_z`) for a word/cluster hub is still far above a clean gallery item's — which is why an unconstrained threshold, as used by the self-built detector, flags it easily — but it is systematically smaller than a universal hub's signal, since a word hub only dominates the narrow slice of queries containing that concept. Once candidates have to compete for a small, realistic alert budget, the larger-signal universal hubs crowd out the smaller-signal word/cluster hubs — reproducing, at the level of individual hubs, exactly the budget-saturation limitation the follow-up paper reports in its own ablation.

**Supporting Files**

| Evidence | File |
| --- | --- |
| Self-built detector performance | `results/openai_clip/detector_summary.json` |
| Self-built detector, per-hub-type detection breakdown | `results/openai_clip/detector_summary.json` (`hub_details`, grouped by tag prefix) |
| `mixed` / `real_queries` baseline (8.8% hub ratio) | `results/openai_clip/official_detector/mixed/report.json`, `results/openai_clip/official_detector/real_queries/report.json` (hub counts taken directly from the `verdict_counts` and `suspicious_documents` arrays; not stored as a separate recall field) |
| `real_queries` top-100 candidate list, per-hub-type verdict breakdown | `results/openai_clip/official_detector/real_queries/report.json` (`suspicious_documents`, grouped by `metadata.doc_id` prefix) |
| Single-run 2.0% / 0.2% hub ratio | `results/openai_clip/official_detector/mixed_ratio_2pct/full_eval_summary.json`, `results/openai_clip/official_detector/mixed_ratio_0.2pct/full_eval_summary.json` |
| 2.0% / 0.2% hub ratio, 8-run mean & std | `results/openai_clip/detector_ratio_repeat_summary.json` |
| Universal-hub-only re-check | `results/openai_clip/official_detector/mixed_universal_only/full_eval_summary.json` |
| Surrogate-model difference (ImageBind) re-check | `results/imagebind/official_detector/mixed/full_eval_summary.json` |

### 4. Architectural Similarity and Transfer Attack Performance

RSA (Representational Similarity Analysis) and Alignment (query alignment) were measured across 8 models (ViT-B/16, ViT-B/32, ViT-L/14, ViT-H/14, RN50, ImageBind, etc.).

- **RSA–Alignment correlation was strongly significant** (Pearson r=0.82, p<0.0001), supporting the validity of both metrics as measurement tools. In fact, OpenCLIP ViT-H/14 and ImageBind were found to produce essentially identical embeddings (**RSA = 1.0000**), further corroborating this validity.
- **RSA–ASR correlation was not significant** (Pearson r=0.43, p=0.33). In other words, the extended hypothesis — "more similar architectures transfer attacks better" — was not statistically supported.
- The moment a hub left the surrogate model (self-space ASR of 99.8% vs. 12–16% on every other target), attack strength dropped to a similarly low level across all targets, with no pattern of "more similar encoding → smaller drop." This suggests that **transfer-attack performance stays broadly low unless the surrogate shares the exact same underlying encoder as the target**.
- The type strongest in its own space (Universal, self-space ASR@1 up to 99.8%) paradoxically suffered the largest collapse when transferred to other models. This pattern reproduced consistently across four independent experiments with different surrogate/target combinations, making it one of the most reliable original findings of this project.

**Supporting Files**

| Evidence | File |
| --- | --- |
| Full RSA / Alignment / Transfer ASR results across 8 models | `results/model_comparison/summary.json` |
| Sample composition used for comparison (500-item gallery/query/hub subsample) | `results/model_comparison/manifest.json` |
| Universal hub self-space ASR (25,000 queries, per surrogate) | `results/openai_clip/asr_summary.json`, `results/laion_clip/asr_summary.json` |

## Environment

| Conda Environment | Purpose |
| --- | --- |
| `hubness` (Python 3.10, torch cu121) | Full CLIP-based pipeline, self-built detector, model comparison |
| `ahd` (Python 3.11, CPU) | Running the official Adversarial Hubness Detector |
| `imagebind` (Python 3.10, torch cu121) | ImageBind-based hub generation and embedding extraction |

Hardware: RTX 2070 SUPER (8 GB VRAM), WSL2 + Ubuntu. The methodological reductions relative to the original paper (see "Methodological Differences" below) stem from this hardware constraint.

## Dataset

- Gallery: 5,000 images from MS-COCO val2017
- Queries: 5 captions per image, 25,000 total
- Indexed with FAISS `IndexFlatIP` (exhaustive search); baseline Recall@1/5/10 = 28.7% / 53.1% / 64.7%

## External Tools

Kept under `external/`, excluded from git tracking — clone/install separately to reproduce.

- [`adv-hubness-detector`](https://github.com/cisco-ai-defense/adversarial-hubness-detector): official implementation of the Adversarial Hubness Detector (Habler et al.)
- [`ImageBind`](https://github.com/facebookresearch/ImageBind): Meta's official ImageBind implementation
- [`adv_hub`](https://github.com/Tingwei-Zhang/adv_hub): original authors' (Zhang et al.) public code (used to check implementation details)

## Methodological Differences from the Original Paper

Resource constraints led to the following reductions relative to the original paper:

| Item | Original Paper | This Project |
| --- | --- | --- |
| PGD iterations | T = 1,000 | T = 600 (T = 300 for ImageBind) |
| Qt size cap | Unbounded (all captions containing the target word) | Capped at 100 |
| Number of clusters | 25,000 queries split into 1,000 clusters | Same 1,000-way split, then 5 clusters of size 15–80 randomly selected |
| Repetitions (hubs per condition) | 100 per condition, mean ± std reported | 30 per condition (CLIP), 10 per condition (ImageBind) |
| Word selection | 100 concepts extracted via ChatGPT | 10 words chosen manually |

Additional structural differences were also identified:

- The original paper does **not specify its cluster-selection criteria** for concept-specific (word/cluster) hubs, and the public code (`adv_hub`) contains no logic for this step either — meaning exact reproduction is fundamentally limited.
- The ~76% transfer performance reported in the original paper was obtained **only with a 3-model ensemble surrogate**; the paper explicitly states that single-model surrogate results were omitted because "ensembles work better" (§7.1: "we omit the results using individual models as surrogates because ensembles work better"). This project's single-surrogate experiments (Key Finding 4) are an independent check under different conditions, not a direct reproduction of the ensemble surrogate itself.

## Script Structure

```
scripts/
├── 01_baseline/                 Build gallery/queries, measure baseline performance
├── 02_hub_generation/           Generate adversarial hubs, evaluate attack success rate (ASR)
├── 03_detector/                 Hub detection (self-built + official implementation, various ablations)
├── 04_model_comparison/         Compare representational similarity (RSA), alignment, and transfer ASR across 8 models
└── 05_word_variance_analysis/   Analyze the source of per-concept (per-word) variance in attack strength
```

| Folder | Script | Role |
| --- | --- | --- |
| `01_baseline` | `01_cache_images.py` | Cache 5,000 images + 5 captions each locally |
| | `02_build_index.py` / `02_build_index_laion.py` | Build CLIP (OpenAI/LAION) embeddings + FAISS index |
| | `03_query_test.py` | Qualitative sanity check of the retrieval pipeline |
| | `04_recall_eval.py` | Measure baseline Recall@1/5/10 |
| `02_hub_generation` | `01_generate_hub.py` / `01_generate_hub_laion.py` | Generate a single Universal/Word-based/Cluster-based hub (PGD) |
| | `02_generate_hub_imagebind.py` | Generate a hub via ImageBind gradients (no cluster-size cap) |
| | `03_batch_generate.py` / `04_batch_generate_laion.py` | Batch-generate 30 hubs per condition (480 total) |
| | `05~07_evaluate_hub_asr*.py` | Evaluate in-sample / held-out ASR per hub |
| `03_detector` | `01~02_hubness_detector*.py` | Self-built detector (median/MAD z-score, cluster spread, stability) |
| | `03~07_prepare_for_official*.py` | Prepare inputs for the official detector (full/reduced ratio/universal-only/ImageBind) |
| | `08~12_run_official_scan*.py` | Run the official detector (`mixed`/`real_queries` modes) |
| | `13~14_repeat_ratio_test*.py` | Repeated verification across different hub-to-corpus ratios |
| `04_model_comparison` | `01_build_gallery_imagebind.py` | Re-encode the full gallery/query set with ImageBind |
| | `02~03_extract_embeddings*.py` | Re-encode gallery/query/hub with 8 models (6 CLIP variants + RN50 + ImageBind) |
| | `04_compare_models.py` | Compute RSA / Alignment / Transfer ASR |
| `05_word_variance_analysis` | `01_pool_size_correlation.py` | Correlation between Qt caption-pool size and ASR |
| | `02_cohesion_correlation.py` | Correlation between caption semantic cohesion and ASR |
| | `03_scale_effect_analysis.py` | Effect of evaluation scale (500 vs. 25,000 queries) on Word/Cluster ranking |

## Results Structure

```
results/
├── openai_clip/          CLIP ViT-B/32 (OpenAI) surrogate — 480 hubs, main experiment
├── laion_clip/            OpenCLIP ViT-B/32 (LAION-2B) surrogate — 480 hubs, surrogate-check
├── imagebind/              ImageBind surrogate — 150 hubs, model-difference check
└── model_comparison/       RSA / Alignment / ASR comparison across 8 models
```

Inside each surrogate folder:

```
hubs/                       Hub images, embeddings, metadata
asr_summary.json            Held-out ASR@k
detector_summary.json       Self-built detector results (openai_clip, laion_clip only)
official_detector/
  mixed/                     Official detector, self-sampled queries
  real_queries/              Official detector, real caption-based queries
  mixed_ratio_*/             Re-checks with hub-to-corpus ratio reduced to 2.0%/0.2% (openai_clip only)
  mixed_universal_only/      Re-check isolating universal hubs only (openai_clip only)
```

## Limitations

Given resource and time constraints, this project has the following limitations:

- **Source of per-concept attack-strength variance not identified**: correlations between Qt caption-pool size, semantic cohesion, and ASR were analyzed (see Key Finding 2), but the two variables are strongly confounded and the sample (10 words) is too small to statistically isolate their independent effects. Image-level saliency remains a candidate cause but was out of scope, requiring a separate object-detection-based labeling tool.
- **Exact cause of the Word vs. Cluster discrepancy not identified**: three candidates (evaluation scale, surrogate model, Qt/cluster size cap) were all rejected, but no alternative explanation was established.
- **Quantitative cause of `mixed` detector failure not identified**: a qualitative explanation (modality mismatch) was reached, but hub-optimization strength (PGD steps, number of target queries) was not scaled up to the original paper's level (1,000 steps, 200 queries) to verify it directly.
- **Ensemble surrogate not tested**: the fact that the original paper's ~76% transfer performance came only from a 3-model ensemble was confirmed by reading the paper itself, not by direct reproduction in this project.
- **Limited scope of model comparison**: AudioCLIP was excluded due to installation and environment-setup risk.
- **Limited concept-word sample**: the correlation analysis over 10 words has low statistical power; a multiple regression over 30–50 words is needed as follow-up.

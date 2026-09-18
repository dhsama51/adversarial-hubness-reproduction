# Adversarial Hubness in Multi-Modal Retrieval — 재현 및 확장 실험

[Adversarial Hubness in Multi-Modal Retrieval (Zhang et al.)](https://arxiv.org/abs/2412.14113)의 핵심 공격 기법(adversarial hub)을 재현하고, 후속 논문 [Adversarial Hubness Detector (Habler et al.)](https://arxiv.org/abs/2602.22427)의 탐지 기법을 자체 구현 및 공식 구현체와 비교한 프로젝트입니다.

원 논문은 ImageBind가 OpenCLIP-ViT의 구조를 확장하면서도 핵심 비전 인코더(core visual encoder)는 그대로 보존한다는 사실("extends OpenCLIP-ViT's architecture with additional modalities while preserving the core visual encoder")을 근거로, 두 모델 사이의 강한 공격 전이(transfer)를 설명한 바 있습니다. 본 프로젝트는 이 서술로부터 **"인코딩된 이미지의 분포가 유사한 모델일수록, 한 모델에서 만든 hub가 다른 모델에서도 유사한 공격 성공률(ASR)을 보일 것"**이라는 가설을 도출하고, 8개 모델을 대상으로 이를 정량적으로 검증하는 확장 실험을 수행하였습니다.

## Key Findings (핵심 결과 요약)

### 1. Hub Attack Reproduction (Hub 공격 재현)

원 논문이 보고한 현상을 **성공적으로 재현**하였습니다. 소수의 쿼리로 최적화한 이미지(hub)가 이와 무관한 대규모 쿼리 집합에서도 높은 확률로 검색됨을 확인하였으며, Universal hub의 held-out ASR@1은 약 70%대로 나타났습니다. 이 현상은 CLIP·LAION·ImageBind 세 가지 surrogate 모델 전반에서 일관되게 재현되어, 견고한 현상임을 확인하였습니다.

**Supporting Files**

| 근거 | 파일 |
|---|---|
| Baseline Recall@1/5/10 | `results/openai_clip/recall_baseline.json` |
| Universal hub held-out ASR@1 (CLIP, OpenAI) | `results/openai_clip/asr_summary.json` |
| Universal hub held-out ASR@1 (CLIP, LAION 재현) | `results/laion_clip/asr_summary.json` |
| Universal hub held-out ASR@1 (ImageBind 재현) | `results/imagebind/asr_summary.json` |

### 2. Word vs Cluster Attack Strength — Discrepancy with the Original Paper (원 논문과의 불일치)

[원 논문](https://arxiv.org/abs/2412.14113)이 보고한 결과(Cluster R@1=100% ≫ Word R@1=28.6%)와 달리, 본 실험에서는 Word 기반 공격이 Cluster 기반 공격보다 일관되게 강한 것으로 나타났습니다(CLIP 기준 dog 39.7% vs cluster 8.9%). 이러한 **불일치의 원인 후보로 평가 규모, surrogate 모델 종류, Qt/클러스터 크기 제한 세 가지를 설정하여 각각 검증하였으나 모두 기각**되었습니다. 위 세 요인 외의 다른 원인에 기인한 것으로 추정되나, 시간 및 자원의 제약으로 명확한 원인은 규명하지 못하였습니다.

이 불일치의 배경이 될 수 있는 추가 요인으로, 개념(단어)별 공격 강도 편차의 원인을 별도로 분석하였습니다. Qt 캡션 풀 크기 및 캡션 간 의미적 응집도(pairwise cosine similarity)를 ASR과 상관분석한 결과, 두 변수 모두 Pearson 기준으로는 유의미하였으나(풀 크기 r=0.79, p=0.006; 응집도 r=-0.74, p=0.015) Spearman 기준으로는 유의미하지 않았고(각각 p=0.128, p=0.260), 두 변수 자체가 서로 강하게 얽혀 있어(r=-0.66, p=0.037) 어느 것이 실제 원인인지 이 데이터만으로는 분리하지 못하였습니다.

**Supporting Files**

| 근거 | 파일 |
|---|---|
| Word/Cluster별 ASR (CLIP, OpenAI) | `results/openai_clip/asr_summary.json` |
| Word/Cluster별 ASR (ImageBind, 모델 차이 검증용) | `results/imagebind/asr_summary.json` |
| 평가 규모(500 vs 25,000쿼리) 순위 비교 | `results/laion_clip/scale_effect_analysis.json` |
| Qt 풀 크기 상관분석 | `results/openai_clip/word_variance/pool_size_analysis.json` |
| 캡션 응집도 상관분석 | `results/openai_clip/word_variance/cohesion_analysis.json` |

### 3. Hub Detection — Query Sampling Strategy as the Dominant Factor (쿼리 샘플링 방식의 지배적 영향)

자체 구현 detector의 탐지 성능은 480개 hub 기준 ROC-AUC(Receiver Operating Characteristic – Area Under Curve) **0.994**로 나타났습니다. 공식 구현체는 쿼리 소스의 종류에 따라 성능이 극단적으로 갈리는 양상을 보였습니다.

- `mixed`(detector가 gallery 이미지 임베딩을 기반으로 자체 샘플링한 쿼리 사용): ROC-AUC **0.36~0.57** (사실상 무작위 수준)
- `real_queries`(실제 caption 기반 쿼리 사용): ROC-AUC **0.94**

`mixed` 방식의 성능 저하 원인으로 corpus 대비 hub 비율, hub 유형 혼합, surrogate 모델 차이(CLIP/ImageBind)를 각각 검증하였으나 모두 기각되었습니다. 가장 유력한 설명은 **hub가 텍스트 캡션의 centroid를 겨냥하도록 최적화된 반면 `mixed`의 검사 쿼리는 이미지 임베딩 기반이라는 모달리티 불일치**이나, 시간 및 자원의 제약으로 이를 정량적으로 격리하여 검증하지는 못하였습니다.

한편 `real_queries` 방식은 전체 480개 hub 중 top alert budget 내에서 28개(5.8%)만을 탐지하였는데(HIGH 12개 + MEDIUM 16개, 정밀도는 100%), 이는 후속 논문(Habler et al.)이 스스로 명시한 한계("작은 alert budget에서는 domain-specific hub가 다른 고득점 항목에 밀려 top-K 밖으로 나가는 budget saturation이 발생한다")와 정확히 일치하는 결과입니다. 다만 이와는 별개로, `mixed` 방식에 대해 corpus 대비 hub 비율을 8.8%→2.0%→0.2%로 낮춰가며 반복 검증한 결과, **비율이 낮을수록 성능이 계속 개선되는 단조적 관계는 관찰되지 않았습니다**(2.0%에서 평균 ROC-AUC 0.566으로 최고치를 보인 반면, 논문과 동일한 0.2%에서는 오히려 0.414로 가장 낮았습니다. 각 8회 반복 기준).

**Supporting Files**

| 근거 | 파일 |
|---|---|
| 자체 구현 detector 성능 | `results/openai_clip/detector_summary.json` |
| `mixed` / `real_queries` 기본 결과 (corpus 비율 8.8%) | `results/openai_clip/official_detector/mixed/report.json`, `results/openai_clip/official_detector/real_queries/report.json` (verdict_counts 및 suspicious_documents 배열에서 hub 개수를 직접 확인한 값이며, 별도의 recall 필드로 저장되어 있지는 않음) |
| Hub 비율 2.0%/0.2% 단일 측정 | `results/openai_clip/official_detector/mixed_ratio_2pct/full_eval_summary.json`, `results/openai_clip/official_detector/mixed_ratio_0.2pct/full_eval_summary.json` |
| Hub 비율 2.0%/0.2% 8회 반복 (평균·표준편차) | `results/openai_clip/detector_ratio_repeat_summary.json` |
| Universal hub만 분리 재검증 | `results/openai_clip/official_detector/mixed_universal_only/full_eval_summary.json` |
| surrogate 모델 차이(ImageBind) 재검증 | `results/imagebind/official_detector/mixed/full_eval_summary.json` |

### 4. Architectural Similarity and Transfer Attack Performance (아키텍처 유사성과 전이 공격의 관계)

RSA(Representational Similarity Analysis, 표현 구조 유사도)와 Alignment(쿼리 정렬도)를 8개 모델(ViT-B/16·B/32·L/14·H/14, RN50, ImageBind 등)을 대상으로 측정하였습니다.

- **RSA-Alignment 상관관계는 강하게 유의미**하였습니다(Pearson r=0.82, p<0.0001). 이는 두 지표가 측정 도구로서 타당함을 뒷받침합니다. 실제로 OpenCLIP ViT-H/14와 ImageBind의 임베딩이 사실상 완전히 동일함을 확인하였으며(**RSA=1.0000**), 이는 두 지표의 타당성을 추가로 뒷받침하는 근거가 되었습니다.
- **RSA-ASR 상관관계는 유의미하지 않았습니다**(Pearson r=0.43, p=0.33). 즉 "아키텍처가 유사하면 공격 전이가 잘 이루어진다"는 확장 가설은 통계적으로 지지되지 않았습니다.
- surrogate 모델을 벗어나는 즉시(자기 공간 ASR 99.8% → 타 모델 12~16%) 모든 대상 모델에서 공격력이 유사한 수준으로 하락하였으며, "인코딩 결과가 유사할수록 하락 폭이 작다"는 패턴은 관찰되지 않았습니다. 이는 **원본 encoder를 완전히 계승한 모델을 surrogate로 구성하지 않는 한, 전이 공격(transfer attack)의 성능이 전반적으로 낮게 유지**될 것임을 시사합니다.
- 자기 공간에서 가장 강력한 유형(Universal, self-space ASR@1 최대 99.8%)이 다른 모델로 전이 시 오히려 가장 크게 무너지는 역설적 패턴이 관찰되었습니다. 이는 surrogate와 target 조합을 달리한 네 차례의 독립적 실험 모두에서 일관되게 재현되어, 본 프로젝트에서 확보한 가장 신뢰도 높은 독자적 발견 중 하나입니다.

**Supporting Files**

| 근거 | 파일 |
|---|---|
| 8개 모델 RSA / Alignment / Transfer ASR 전체 결과 | `results/model_comparison/summary.json` |
| 비교에 사용한 샘플 구성(gallery/query/hub 500개 서브샘플) | `results/model_comparison/manifest.json` |
| Universal hub 자기 공간 ASR (25,000쿼리 기준, surrogate별) | `results/openai_clip/asr_summary.json`, `results/laion_clip/asr_summary.json` |

## Environment (환경)

| conda 환경 | 용도 |
|---|---|
| `hubness` (Python 3.10, torch cu121) | CLIP 기반 파이프라인 전체, 자체 detector, 모델 비교 |
| `ahd` (Python 3.11, CPU) | 공식 Adversarial Hubness Detector 실행 |
| `imagebind` (Python 3.10, torch cu121) | ImageBind 기반 hub 생성 및 임베딩 추출 |

하드웨어: RTX 2070 SUPER (8GB VRAM), WSL2 + Ubuntu. 본 프로젝트 전반에 걸친 방법론적 축소(아래 "Methodological Differences" 참고)는 이러한 자원 제약에 기인합니다.

## Dataset (데이터)

- 갤러리: MS-COCO val2017 이미지 5,000장
- 쿼리: 이미지당 caption 5개, 총 25,000개
- FAISS `IndexFlatIP`(완전탐색)로 인덱스 구축, Baseline Recall@1/5/10 = 28.7% / 53.1% / 64.7%

## External Tools (외부 도구)

`external/`, git 추적 제외 — 재현 시 각자 clone/설치 필요.

- [`adv-hubness-detector`](https://github.com/cisco-ai-defense/adversarial-hubness-detector): Adversarial Hubness Detector (Habler et al.)의 공식 구현체
- [`ImageBind`](https://github.com/facebookresearch/ImageBind): Meta의 ImageBind 공식 구현체
- [`adv_hub`](https://github.com/Tingwei-Zhang/adv_hub): 원 논문(Zhang et al.) 저자 공개 코드 (구조 확인용)

## Methodological Differences from the Original Paper (방법론적 노트)

자원 제약으로 인해 원 논문 대비 다음과 같은 축소가 있었습니다.

| 항목 | 원 논문 | 본 프로젝트 |
|---|---|---|
| PGD 반복 횟수 | T=1,000 | T=600 (ImageBind는 T=300) |
| Qt 크기 상한 | 제한 없음(해당 단어를 포함하는 caption 전체 사용) | 100개로 제한 |
| Cluster 개수 | 25,000개 쿼리를 1,000개로 분할 | 동일하게 1,000개로 분할한 뒤, 크기 15~80인 클러스터 중 5개를 임의로 선정 |
| 반복 횟수(hub 개수) | 조건당 100회, 평균±표준편차 보고 | CLIP 조건당 30회, ImageBind 조건당 10회 |
| Word 선정 방식 | ChatGPT로 추출한 100개 개념 | 임의로 선정한 10개 단어 |

이 외에도 다음과 같은 구조적 차이를 확인하였습니다.

- 원 논문은 concept-specific(word/cluster) hub의 **클러스터 선별 기준을 본문에 명시하지 않았으며**, 공개 코드(`adv_hub`)에도 해당 로직이 포함되어 있지 않아 정확한 재현에 근본적 한계가 있음을 확인하였습니다.
- 원 논문이 보고한 76% 수준의 transfer 성능은 **3개 모델을 앙상블한 surrogate에서만 보고된 수치**이며, 논문은 단일 모델 surrogate의 경우 성능이 낮아 결과를 생략하였다고 명시하였습니다(§7.1: "we omit the results using individual models as surrogates because ensembles work better"). 본 프로젝트의 단일 surrogate 실험(Key Findings 4)은 이와 다른 조건에서의 독자적 검증이며, 앙상블 surrogate 자체를 직접 구현·재현한 것은 아닙니다.

## Script Structure (스크립트 구성)

```
scripts/
├── 01_baseline/                 갤러리·쿼리 구축, baseline 성능 측정
├── 02_hub_generation/           adversarial hub 생성 및 공격 성공률(ASR) 평가
├── 03_detector/                 hub 탐지 (자체 구현 + 공식 구현체, 각종 변형 실험)
├── 04_model_comparison/         8개 모델 간 표현 구조 유사도(RSA)·정렬도(Alignment)·전이 ASR 비교
└── 05_word_variance_analysis/   개념(단어)별 공격 강도 편차의 원인 분석
```

| 폴더 | 스크립트 | 역할 |
|---|---|---|
| `01_baseline` | `01_cache_images.py` | 이미지 5,000장 + caption 5개씩 로컬 캐싱 |
| | `02_build_index.py` / `02_build_index_laion.py` | CLIP(OpenAI/LAION) 임베딩 + FAISS 인덱스 구축 |
| | `03_query_test.py` | 검색 파이프라인 정성 테스트 |
| | `04_recall_eval.py` | Baseline Recall@1/5/10 측정 |
| `02_hub_generation` | `01_generate_hub.py` / `01_generate_hub_laion.py` | Universal/Word-based/Cluster-based hub 단발 생성 (PGD) |
| | `02_generate_hub_imagebind.py` | ImageBind gradient로 hub 생성 (클러스터 크기 제한 없음) |
| | `03_batch_generate.py` / `04_batch_generate_laion.py` | 조건별 hub 30회씩 배치 생성 (각 480개) |
| | `05~07_evaluate_hub_asr*.py` | hub별 in-sample/held-out ASR 평가 |
| `03_detector` | `01~02_hubness_detector*.py` | 자체 구현 detector (median/MAD z-score, cluster spread, stability) |
| | `03~07_prepare_for_official*.py` | 공식 detector 입력용 데이터 준비 (전체/축소 비율/universal만/ImageBind) |
| | `08~12_run_official_scan*.py` | 공식 detector 실행 (mixed/real_queries 모드) |
| | `13~14_repeat_ratio_test*.py` | corpus 대비 hub 비율을 바꿔가며 반복 검증 |
| `04_model_comparison` | `01_build_gallery_imagebind.py` | ImageBind로 gallery/query 전체 재인코딩 |
| | `02~03_extract_embeddings*.py` | 8개 모델(CLIP 계열 6종 + RN50 + ImageBind)로 gallery/query/hub 재인코딩 |
| | `04_compare_models.py` | RSA / Alignment / Transfer ASR 계산 |
| `05_word_variance_analysis` | `01_pool_size_correlation.py` | Qt 캡션 풀 크기와 ASR의 상관관계 |
| | `02_cohesion_correlation.py` | 캡션 의미적 응집도와 ASR의 상관관계 |
| | `03_scale_effect_analysis.py` | 평가 규모(500 vs 25,000쿼리)가 Word/Cluster 순위에 미치는 영향 |

## Results Structure (결과 구조)

```
results/
├── openai_clip/          CLIP ViT-B/32(OpenAI) surrogate — hub 480개, 메인 실험
├── laion_clip/            OpenCLIP ViT-B/32(LAION-2B) surrogate — hub 480개, surrogate 검증용
├── imagebind/              ImageBind surrogate — hub 150개, 모델 차이 검증용
└── model_comparison/       8개 모델 간 RSA·Alignment·ASR 비교
```

각 surrogate 폴더 내부:
```
hubs/                       hub 이미지·임베딩·메타데이터
asr_summary.json            held-out ASR@k
detector_summary.json       자체 detector 결과 (openai_clip, laion_clip만)
official_detector/
  mixed/                     공식 detector, 자체 샘플링 쿼리
  real_queries/              공식 detector, 실제 caption 쿼리
  mixed_ratio_*/             corpus 대비 hub 비율을 2.0%/0.2%로 축소한 재검증 (openai_clip만)
  mixed_universal_only/      Universal hub만 분리한 재검증 (openai_clip만)
```

## Limitations

본 프로젝트는 자원 및 시간의 제약으로 인해 다음과 같은 한계를 가집니다.

- **개념별 공격 강도 편차의 원인 미규명**: Qt 캡션 풀 크기 및 의미적 응집도와 ASR 간의 상관관계를 분석하였으나(결과: Key Findings 2 참고), 두 변수가 서로 강하게 얽혀 있고 표본(10개 단어)이 작아 두 변수의 독립적 효과를 통계적으로 분리하지 못하였습니다. 이미지 내 현저성(saliency) 역시 원인 후보로 남아 있으나, 별도의 object detection 기반 라벨링 도구가 요구되어 본 프로젝트 범위에서 다루지 못하였습니다.
- **Word vs Cluster 불일치의 정확한 원인 미규명**: 세 가지 후보(평가 규모, surrogate 모델, Qt/클러스터 크기 제한)를 모두 기각하였으나, 대안적 원인을 제시하지는 못하였습니다.
- **`mixed` detector 실패의 정량적 원인 미규명**: 모달리티 불일치라는 정성적 설명에 도달하였으나, hub 최적화 강도(PGD step 수, 타겟 쿼리 수)를 논문 수준(1,000 step, 200쿼리)까지 확대하여 검증하지는 못하였습니다.
- **앙상블 surrogate 미검증**: 원 논문이 보고한 76% 수준의 transfer 성능이 3-모델 앙상블에서만 나온다는 사실은 논문 원문으로 확인하였으나, 이를 본 프로젝트에서 직접 구현·재현하지는 못하였습니다.
- **모델 비교 범위의 제한**: AudioCLIP은 설치 및 환경 구성의 리스크가 높아 비교 대상에서 제외하였습니다.
- **개념 단어 표본의 제한**: 10개 단어로 진행한 상관분석은 통계적 검정력이 낮아, 30~50개로 확장한 다중회귀 분석이 후속 검증으로 요구됩니다.
import os
import json
import re
import numpy as np
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict

RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip")
COMPARE_PATH = os.path.expanduser("~/projects/hubness/results/model_comparison/summary.json")

# ---- 1. 25,000쿼리 전체 기준 (day2, held-out ASR@1) 유형별 평균 ----
with open(os.path.join(RESULTS_DIR, "day2_hub_asr_summary.json"), "r", encoding="utf-8") as f:
    day2 = json.load(f)

def get_type(tag):
    if tag.startswith("universal"):
        return "universal"
    if tag.startswith("cluster"):
        return "cluster"
    m = re.match(r"domain_([a-zA-Z]+)_gc\d+", tag)
    if m:
        return f"domain_{m.group(1)}"
    return None

full_by_type = defaultdict(list)
for tag, v in day2.items():
    t = get_type(tag)
    if t:
        full_by_type[t].append(v["held_out"]["asr@1"])

full_mean = {t: float(np.mean(v)) for t, v in full_by_type.items()}

print("=== 25,000쿼리 전체 기준, 유형별 평균 held-out ASR@1 ===")
for t, a in sorted(full_mean.items(), key=lambda x: -x[1]):
    print(f"  {t:20s} {a*100:5.1f}%  (n={len(full_by_type[t])})")

# ---- 2. 500쿼리 서브샘플 기준 (11a/12, surrogate 자기 공간) ----
with open(COMPARE_PATH, "r", encoding="utf-8") as f:
    compare = json.load(f)

# surrogate(자기 공간) 항목 자동 탐지: overall asr@1이 가장 높은(≈1.0) 모델
surrogate_tag = max(compare["transfer_asr"].keys(),
                     key=lambda k: compare["transfer_asr"][k]["overall"]["asr@1"])
print(f"\n(surrogate로 판단된 모델: {surrogate_tag})")

sub_by_type = compare["transfer_asr"][surrogate_tag]["by_type"]
sub_mean = {t: v["asr@1"] for t, v in sub_by_type.items()}

print("\n=== 500쿼리 서브샘플 기준, 유형별 ASR@1 (surrogate 자기 공간) ===")
for t, a in sorted(sub_mean.items(), key=lambda x: -x[1]):
    print(f"  {t:20s} {a*100:5.1f}%")

# ---- 3. 공통 유형에 대해 두 순위 비교 ----
common_types = [t for t in full_mean if t in sub_mean]
x = np.array([full_mean[t] for t in common_types])
y = np.array([sub_mean[t] for t in common_types])

print(f"\n=== 두 평가 규모 간 순위 일치도 (n={len(common_types)}개 유형) ===")
rs, ps = spearmanr(x, y)
r, p = pearsonr(x, y)
print(f"Spearman r={rs:.3f}, p={ps:.4f}")
print(f"Pearson  r={r:.3f}, p={p:.4f}")

print("\n(유형, 25000쿼리ASR, 500쿼리ASR, 두 순위 차이) 상세:")
full_rank = {t: i+1 for i, t in enumerate(sorted(common_types, key=lambda t: -full_mean[t]))}
sub_rank = {t: i+1 for i, t in enumerate(sorted(common_types, key=lambda t: -sub_mean[t]))}
for t in sorted(common_types, key=lambda t: full_rank[t]):
    print(f"  {t:20s} 25000쿼리={full_mean[t]*100:5.1f}%(순위{full_rank[t]:2d})  "
          f"500쿼리={sub_mean[t]*100:5.1f}%(순위{sub_rank[t]:2d})  "
          f"순위차={abs(full_rank[t]-sub_rank[t])}")

# ---- 4. cluster vs word 순위 비교 (핵심 질문) ----
print("\n=== Cluster vs Word 최강 단어 비교 ===")
cluster_full = full_mean.get("cluster")
cluster_sub = sub_mean.get("cluster")
word_types_full = {t: v for t, v in full_mean.items() if t.startswith("domain_")}
word_types_sub = {t: v for t, v in sub_mean.items() if t.startswith("domain_")}
best_word_full = max(word_types_full, key=word_types_full.get)
best_word_sub = max(word_types_sub, key=word_types_sub.get)

print(f"25,000쿼리: cluster={cluster_full*100:.1f}%  최강 word({best_word_full})={word_types_full[best_word_full]*100:.1f}%  "
      f"-> {'Word가 더 강함' if word_types_full[best_word_full] > cluster_full else 'Cluster가 더 강함'}")
print(f"500쿼리:    cluster={cluster_sub*100:.1f}%  최강 word({best_word_sub})={word_types_sub[best_word_sub]*100:.1f}%  "
      f"-> {'Word가 더 강함' if word_types_sub[best_word_sub] > cluster_sub else 'Cluster가 더 강함'}")

out = {
    "full_25000_mean_asr": full_mean,
    "sub_500_mean_asr": sub_mean,
    "correlation": {"spearman_r": float(rs), "spearman_p": float(ps),
                     "pearson_r": float(r), "pearson_p": float(p)},
}
with open(os.path.join(RESULTS_DIR, "scale_effect_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n저장 -> {RESULTS_DIR}/scale_effect_analysis.json")

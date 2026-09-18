import os
import json
import re
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip")

WORDS = ["dog", "bicycle", "pizza", "cat", "elephant", "umbrella",
         "skateboard", "surfboard", "motorcycle", "kite"]

with open(os.path.join(EMB_DIR, "queries.json"), "r", encoding="utf-8") as f:
    queries = json.load(f)

# ---- 1. 단어별 caption 풀 크기 ----
pool_sizes = {}
for w in WORDS:
    pool_sizes[w] = sum(1 for q in queries if w in q["caption"].lower())

print("=== 단어별 caption 풀 크기 ===")
for w, n in sorted(pool_sizes.items(), key=lambda x: -x[1]):
    print(f"  {w}: {n}개")

# ---- 2. 단어별 평균 held-out ASR@1 ----
with open(os.path.join(RESULTS_DIR, "day2_hub_asr_summary.json"), "r", encoding="utf-8") as f:
    asr_data = json.load(f)

asr_by_word = defaultdict(list)
for tag, v in asr_data.items():
    m = re.match(r"domain_([a-zA-Z]+)_gc\d+", tag)
    if m and m.group(1) in WORDS:
        asr_by_word[m.group(1)].append(v["held_out"]["asr@1"])

mean_asr = {w: np.mean(vals) for w, vals in asr_by_word.items()}

print("\n=== 단어별 평균 held-out ASR@1 (hub 개수) ===")
for w, a in sorted(mean_asr.items(), key=lambda x: -x[1]):
    print(f"  {w}: {a*100:.1f}%  (n={len(asr_by_word[w])})")

# ---- 3. 상관관계 계산 ----
common_words = [w for w in WORDS if w in mean_asr and w in pool_sizes]
x = np.array([pool_sizes[w] for w in common_words])
y = np.array([mean_asr[w] for w in common_words])

print(f"\n=== Qt 풀 크기 vs ASR 상관관계 (n={len(common_words)}단어) ===")
r, p = pearsonr(x, y)
print(f"Pearson  r={r:.3f}, p={p:.4f}")
rs, ps = spearmanr(x, y)
print(f"Spearman r={rs:.3f}, p={ps:.4f}")

print("\n(단어, 풀크기, 평균ASR) 상세:")
for w in sorted(common_words, key=lambda w: -pool_sizes[w]):
    print(f"  {w:12s} pool={pool_sizes[w]:5d}  asr@1={mean_asr[w]*100:5.1f}%")

# ---- 4. 결과 저장 ----
out = {
    "pool_sizes": pool_sizes,
    "mean_asr": mean_asr,
    "correlation": {"pearson_r": float(r), "pearson_p": float(p),
                     "spearman_r": float(rs), "spearman_p": float(ps)},
}
with open(os.path.join(RESULTS_DIR, "word_variance_pool_size_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n저장 -> {RESULTS_DIR}/word_variance_pool_size_analysis.json")

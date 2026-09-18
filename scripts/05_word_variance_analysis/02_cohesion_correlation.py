import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import re

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip")
QUERY_EMB_PATH = os.path.expanduser("~/projects/hubness/results/openai_clip/ahd_input/query_embeddings.npy")

WORDS = ["dog", "bicycle", "pizza", "cat", "elephant", "umbrella",
         "skateboard", "surfboard", "motorcycle", "kite"]

with open(os.path.join(EMB_DIR, "queries.json"), "r", encoding="utf-8") as f:
    queries = json.load(f)

query_emb = np.load(QUERY_EMB_PATH)
assert query_emb.shape[0] == len(queries), "queries.json과 query_embeddings.npy 개수가 다름 -- 순서 확인 필요"

# ---- 1. 단어별 caption pool 인덱스 ----
word_indices = {}
for w in WORDS:
    idxs = [i for i, q in enumerate(queries) if w in q["caption"].lower()]
    word_indices[w] = idxs

# ---- 2. 응집도(cohesion) 계산: pool 안 caption들의 평균 pairwise cosine similarity ----
def cohesion_score(embs):
    # embs: (n, d), 이미 L2 정규화되어 있다고 가정 (안전하게 재정규화)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim_matrix = embs @ embs.T
    n = sim_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(sim_matrix[iu].mean())

cohesion = {}
pool_sizes = {}
for w, idxs in word_indices.items():
    embs = query_emb[idxs]
    pool_sizes[w] = len(idxs)
    cohesion[w] = cohesion_score(embs)

print("=== 단어별 응집도 (평균 pairwise cosine similarity) ===")
for w, c in sorted(cohesion.items(), key=lambda x: -x[1]):
    print(f"  {w:12s} cohesion={c:.4f}  (pool={pool_sizes[w]})")

# ---- 3. ASR 데이터 다시 로드 ----
with open(os.path.join(RESULTS_DIR, "day2_hub_asr_summary.json"), "r", encoding="utf-8") as f:
    asr_data = json.load(f)

asr_by_word = defaultdict(list)
for tag, v in asr_data.items():
    m = re.match(r"domain_([a-zA-Z]+)_gc\d+", tag)
    if m and m.group(1) in WORDS:
        asr_by_word[m.group(1)].append(v["held_out"]["asr@1"])

mean_asr = {w: float(np.mean(vals)) for w, vals in asr_by_word.items()}

# ---- 4. 상관관계 ----
common = [w for w in WORDS if w in mean_asr and w in cohesion]
x = np.array([cohesion[w] for w in common])
y = np.array([mean_asr[w] for w in common])

print(f"\n=== 응집도 vs ASR 상관관계 (n={len(common)}단어) ===")
r, p = pearsonr(x, y)
print(f"Pearson  r={r:.3f}, p={p:.4f}")
rs, ps = spearmanr(x, y)
print(f"Spearman r={rs:.3f}, p={ps:.4f}")

print("\n(단어, 응집도, 풀크기, ASR) 상세:")
for w in sorted(common, key=lambda w: -cohesion[w]):
    print(f"  {w:12s} cohesion={cohesion[w]:.4f}  pool={pool_sizes[w]:5d}  asr@1={mean_asr[w]*100:5.1f}%")

# ---- 5. 풀 크기와 응집도 자체의 관계도 확인 (혼입 여부 체크) ----
pool_x = np.array([pool_sizes[w] for w in common])
r_pool_coh, p_pool_coh = pearsonr(pool_x, x)
print(f"\n(참고) 풀 크기 vs 응집도 상관: r={r_pool_coh:.3f}, p={p_pool_coh:.4f}")

# ---- 6. 저장 ----
out = {
    "cohesion": cohesion,
    "pool_sizes": pool_sizes,
    "mean_asr": mean_asr,
    "correlation_cohesion_asr": {"pearson_r": float(r), "pearson_p": float(p),
                                   "spearman_r": float(rs), "spearman_p": float(ps)},
    "correlation_pool_cohesion": {"pearson_r": float(r_pool_coh), "pearson_p": float(p_pool_coh)},
}
with open(os.path.join(RESULTS_DIR, "word_variance_cohesion_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n저장 -> {RESULTS_DIR}/word_variance_cohesion_analysis.json")

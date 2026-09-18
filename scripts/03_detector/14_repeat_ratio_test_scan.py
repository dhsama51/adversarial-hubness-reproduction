import os
import json
import glob
import numpy as np
from adversarial_hubness_detector import scan
from sklearn.metrics import roc_auc_score

HUB_TARGETS = [10, 100]
RESULT_PATH = os.path.expanduser("~/projects/hubness/results/openai_clip/detector_ratio_repeat_summary.json")

all_seed_dirs = []
for n_hubs_target in HUB_TARGETS:
    out_base = os.path.expanduser(f"~/projects/hubness/results/openai_clip/ahd_input_repeat_{n_hubs_target}")
    dirs = sorted(glob.glob(os.path.join(out_base, "seed*")))
    all_seed_dirs.extend([(n_hubs_target, d) for d in dirs])
print(f"발견된 반복 세트: {len(all_seed_dirs)}개")

results_by_target = {t: [] for t in HUB_TARGETS}
details = []

for n_hubs_target, d in all_seed_dirs:
    seed_name = os.path.basename(d)
    with open(os.path.join(d, "combined_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    labels = np.array([1 if r["is_hub"] else 0 for r in meta])
    n_hubs = int(labels.sum())

    results = scan(
        embeddings_path=os.path.join(d, "combined_embeddings.npy"),
        metadata_path=os.path.join(d, "combined_metadata.json"),
        output_dir=d,
        k=10,
        num_queries=5000,
        query_sampling="mixed",
    )

    combined_scores = results.get("combined_scores")
    if combined_scores is not None:
        scores_arr = np.asarray(combined_scores)
        if scores_arr.shape[0] == len(labels):
            auc = roc_auc_score(labels, scores_arr)
            results_by_target[n_hubs_target].append(auc)
            details.append({"target": n_hubs_target, "seed": seed_name, "n_hubs": n_hubs, "roc_auc": float(auc)})
            print(f"[target={n_hubs_target}, {seed_name}] n_hubs={n_hubs} ROC-AUC={auc:.4f}")

summary = {"individual_aucs": details, "by_target": {}}
print(f"\n=== 비율별 반복 결과 ===")
for t, aucs in results_by_target.items():
    aucs = np.array(aucs)
    print(f"[hub={t}개] 평균={aucs.mean():.4f} 표준편차={aucs.std():.4f} 범위=[{aucs.min():.4f}, {aucs.max():.4f}] (n={len(aucs)})")
    summary["by_target"][str(t)] = {
        "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std()),
        "min_auc": float(aucs.min()), "max_auc": float(aucs.max()), "n": len(aucs),
    }

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n저장 -> {RESULT_PATH}")

import os
import json
import numpy as np
from adversarial_hubness_detector import scan
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve

DATA_DIR = os.path.expanduser("~/projects/hubness/results/imagebind/ahd_input")
OUT_DIR = os.path.expanduser("~/projects/hubness/results/imagebind/official_detector/mixed")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "combined_metadata.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)
labels = np.array([1 if r["is_hub"] else 0 for r in meta])
n_hubs = int(labels.sum())
print(f"전체 {len(labels)}개 (hub {n_hubs}개, 비율 {n_hubs/len(labels)*100:.2f}%)")

results = scan(
    embeddings_path=os.path.join(DATA_DIR, "combined_embeddings.npy"),
    metadata_path=os.path.join(DATA_DIR, "combined_metadata.json"),
    output_dir=OUT_DIR,
    k=10,
    num_queries=5000,
    query_sampling="mixed",
)

print(f"runtime: {results.get('runtime')}")
combined_scores = results.get("combined_scores")
if combined_scores is not None:
    scores_arr = np.asarray(combined_scores)
    if scores_arr.shape[0] == len(labels):
        auc = roc_auc_score(labels, scores_arr)
        fpr, tpr, thresholds = roc_curve(labels, scores_arr)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        preds = (scores_arr >= best_threshold).astype(int)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        print(f"ROC-AUC: {auc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")
        with open(os.path.join(OUT_DIR, "full_eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"n_hubs_true": n_hubs, "n_total": len(labels),
                       "roc_auc": float(auc), "precision": float(precision), "recall": float(recall)}, f, indent=2)
print(f"결과 -> {OUT_DIR}")

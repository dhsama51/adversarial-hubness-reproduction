import os
import json
import numpy as np
from adversarial_hubness_detector import scan
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve

DATA_DIR = os.path.expanduser("~/projects/hubness/results/laion_clip/ahd_input")
BASE_OUT_DIR = os.path.expanduser("~/projects/hubness/results/laion_clip/official_detector")

with open(os.path.join(DATA_DIR, "combined_metadata.json"), "r", encoding="utf-8") as f:
    combined_metadata = json.load(f)
labels = np.array([1 if rec["is_hub"] else 0 for rec in combined_metadata])

def run_scan(query_sampling, extra_kwargs, out_subdir):
    out_dir = os.path.join(BASE_OUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    kwargs = dict(
        embeddings_path=os.path.join(DATA_DIR, "combined_embeddings.npy"),
        metadata_path=os.path.join(DATA_DIR, "combined_metadata.json"),
        output_dir=out_dir, k=10, num_queries=5000, query_sampling=query_sampling,
    )
    kwargs.update(extra_kwargs)
    results = scan(**kwargs)
    print(f"\n=== [{query_sampling}] runtime: {results.get('runtime')} ===")

    combined_scores = results.get("combined_scores")
    if combined_scores is not None:
        scores_arr = np.asarray(combined_scores)
        np.save(os.path.join(out_dir, "combined_scores_full.npy"), scores_arr)
        if scores_arr.shape[0] == len(labels):
            auc = roc_auc_score(labels, scores_arr)
            fpr, tpr, thresholds = roc_curve(labels, scores_arr)
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            preds = (scores_arr >= best_threshold).astype(int)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            print(f"ROC-AUC: {auc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")
            with open(os.path.join(out_dir, "full_eval_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"roc_auc": float(auc), "precision": float(precision), "recall": float(recall)}, f, indent=2)

    with open(os.path.join(out_dir, "sdk_result_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results.get("json_report", {}), f, ensure_ascii=False, indent=2, default=str)
    return results

run_scan("mixed", {}, "mixed_default")
run_scan("real_queries", {"query_embeddings_path": os.path.join(DATA_DIR, "query_embeddings.npy")}, "real_queries")
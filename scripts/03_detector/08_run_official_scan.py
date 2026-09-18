import os
import json
import numpy as np
from adversarial_hubness_detector import scan

try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

DATA_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip/ahd_input")
BASE_OUT_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip/official_detector")

with open(os.path.join(DATA_DIR, "combined_metadata.json"), "r", encoding="utf-8") as f:
    combined_metadata = json.load(f)

labels = np.array([1 if rec["is_hub"] else 0 for rec in combined_metadata])
n_hubs_true = int(labels.sum())
print(f"전체 문서 {len(labels)}개 (hub {n_hubs_true}개, gallery {len(labels) - n_hubs_true}개)")


def run_scan(query_sampling, extra_kwargs, out_subdir):
    out_dir = os.path.join(BASE_OUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    kwargs = dict(
        embeddings_path=os.path.join(DATA_DIR, "combined_embeddings.npy"),
        metadata_path=os.path.join(DATA_DIR, "combined_metadata.json"),
        output_dir=out_dir,
        k=10,
        num_queries=5000,
        query_sampling=query_sampling,
    )
    kwargs.update(extra_kwargs)

    results = scan(**kwargs)

    print(f"\n=== [{query_sampling}] ===")
    print("runtime:", results.get("runtime"))

    # ---- 1. verdicts 전체 저장 ----
    verdicts = results.get("verdicts")
    if isinstance(verdicts, dict):
        # 키가 numpy int일 수 있으므로 문자열로 정규화
        verdicts_clean = {str(k): v for k, v in verdicts.items()}
        with open(os.path.join(out_dir, "verdicts_full.json"), "w", encoding="utf-8") as f:
            json.dump(verdicts_clean, f, ensure_ascii=False, indent=2)
        from collections import Counter
        print("verdict_counts (전체):", dict(Counter(verdicts.values())))

    # ---- 2. combined_scores 전체 저장 ----
    combined_scores = results.get("combined_scores")
    if combined_scores is not None:
        scores_arr = np.asarray(combined_scores)
        np.save(os.path.join(out_dir, "combined_scores_full.npy"), scores_arr)
        print(f"combined_scores shape: {scores_arr.shape}")

        # ---- 3. 480개 hub 전체 기준 Recall/Precision/ROC-AUC 계산 ----
        if HAS_SKLEARN and scores_arr.shape[0] == len(labels):
            auc = roc_auc_score(labels, scores_arr)
            fpr, tpr, thresholds = roc_curve(labels, scores_arr)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_idx]
            preds = (scores_arr >= best_threshold).astype(int)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)

            print(f"\n=== 전체 {n_hubs_true}개 hub 기준 성능 (combined_scores 기반) ===")
            print(f"ROC-AUC: {auc:.4f}")
            print(f"best threshold: {best_threshold:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")

            full_eval = {
                "n_hubs_true": n_hubs_true,
                "n_total": len(labels),
                "roc_auc": float(auc),
                "best_threshold": float(best_threshold),
                "precision_at_best_threshold": float(precision),
                "recall_at_best_threshold": float(recall),
            }
            with open(os.path.join(out_dir, "full_eval_summary.json"), "w", encoding="utf-8") as f:
                json.dump(full_eval, f, ensure_ascii=False, indent=2)
        elif scores_arr.shape[0] != len(labels):
            print(f"경고: combined_scores 길이({scores_arr.shape[0]})가 metadata 개수({len(labels)})와 안 맞음 — 순서 확인 필요")

    with open(os.path.join(out_dir, "sdk_result_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results.get("json_report", {}), f, ensure_ascii=False, indent=2, default=str)

    print(f"결과 -> {out_dir}")
    return results


run_scan("mixed", {}, "mixed_default")
run_scan("real_queries", {"query_embeddings_path": os.path.join(DATA_DIR, "query_embeddings.npy")}, "real_queries")
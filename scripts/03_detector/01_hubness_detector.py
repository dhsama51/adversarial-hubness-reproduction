import os
import json
import glob
import time
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
from tqdm import tqdm

SCRIPT_START = time.time()

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
INDEX_DIR = os.path.expanduser("~/projects/hubness/index")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip")
GALLERY_META_PATH = os.path.join(EMB_DIR, "metadata.json")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 10
N_CLUSTERS = 20
Z_THRESHOLD = 3.5
N_BOOTSTRAP = 30

gallery_emb = np.load(os.path.join(EMB_DIR, "image_embeddings.npy")).astype("float32")
with open(GALLERY_META_PATH, "r", encoding="utf-8") as f:
    gallery_records = json.load(f)
base_n = gallery_emb.shape[0]

meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hub_*_meta.json")))
hub_tags, hub_embs = [], []
for meta_path in meta_files:
    tag = os.path.basename(meta_path).replace("hub_", "").replace("_meta.json", "")
    emb_path = os.path.join(RESULTS_DIR, f"hub_{tag}_embedding.npy")
    if not os.path.exists(emb_path):
        continue
    hub_tags.append(tag)
    hub_embs.append(np.load(emb_path).astype("float32"))

n_hubs = len(hub_tags)
print(f"발견된 hub: {n_hubs}개")
assert n_hubs > 0, "hub_*_embedding.npy 파일이 없습니다."

hub_emb_matrix = np.concatenate(hub_embs, axis=0)
combined = np.concatenate([gallery_emb, hub_emb_matrix], axis=0).astype("float32")
n_items = combined.shape[0]
labels = np.array([0] * base_n + [1] * n_hubs)

combined_index = faiss.IndexFlatIP(combined.shape[1])
combined_index.add(combined)

model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)
captions = [r["caption"] for r in query_records]

def embed_texts(texts, batch_size=64):
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Query bank 임베딩"):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs, return_dict=True)
            feats = outputs.pooler_output
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype("float32")

q_emb = embed_texts(captions)
n_queries = q_emb.shape[0]

print(f"\nKMeans 클러스터링 (K={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10).fit(q_emb)
query_cluster = kmeans.labels_

print(f"전체 query bank로 top-{TOP_K} 검색 중...")
_, topk_idxs = combined_index.search(q_emb, TOP_K)

Nk = np.bincount(topk_idxs.reshape(-1), minlength=n_items)

retriever_clusters = {i: [] for i in range(n_items)}
for qi in range(n_queries):
    for item_idx in topk_idxs[qi]:
        retriever_clusters[item_idx].append(query_cluster[qi])

def cluster_spread(item_idx):
    cl = retriever_clusters[item_idx]
    if len(cl) == 0:
        return 0.0
    counts = np.bincount(cl, minlength=N_CLUSTERS)
    p = counts / counts.sum()
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(N_CLUSTERS))

spread = np.array([cluster_spread(i) for i in range(n_items)])

clean_Nk = Nk[:base_n]
median_Nk = np.median(clean_Nk)
mad = np.median(np.abs(clean_Nk - median_Nk))
mad_safe = mad if mad > 0 else 1e-6
z_score = 0.6745 * (Nk - median_Nk) / mad_safe

print(f"\nStability 측정 (bootstrap {N_BOOTSTRAP}회)...")
rng = np.random.default_rng(0)
boot_Nk_matrix = np.zeros((N_BOOTSTRAP, n_items), dtype=int)
for b in tqdm(range(N_BOOTSTRAP), desc="Bootstrap"):
    sampled_qi = rng.integers(0, n_queries, size=n_queries)
    sampled_idxs = topk_idxs[sampled_qi].reshape(-1)
    boot_Nk_matrix[b] = np.bincount(sampled_idxs, minlength=n_items)

stability_mean = boot_Nk_matrix.mean(axis=0)
stability_std = boot_Nk_matrix.std(axis=0)
stability_cv = stability_std / (stability_mean + 1e-6)

auc = roc_auc_score(labels, z_score)
preds = (z_score > Z_THRESHOLD).astype(int)
precision = precision_score(labels, preds, zero_division=0)
recall = recall_score(labels, preds, zero_division=0)

print(f"\n=== Detector 성능 (median/MAD z-score, threshold={Z_THRESHOLD}) ===")
print(f"ROC-AUC:   {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"(clean gallery median N_k={median_Nk:.1f}, MAD={mad:.1f})")

fpr, tpr, thresholds = roc_curve(labels, z_score)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

best_preds = (z_score >= best_threshold).astype(int)
best_precision = precision_score(labels, best_preds, zero_division=0)
best_recall = recall_score(labels, best_preds, zero_division=0)

print(f"\n=== 최적 threshold (Youden's J) ===")
print(f"threshold: {best_threshold:.2f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall:    {best_recall:.4f}")

print(f"\n=== Hub별 상세 지표 ===")
hub_details = []
for i, tag in enumerate(hub_tags):
    idx = base_n + i
    detail = {
        "tag": tag,
        "Nk": int(Nk[idx]),
        "z_score": float(z_score[idx]),
        "cluster_spread": float(spread[idx]),
        "stability_cv": float(stability_cv[idx]),
        "detected": bool(preds[idx]),
    }
    hub_details.append(detail)
    print(f"  [{tag}] N_k={detail['Nk']}  z={detail['z_score']:.2f}  "
          f"spread={detail['cluster_spread']:.3f}  stability_cv={detail['stability_cv']:.3f}")

clean_top5_idx = np.argsort(-clean_Nk)[:5]
natural_hub_details = []
for idx in clean_top5_idx:
    detail = {
        "record_id": gallery_records[idx]["id"],
        "caption": gallery_records[idx]["caption"],
        "Nk": int(Nk[idx]),
        "z_score": float(z_score[idx]),
        "cluster_spread": float(spread[idx]),
        "stability_cv": float(stability_cv[idx]),
    }
    natural_hub_details.append(detail)

summary = {
    "config": {"top_k": TOP_K, "n_clusters": N_CLUSTERS, "z_threshold": Z_THRESHOLD,
               "n_bootstrap": N_BOOTSTRAP, "n_hubs_evaluated": n_hubs, "n_queries": n_queries},
    "clean_gallery_stats": {"median_Nk": float(median_Nk), "mad": float(mad)},
    "detector_performance": {
        "roc_auc": float(auc), "precision": float(precision), "recall": float(recall),
        "best_threshold": float(best_threshold),
        "best_threshold_precision": float(best_precision),
        "best_threshold_recall": float(best_recall),
    },
    "hub_details": hub_details,
    "natural_hub_top5": natural_hub_details,
}

out_path = os.path.join(RESULTS_DIR, "day3_detector_summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n결과 저장 -> {out_path}")

elapsed = time.time() - SCRIPT_START
print(f"총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")

os._exit(0)
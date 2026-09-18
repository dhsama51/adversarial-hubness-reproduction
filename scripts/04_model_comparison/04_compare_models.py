import os
import json
import numpy as np
from scipy.stats import spearmanr
import faiss

OUT_DIR = os.path.expanduser("~/projects/hubness/results/model_comparison")
with open(os.path.join(OUT_DIR, "manifest.json"), "r", encoding="utf-8") as f:
    manifest = json.load(f)
MODEL_TAGS = ["clip_vitb32_base_openai", "openclip_vitb32_base_laion2b", "openclip_vitb16_base_laion2b", "openclip_vitl14_large_laion2b", "openclip_vitb32_base_laion400m", "openclip_vith14_huge_laion2b", "openclip_rn50_base_openai", "imagebind"]
K = 10

def load_model_data(tag):
    return {
        "gallery": np.load(os.path.join(OUT_DIR, f"{tag}_gallery.npy")),
        "query": np.load(os.path.join(OUT_DIR, f"{tag}_query.npy")),
        "hub": np.load(os.path.join(OUT_DIR, f"{tag}_hub.npy")),
    }

data = {tag: load_model_data(tag) for tag in MODEL_TAGS if os.path.exists(os.path.join(OUT_DIR, f"{tag}_gallery.npy"))}
available = list(data.keys())
print(f"사용 가능한 모델: {available}")

def rsa_score(emb_a, emb_b):
    sim_a = emb_a @ emb_a.T
    sim_b = emb_b @ emb_b.T
    iu = np.triu_indices_from(sim_a, k=1)
    corr, _ = spearmanr(sim_a[iu], sim_b[iu])
    return float(corr)

def alignment_score(gallery_a, query_a, gallery_b, query_b, k=K):
    idx_a = faiss.IndexFlatIP(gallery_a.shape[1]); idx_a.add(gallery_a.astype("float32"))
    idx_b = faiss.IndexFlatIP(gallery_b.shape[1]); idx_b.add(gallery_b.astype("float32"))
    _, topk_a = idx_a.search(query_a.astype("float32"), k)
    _, topk_b = idx_b.search(query_b.astype("float32"), k)
    overlaps = []
    for ra, rb in zip(topk_a, topk_b):
        overlaps.append(len(set(ra.tolist()) & set(rb.tolist())) / k)
    return float(np.mean(overlaps))

def transfer_asr(gallery, hub, query, k_list=(1, 5, 10)):
    n_gallery = gallery.shape[0]
    n_hub = hub.shape[0]
    combined = np.concatenate([gallery, hub], axis=0).astype("float32")
    index = faiss.IndexFlatIP(combined.shape[1]); index.add(combined)
    max_k = max(k_list)
    _, topk = index.search(query.astype("float32"), max_k)
    hub_positions = set(range(n_gallery, n_gallery + n_hub))
    hits = {k: 0 for k in k_list}
    for row in topk:
        row_list = row.tolist()
        for k in k_list:
            if any(p in hub_positions for p in row_list[:k]):
                hits[k] += 1
    n_q = query.shape[0]
    return {f"asr@{k}": hits[k] / n_q for k in k_list}

results = {"rsa": {}, "alignment": {}, "transfer_asr": {}}

for i in range(len(available)):
    for j in range(i + 1, len(available)):
        a, b = available[i], available[j]
        pair = f"{a}__{b}"
        print(f"\n=== {pair} ===")

        rsa = rsa_score(data[a]["gallery"], data[b]["gallery"])
        results["rsa"][pair] = rsa
        print(f"RSA (이미지 인코더 구조 유사도): {rsa:.4f}")

        align = alignment_score(data[a]["gallery"], data[a]["query"], data[b]["gallery"], data[b]["query"])
        results["alignment"][pair] = align
        print(f"Alignment (query top-{K} overlap): {align:.4f}")

hub_type_labels = manifest.get("hub_type_labels", {})
hub_tags_list = manifest["hub_tags"]  # data[tag]["hub"]의 행(row) 순서와 일치

def transfer_asr_by_type(gallery, hub, query, hub_tags_list, hub_type_labels, k_list=(1, 5, 10)):
    """유형별로 hub를 나눠서 각각 ASR 계산"""
    types = sorted(set(hub_type_labels.get(t, "unknown") for t in hub_tags_list))
    result = {}
    for t in types:
        idxs = [i for i, tag in enumerate(hub_tags_list) if hub_type_labels.get(tag) == t]
        if not idxs:
            continue
        sub_hub = hub[idxs]
        result[t] = transfer_asr(gallery, sub_hub, query, k_list)
        result[t]["n_hubs"] = len(idxs)
    return result

print("\n=== Transfer ASR (각 모델 자체 공간에서 hub 재인코딩 결과, 유형별) ===")
for tag in available:
    asr_overall = transfer_asr(data[tag]["gallery"], data[tag]["hub"], data[tag]["query"])
    asr_by_type = transfer_asr_by_type(
        data[tag]["gallery"], data[tag]["hub"], data[tag]["query"],
        hub_tags_list, hub_type_labels
    )
    results["transfer_asr"][tag] = {"overall": asr_overall, "by_type": asr_by_type}

    print(f"\n[{tag}] 전체: {asr_overall}")
    for t, v in asr_by_type.items():
        print(f"  - {t} (n={v['n_hubs']}): asr@1={v['asr@1']:.3f}, asr@5={v['asr@5']:.3f}, asr@10={v['asr@10']:.3f}")

with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n결과 저장 -> {OUT_DIR}/summary.json")
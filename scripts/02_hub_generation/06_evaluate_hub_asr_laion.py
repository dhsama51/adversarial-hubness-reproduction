import os
import json
import time
import glob
import random
import numpy as np
import torch
import open_clip
import faiss
from tqdm import tqdm

SCRIPT_START = time.time()

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
INDEX_DIR = os.path.expanduser("~/projects/hubness/index")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/laion_clip")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")

MODEL_ARCH = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_LIST = [1, 5, 10]
N_HELDOUT = 400
SEED = 123

model, _, _ = open_clip.create_model_and_transforms(MODEL_ARCH, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_ARCH)
model = model.to(DEVICE).eval()

base_index = faiss.read_index(os.path.join(INDEX_DIR, "image_index_laion.faiss"))
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)
query_by_id = {r["query_id"]: r for r in query_records}

def embed_texts(texts, batch_size=64):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size]).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype("float32")

def compute_asr(index_with_hub, hub_position, query_captions, max_k):
    q_emb = embed_texts(query_captions)
    _, idxs = index_with_hub.search(q_emb, max_k)
    hits = {k: 0 for k in K_LIST}
    for row in idxs:
        row_list = row.tolist()
        for k in K_LIST:
            if hub_position in row_list[:k]:
                hits[k] += 1
    n = len(query_captions)
    return {f"asr@{k}": hits[k] / n for k in K_LIST}, n

random.seed(SEED)
all_query_ids = [r["query_id"] for r in query_records]

meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hubs", "hub_*_meta.json")))
print(f"발견된 hub meta 파일: {len(meta_files)}개")

summary = {}
for meta_path in meta_files:
    with open(meta_path, "r", encoding="utf-8") as f:
        hub_meta = json.load(f)
    tag = os.path.basename(meta_path).replace("hub_", "").replace("_meta.json", "")
    emb_path = os.path.join(RESULTS_DIR, "hubs", f"hub_{tag}_embedding.npy")
    if not os.path.exists(emb_path):
        continue
    hub_emb = np.load(emb_path).astype("float32")

    index_with_hub = faiss.IndexFlatIP(hub_emb.shape[1])
    index_with_hub.add(base_index.reconstruct_n(0, base_index.ntotal))
    index_with_hub.add(hub_emb)
    hub_position = base_index.ntotal

    qt_query_ids = hub_meta["qt_query_ids"]
    qt_captions = [query_by_id[q]["caption"] for q in qt_query_ids]
    qt_image_ids = {query_by_id[q]["image_id"] for q in qt_query_ids}

    heldout_pool = [qid for qid in all_query_ids if query_by_id[qid]["image_id"] not in qt_image_ids]
    heldout_query_ids = random.sample(heldout_pool, min(N_HELDOUT, len(heldout_pool)))
    heldout_captions = [query_by_id[q]["caption"] for q in heldout_query_ids]

    max_k = max(K_LIST)
    in_sample_asr, n_in = compute_asr(index_with_hub, hub_position, qt_captions, max_k)
    heldout_asr, n_out = compute_asr(index_with_hub, hub_position, heldout_captions, max_k)

    summary[tag] = {
        "hub_meta": hub_meta,
        "in_sample": {"n": n_in, **in_sample_asr},
        "held_out": {"n": n_out, **heldout_asr},
    }
    print(f"[{tag}] in-sample ASR@1={in_sample_asr['asr@1']:.3f}  held-out ASR@1={heldout_asr['asr@1']:.3f}")

out_path = os.path.join(RESULTS_DIR, "asr_summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n저장 -> {out_path}")
print(f"총 소요 시간: {(time.time()-SCRIPT_START)/60:.1f}분")
os._exit(0)
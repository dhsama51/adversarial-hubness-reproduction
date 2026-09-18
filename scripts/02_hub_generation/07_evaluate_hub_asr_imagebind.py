import os
import sys
import json
import time
import glob
import random
import numpy as np
import torch
import faiss

SCRIPT_START = time.time()
sys.path.insert(0, os.path.expanduser("~/projects/hubness/external/ImageBind"))
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
INDEX_DIR = os.path.expanduser("~/projects/hubness/index")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/imagebind")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_LIST = [1, 5, 10]
N_HELDOUT = 400
SEED = 123

print("모델 로딩...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval().to(DEVICE)

base_index = faiss.read_index(os.path.join(INDEX_DIR, "image_index_imagebind.faiss"))
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)
query_by_id = {r["query_id"]: r for r in query_records}
all_query_ids = [r["query_id"] for r in query_records]

def embed_texts(texts, batch_size=64):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = {ModalityType.TEXT: data.load_and_transform_text(batch, DEVICE)}
        with torch.no_grad():
            out = model(inputs)[ModalityType.TEXT]
            out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype("float32")

def compute_asr(index_with_hub, hub_position, captions, max_k):
    q_emb = embed_texts(captions)
    _, idxs = index_with_hub.search(q_emb, max_k)
    hits = {k: 0 for k in K_LIST}
    for row in idxs:
        row_list = row.tolist()
        for k in K_LIST:
            if hub_position in row_list[:k]:
                hits[k] += 1
    n = len(captions)
    return {f"asr@{k}": hits[k] / n for k in K_LIST}, n

random.seed(SEED)
meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hubs", "hub_*_meta.json")))
print(f"발견된 hub: {len(meta_files)}개")

summary = {}
for i, meta_path in enumerate(meta_files):
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
    heldout_ids = random.sample(heldout_pool, min(N_HELDOUT, len(heldout_pool)))
    heldout_captions = [query_by_id[q]["caption"] for q in heldout_ids]

    max_k = max(K_LIST)
    in_sample, n_in = compute_asr(index_with_hub, hub_position, qt_captions, max_k)
    held_out, n_out = compute_asr(index_with_hub, hub_position, heldout_captions, max_k)

    summary[tag] = {"hub_meta": hub_meta, "in_sample": {"n": n_in, **in_sample}, "held_out": {"n": n_out, **held_out}}
    print(f"[{i+1}/{len(meta_files)}] {tag}  held-out ASR@1={held_out['asr@1']:.3f}")

out_path = os.path.join(RESULTS_DIR, "asr_summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n저장 -> {out_path}")
print(f"총 소요 시간: {(time.time()-SCRIPT_START)/60:.1f}분")
os._exit(0)

import os
import json
import glob
import time
import numpy as np
import torch
import open_clip
from tqdm import tqdm

SCRIPT_START = time.time()

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/laion_clip")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")

OUT_DIR = os.path.expanduser("~/projects/hubness/results/laion_clip/ahd_input")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ARCH = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gallery_emb = np.load(os.path.join(EMB_DIR, "image_embeddings_laion.npy")).astype("float32")
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

hub_emb_matrix = np.concatenate(hub_embs, axis=0)
combined_emb = np.concatenate([gallery_emb, hub_emb_matrix], axis=0).astype("float32")
np.save(os.path.join(OUT_DIR, "combined_embeddings.npy"), combined_emb)

combined_records = []
for i in range(base_n):
    combined_records.append({"doc_id": f"gallery_{i}", "is_hub": False, "hub_tag": None})
for tag in hub_tags:
    combined_records.append({"doc_id": f"hub_{tag}", "is_hub": True, "hub_tag": tag})
with open(os.path.join(OUT_DIR, "combined_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(combined_records, f, ensure_ascii=False, indent=2)

query_emb = np.load(os.path.join(EMB_DIR, "query_embeddings_laion.npy"))
np.save(os.path.join(OUT_DIR, "query_embeddings.npy"), query_emb)

print(f"gallery+hub: {combined_emb.shape}, hub {len(hub_tags)}개")
print(f"저장 -> {OUT_DIR}")
print(f"총 소요 시간: {(time.time()-SCRIPT_START)/60:.1f}분")
os._exit(0)
import os
import json
import glob
import time
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

SCRIPT_START = time.time()

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")
GALLERY_META_PATH = os.path.join(EMB_DIR, "metadata.json")

OUT_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip/ahd_input")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. gallery + 모든 hub 임베딩 결합 ----
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
print(f"gallery {base_n}개 + hub {n_hubs}개 결합")

hub_emb_matrix = np.concatenate(hub_embs, axis=0)
combined_emb = np.concatenate([gallery_emb, hub_emb_matrix], axis=0).astype("float32")
np.save(os.path.join(OUT_DIR, "combined_embeddings.npy"), combined_emb)

# ---- 2. 결합 metadata (list-of-dicts, is_hub 라벨 포함) ----
combined_records = []
for r in gallery_records:
    combined_records.append({"doc_id": f"gallery_{r['id']}", "caption": r["caption"], "is_hub": False, "hub_tag": None})
for tag in hub_tags:
    combined_records.append({"doc_id": f"hub_{tag}", "caption": None, "is_hub": True, "hub_tag": tag})

with open(os.path.join(OUT_DIR, "combined_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(combined_records, f, ensure_ascii=False, indent=2)

# ---- 3. 실제 caption(25,000개) 쿼리 임베딩 (real query mode용) ----
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)
captions = [r["caption"] for r in query_records]

model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

feats_list = []
for i in tqdm(range(0, len(captions), 128), desc="쿼리 텍스트 임베딩"):
    batch = captions[i:i + 128]
    inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs, return_dict=True)
        feats = outputs.pooler_output
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    feats_list.append(feats.cpu().numpy())

query_emb = np.concatenate(feats_list, axis=0).astype("float32")
np.save(os.path.join(OUT_DIR, "query_embeddings.npy"), query_emb)

print(f"\n저장 완료 -> {OUT_DIR}")
print(f"  combined_embeddings.npy: {combined_emb.shape}")
print(f"  combined_metadata.json: {len(combined_records)}개 (hub {n_hubs}개 라벨링됨)")
print(f"  query_embeddings.npy: {query_emb.shape}")

elapsed = time.time() - SCRIPT_START
print(f"\n총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")

os._exit(0)
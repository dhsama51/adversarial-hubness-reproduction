import os
import json
import glob
import random
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")

OUT_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip/ahd_input_ratio_test")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_HUBS_TARGET = 10  # gallery 5000 기준 2.0% 비율
SEED = 42

random.seed(SEED)

gallery_emb = np.load(os.path.join(EMB_DIR, "image_embeddings.npy")).astype("float32")
base_n = gallery_emb.shape[0]

meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hub_*_meta.json")))
hub_tags_all = [os.path.basename(f).replace("hub_", "").replace("_meta.json", "") for f in meta_files]
hub_tags_all = [t for t in hub_tags_all if os.path.exists(os.path.join(RESULTS_DIR, f"hub_{t}_embedding.npy"))]

hub_sample_tags = random.sample(hub_tags_all, min(N_HUBS_TARGET, len(hub_tags_all)))
print(f"전체 hub {len(hub_tags_all)}개 중 {len(hub_sample_tags)}개 샘플링 "
      f"(gallery {base_n}개 대비 비율 {len(hub_sample_tags)/(base_n+len(hub_sample_tags))*100:.2f}%)")

hub_embs = [np.load(os.path.join(RESULTS_DIR, f"hub_{t}_embedding.npy")).astype("float32") for t in hub_sample_tags]
hub_emb_matrix = np.concatenate(hub_embs, axis=0)
combined_emb = np.concatenate([gallery_emb, hub_emb_matrix], axis=0).astype("float32")
np.save(os.path.join(OUT_DIR, "combined_embeddings.npy"), combined_emb)

with open(os.path.join(EMB_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    gallery_records = json.load(f)

combined_records = []
for r in gallery_records:
    combined_records.append({"doc_id": f"gallery_{r['id']}", "caption": r["caption"], "is_hub": False, "hub_tag": None})
for tag in hub_sample_tags:
    combined_records.append({"doc_id": f"hub_{tag}", "caption": None, "is_hub": True, "hub_tag": tag})
with open(os.path.join(OUT_DIR, "combined_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(combined_records, f, ensure_ascii=False, indent=2)

# real_queries용 쿼리 임베딩도 재사용 가능하면 복사, 없으면 새로 생성
existing_q = os.path.expanduser("~/projects/hubness/results/openai_clip/ahd_input/query_embeddings.npy")
if os.path.exists(existing_q):
    import shutil
    shutil.copy(existing_q, os.path.join(OUT_DIR, "query_embeddings.npy"))
    print("기존 query_embeddings.npy 재사용")
else:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        query_records = json.load(f)
    captions = [r["caption"] for r in query_records]
    model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    feats_list = []
    for i in tqdm(range(0, len(captions), 128), desc="쿼리 텍스트 임베딩"):
        batch = captions[i:i+128]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs, return_dict=True)
            feats = outputs.pooler_output
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        feats_list.append(feats.cpu().numpy())
    query_emb = np.concatenate(feats_list, axis=0).astype("float32")
    np.save(os.path.join(OUT_DIR, "query_embeddings.npy"), query_emb)

print(f"저장 완료 -> {OUT_DIR}")
print(f"  combined_embeddings.npy: {combined_emb.shape}")
print(f"  combined_metadata.json: {len(combined_records)}개 (hub {len(hub_sample_tags)}개)")

os._exit(0)

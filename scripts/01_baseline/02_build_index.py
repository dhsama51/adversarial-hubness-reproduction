import os
import json
import time
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import faiss
from tqdm import tqdm

SCRIPT_START = time.time()

DATA_DIR = os.path.expanduser("~/projects/hubness/data")
IMG_DIR = os.path.join(DATA_DIR, "images")
DATA_META_PATH = os.path.join(DATA_DIR, "metadata.json")

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
INDEX_DIR = os.path.expanduser("~/projects/hubness/index")
GALLERY_META_PATH = os.path.join(EMB_DIR, "metadata.json")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

BATCH_SIZE = 32
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

with open(DATA_META_PATH, "r", encoding="utf-8") as f:
    data_records = json.load(f)
print(f"로컬 이미지 {len(data_records)}장 확인")

model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

gallery_records = []
embeddings = []
images_batch, meta_batch = [], []

def flush():
    if not images_batch:
        return
    inputs = processor(images=images_batch, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs, return_dict=True)
        feats = outputs.pooler_output
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    embeddings.append(feats.cpu().numpy())
    gallery_records.extend(meta_batch)

for rec in tqdm(data_records, desc="갤러리 이미지 임베딩"):
    img = Image.open(os.path.join(IMG_DIR, rec["filename"])).convert("RGB")
    images_batch.append(img)
    meta_batch.append({"id": rec["id"], "caption": rec["captions"][0]})

    if len(images_batch) == BATCH_SIZE:
        flush()
        images_batch, meta_batch = [], []

flush()

emb_matrix = np.concatenate(embeddings, axis=0).astype("float32")
print("갤러리 임베딩 shape:", emb_matrix.shape)

np.save(os.path.join(EMB_DIR, "image_embeddings.npy"), emb_matrix)
with open(GALLERY_META_PATH, "w", encoding="utf-8") as f:
    json.dump(gallery_records, f, ensure_ascii=False, indent=2)

index = faiss.IndexFlatIP(emb_matrix.shape[1])
index.add(emb_matrix)
faiss.write_index(index, os.path.join(INDEX_DIR, "image_index.faiss"))

query_records = []
qid = 0
for rec in data_records:
    for cap in rec["captions"]:
        query_records.append({"query_id": qid, "image_id": rec["id"], "caption": cap})
        qid += 1

with open(QUERIES_PATH, "w", encoding="utf-8") as f:
    json.dump(query_records, f, ensure_ascii=False, indent=2)

print(f"완료: 갤러리 {index.ntotal}개, 쿼리 {len(query_records)}개")
print(f"임베딩       -> {EMB_DIR}/image_embeddings.npy")
print(f"갤러리 메타  -> {GALLERY_META_PATH}")
print(f"쿼리 메타    -> {QUERIES_PATH}")
print(f"인덱스       -> {INDEX_DIR}/image_index.faiss")

elapsed = time.time() - SCRIPT_START
print(f"\n총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")

os._exit(0)
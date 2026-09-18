import os
import json
import time
import numpy as np
import torch
from PIL import Image
import open_clip
import faiss
from tqdm import tqdm

SCRIPT_START = time.time()

DATA_DIR = os.path.expanduser("~/projects/hubness/data")
IMG_DIR = os.path.join(DATA_DIR, "images")
DATA_META_PATH = os.path.join(DATA_DIR, "metadata.json")
EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")

MODEL_ARCH = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ARCH, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_ARCH)
model = model.to(DEVICE).eval()

with open(DATA_META_PATH, "r", encoding="utf-8") as f:
    data_records = json.load(f)

images_batch, ids_batch = [], []
embeddings = []

def flush():
    if not images_batch:
        return
    batch_tensor = torch.stack(images_batch).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(batch_tensor)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    embeddings.append(feats.cpu().numpy())

for rec in tqdm(data_records, desc="LAION 갤러리 이미지 임베딩"):
    img = preprocess(Image.open(os.path.join(IMG_DIR, rec["filename"])).convert("RGB"))
    images_batch.append(img)
    ids_batch.append(rec["id"])
    if len(images_batch) == BATCH_SIZE:
        flush()
        images_batch = []
flush()

emb_matrix = np.concatenate(embeddings, axis=0).astype("float32")
np.save(os.path.join(EMB_DIR, "image_embeddings_laion.npy"), emb_matrix)

index = faiss.IndexFlatIP(emb_matrix.shape[1])
index.add(emb_matrix)
faiss.write_index(index, os.path.expanduser("~/projects/hubness/index/image_index_laion.faiss"))

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)
captions = [r["caption"] for r in query_records]

feats_list = []
for i in tqdm(range(0, len(captions), 128), desc="LAION 쿼리 텍스트 임베딩"):
    tokens = tokenizer(captions[i:i+128]).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    feats_list.append(feats.cpu().numpy())
query_emb = np.concatenate(feats_list, axis=0).astype("float32")
np.save(os.path.join(EMB_DIR, "query_embeddings_laion.npy"), query_emb)

print(f"gallery: {emb_matrix.shape}, query: {query_emb.shape}")
print(f"총 소요 시간: {(time.time()-SCRIPT_START)/60:.1f}분")
os._exit(0)
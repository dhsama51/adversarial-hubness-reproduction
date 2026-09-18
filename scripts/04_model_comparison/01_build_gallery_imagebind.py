import os
import sys
import json
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import faiss

SCRIPT_START = time.time()
sys.path.insert(0, os.path.expanduser("~/projects/hubness/external/ImageBind"))
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

DATA_DIR = os.path.expanduser("~/projects/hubness/data")
IMG_DIR = os.path.join(DATA_DIR, "images")
DATA_META_PATH = os.path.join(DATA_DIR, "metadata.json")
EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")
INDEX_DIR = os.path.expanduser("~/projects/hubness/index")
os.makedirs(INDEX_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

print("모델 로딩...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval().to(DEVICE)

with open(DATA_META_PATH, "r", encoding="utf-8") as f:
    data_records = json.load(f)
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)

# ---- gallery 5,000장 인코딩 ----
gallery_feats = []
batch = []
for i, rec in enumerate(data_records):
    img = Image.open(os.path.join(IMG_DIR, rec["filename"])).convert("RGB")
    batch.append(preprocess(img))
    if len(batch) == BATCH_SIZE or i == len(data_records) - 1:
        bt = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            out = model({ModalityType.VISION: bt})[ModalityType.VISION]
            out = out / out.norm(dim=-1, keepdim=True)
        gallery_feats.append(out.cpu().numpy())
        batch = []
        if (i + 1) % 500 == 0:
            print(f"gallery {i+1}/{len(data_records)} ({time.time()-SCRIPT_START:.0f}초 경과)")

gallery_emb = np.concatenate(gallery_feats, axis=0).astype("float32")
np.save(os.path.join(EMB_DIR, "image_embeddings_imagebind.npy"), gallery_emb)

index = faiss.IndexFlatIP(gallery_emb.shape[1])
index.add(gallery_emb)
faiss.write_index(index, os.path.join(INDEX_DIR, "image_index_imagebind.faiss"))
print(f"gallery 완료: {gallery_emb.shape}")

# ---- query 25,000개 인코딩 ----
captions = [r["caption"] for r in query_records]
query_feats = []
for i in range(0, len(captions), 128):
    batch_texts = captions[i:i+128]
    inputs = {ModalityType.TEXT: data.load_and_transform_text(batch_texts, DEVICE)}
    with torch.no_grad():
        out = model(inputs)[ModalityType.TEXT]
        out = out / out.norm(dim=-1, keepdim=True)
    query_feats.append(out.cpu().numpy())
    if (i // 128) % 20 == 0:
        print(f"query {i}/{len(captions)} ({time.time()-SCRIPT_START:.0f}초 경과)")

query_emb = np.concatenate(query_feats, axis=0).astype("float32")
np.save(os.path.join(EMB_DIR, "query_embeddings_imagebind.npy"), query_emb)
print(f"query 완료: {query_emb.shape}")

elapsed = time.time() - SCRIPT_START
print(f"\n총 소요 시간: {elapsed/60:.1f}분")
os._exit(0)

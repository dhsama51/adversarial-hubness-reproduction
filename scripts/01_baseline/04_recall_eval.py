import os
import json
import time
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss
from tqdm import tqdm

SCRIPT_START = time.time()

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
INDEX_DIR = os.path.expanduser("~/projects/hubness/index")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_LIST = [1, 5, 10]
BATCH_SIZE = 64

model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

index = faiss.read_index(os.path.join(INDEX_DIR, "image_index.faiss"))
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)

captions = [r["caption"] for r in query_records]
gt_ids = [r["image_id"] for r in query_records]

max_k = max(K_LIST)
hits = {k: 0 for k in K_LIST}
total = 0

for start in tqdm(range(0, len(captions), BATCH_SIZE), desc="Recall@k 평가 (text-to-image)"):
    batch_captions = captions[start:start + BATCH_SIZE]
    batch_gt = gt_ids[start:start + BATCH_SIZE]

    inputs = processor(text=batch_captions, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs, return_dict=True)
        feats = outputs.pooler_output
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    q_emb = feats.cpu().numpy().astype("float32")

    _, idxs = index.search(q_emb, max_k)

    for row, gt in zip(idxs, batch_gt):
        row_list = row.tolist()
        for k in K_LIST:
            if gt in row_list[:k]:
                hits[k] += 1
    total += len(batch_captions)

print(f"\n총 쿼리 수: {total}")
result_summary = {}
for k in K_LIST:
    recall_k = hits[k] / total
    result_summary[f"recall@{k}"] = recall_k
    print(f"Recall@{k}: {recall_k:.4f} ({hits[k]}/{total})")

with open(os.path.join(RESULTS_DIR, "recall_baseline.json"), "w", encoding="utf-8") as f:
    json.dump(result_summary, f, indent=2)

print(f"\n결과 저장 -> {RESULTS_DIR}/recall_baseline.json")

elapsed = time.time() - SCRIPT_START
print(f"총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")

os._exit(0)
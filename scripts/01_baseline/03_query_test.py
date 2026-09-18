import os
import json
import time
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss

SCRIPT_START = time.time()

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
INDEX_DIR = os.path.expanduser("~/projects/hubness/index")
META_PATH = os.path.join(EMB_DIR, "metadata.json")

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

index = faiss.read_index(os.path.join(INDEX_DIR, "image_index.faiss"))
with open(META_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

def embed_text(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs, return_dict=True)
        feats = outputs.pooler_output
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")

def search(query, k=5):
    q_emb = embed_text(query)
    scores, idxs = index.search(q_emb, k)
    print(f"\n쿼리: \"{query}\"")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        rec = records[idx]
        print(f"  {rank}위 (score={score:.4f}): id={rec['id']} | caption: {rec['caption']}")

if __name__ == "__main__":
    test_queries = [
        "a dog running on the grass",
        "a person riding a bicycle",
        "a plate of food on a table",
    ]
    for q in test_queries:
        search(q, k=5)

    elapsed = time.time() - SCRIPT_START
    print(f"\n총 소요 시간: {elapsed:.1f}초")

    os._exit(0)
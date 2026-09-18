import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/projects/hubness/external/ImageBind"))

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

SCRIPT_START = time.time()

DATA_DIR = os.path.expanduser("~/projects/hubness/data")
IMG_DIR = os.path.join(DATA_DIR, "images")
HUB_SURROGATE = "openai_clip"  # "openai_clip" | "laion_clip" | "imagebind"
RESULTS_DIR = os.path.expanduser(f"~/projects/hubness/results/{HUB_SURROGATE}/hubs")
OUT_DIR = os.path.expanduser("~/projects/hubness/results/model_comparison")

with open(os.path.join(OUT_DIR, "manifest.json"), "r", encoding="utf-8") as f:
    manifest = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("ImageBind 모델 로딩...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval().to(DEVICE)

def encode_images(paths, batch_size=16):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(batch_paths, DEVICE)}
        with torch.no_grad():
            out = model(inputs)[ModalityType.VISION]
        feats.append(out.cpu().numpy())
        print(f"  이미지 {i+len(batch_paths)}/{len(paths)}")
    return np.concatenate(feats, axis=0).astype("float32")

def encode_texts(texts, batch_size=64):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = {ModalityType.TEXT: data.load_and_transform_text(batch_texts, DEVICE)}
        with torch.no_grad():
            out = model(inputs)[ModalityType.TEXT]
        feats.append(out.cpu().numpy())
        print(f"  텍스트 {i+len(batch_texts)}/{len(texts)}")
    return np.concatenate(feats, axis=0).astype("float32")

gallery_paths = [os.path.join(IMG_DIR, fn) for fn in manifest["gallery_filenames"]]
hub_paths = [os.path.join(RESULTS_DIR, f"hub_{t}.png") for t in manifest["hub_tags"]]

print("\n=== ImageBind ===")
gallery_emb = encode_images(gallery_paths)
query_emb = encode_texts(manifest["query_captions"])
hub_emb = encode_images(hub_paths)

np.save(os.path.join(OUT_DIR, "imagebind_gallery.npy"), gallery_emb)
np.save(os.path.join(OUT_DIR, "imagebind_query.npy"), query_emb)
np.save(os.path.join(OUT_DIR, "imagebind_hub.npy"), hub_emb)

print(f"gallery={gallery_emb.shape} query={query_emb.shape} hub={hub_emb.shape}")

elapsed = time.time() - SCRIPT_START
print(f"\n총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")

os._exit(0)
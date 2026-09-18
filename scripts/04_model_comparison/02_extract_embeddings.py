import os
import json
import glob
import random
import time
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import open_clip
from tqdm import tqdm

SCRIPT_START = time.time()

DATA_DIR = os.path.expanduser("~/projects/hubness/data")
IMG_DIR = os.path.join(DATA_DIR, "images")
DATA_META_PATH = os.path.join(DATA_DIR, "metadata.json")
EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")
HUB_SURROGATE = "openai_clip"  # "openai_clip" | "laion_clip" | "imagebind"
RESULTS_DIR = os.path.expanduser(f"~/projects/hubness/results/{HUB_SURROGATE}/hubs")

OUT_DIR = os.path.expanduser("~/projects/hubness/results/model_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GALLERY = 500
N_QUERIES = 500
N_PER_TYPE = 20
SEED = 7

random.seed(SEED)

with open(DATA_META_PATH, "r", encoding="utf-8") as f:
    data_records = json.load(f)
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)

gallery_sample = random.sample(data_records, N_GALLERY)
query_sample = random.sample(query_records, N_QUERIES)

meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hub_*_meta.json")))
hub_tags_all = [os.path.basename(f).replace("hub_", "").replace("_meta.json", "") for f in meta_files]
hub_tags_all = [t for t in hub_tags_all if os.path.exists(os.path.join(RESULTS_DIR, f"hub_{t}.png"))]

def sample_hubs_by_type(hub_tags_all, n_per_type=20):
    import re
    universal = [t for t in hub_tags_all if t.startswith("universal")]
    cluster = [t for t in hub_tags_all if t.startswith("cluster")]
    domain_all = [t for t in hub_tags_all if t.startswith("domain_")]
    domain_by_word = {}
    for t in domain_all:
        m = re.match(r"domain_([a-zA-Z]+)_gc\d+", t)
        if m:
            word = m.group(1)
            domain_by_word.setdefault(word, []).append(t)

    sampled = []
    type_labels = {}

    n = min(n_per_type, len(universal))
    chosen = random.sample(universal, n)
    sampled.extend(chosen)
    for t in chosen:
        type_labels[t] = "universal"
    print(f"  universal: {len(universal)}개 중 {n}개 샘플링")

    for word, group in sorted(domain_by_word.items()):
        n = min(n_per_type, len(group))
        chosen = random.sample(group, n)
        sampled.extend(chosen)
        for t in chosen:
            type_labels[t] = f"domain_{word}"
        print(f"  domain_{word}: {len(group)}개 중 {n}개 샘플링")

    n = min(n_per_type, len(cluster))
    chosen = random.sample(cluster, n)
    sampled.extend(chosen)
    for t in chosen:
        type_labels[t] = "cluster"
    print(f"  cluster: {len(cluster)}개 중 {n}개 샘플링")

    return sampled, type_labels

hub_sample_tags, hub_type_labels = sample_hubs_by_type(hub_tags_all, n_per_type=N_PER_TYPE)

manifest = {
    "gallery_image_ids": [r["id"] for r in gallery_sample],
    "gallery_filenames": [r["filename"] for r in gallery_sample],
    "query_ids": [r["query_id"] for r in query_sample],
    "query_captions": [r["caption"] for r in query_sample],
    "hub_tags": hub_sample_tags,
    "hub_type_labels": hub_type_labels,
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"샘플 고정: gallery {N_GALLERY}, query {N_QUERIES}, hub {len(hub_sample_tags)}")

def save_model_embeddings(tag, gallery_emb, query_emb, hub_emb):
    np.save(os.path.join(OUT_DIR, f"{tag}_gallery.npy"), gallery_emb)
    np.save(os.path.join(OUT_DIR, f"{tag}_query.npy"), query_emb)
    np.save(os.path.join(OUT_DIR, f"{tag}_hub.npy"), hub_emb)
    print(f"[{tag}] gallery={gallery_emb.shape} query={query_emb.shape} hub={hub_emb.shape}")

gallery_paths = [os.path.join(IMG_DIR, fn) for fn in manifest["gallery_filenames"]]
hub_paths = [os.path.join(RESULTS_DIR, f"hub_{t}.png") for t in manifest["hub_tags"]]

# ---- 1. clip_vitb32_base_openai ----
print("\n=== clip_vitb32_base_openai ===")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_encode_images(paths, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="clip_vitb32_base_openai 이미지 인코딩"):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        inputs = clip_processor(images=batch, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = clip_model.get_image_features(**inputs, return_dict=True).pooler_output
            out = out / out.norm(p=2, dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype("float32")

def clip_encode_texts(texts, batch_size=128):
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="clip_vitb32_base_openai 텍스트 인코딩"):
        inputs = clip_processor(text=texts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = clip_model.get_text_features(**inputs, return_dict=True).pooler_output
            out = out / out.norm(p=2, dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype("float32")

clip_gallery_emb = clip_encode_images(gallery_paths)
clip_query_emb = clip_encode_texts(manifest["query_captions"])
clip_hub_emb = clip_encode_images(hub_paths)
save_model_embeddings("clip_vitb32_base_openai", clip_gallery_emb, clip_query_emb, clip_hub_emb)

del clip_model, clip_processor
torch.cuda.empty_cache()

# ---- 2. openclip_vitb32_base_laion2b (surrogate 자기 자신, 기준점) ----
print("\n=== openclip_vitb32_base_laion2b (surrogate) ===")
laion_model, _, laion_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
laion_tokenizer = open_clip.get_tokenizer("ViT-B-32")
laion_model = laion_model.to(DEVICE).eval()

def laion_encode_images(paths, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="openclip_vitb32_base_laion2b 이미지 인코딩"):
        batch = [laion_preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]
        batch_tensor = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            out = laion_model.encode_image(batch_tensor)
            out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype("float32")

def laion_encode_texts(texts, batch_size=128):
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="openclip_vitb32_base_laion2b 텍스트 인코딩"):
        tokens = laion_tokenizer(texts[i:i+batch_size]).to(DEVICE)
        with torch.no_grad():
            out = laion_model.encode_text(tokens)
            out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype("float32")

laion_gallery_emb = laion_encode_images(gallery_paths)
laion_query_emb = laion_encode_texts(manifest["query_captions"])
laion_hub_emb = laion_encode_images(hub_paths)
save_model_embeddings("openclip_vitb32_base_laion2b", laion_gallery_emb, laion_query_emb, laion_hub_emb)

del laion_model
torch.cuda.empty_cache()

# ---- 3. 추가 OpenCLIP 모델들 (같은 패턴 재사용) ----
EXTRA_MODELS = [
    ("openclip_vitb16_base_laion2b", "ViT-B-16", "laion2b_s34b_b88k"),
    ("openclip_vitl14_large_laion2b", "ViT-L-14", "laion2b_s32b_b82k"),
    ("openclip_vitb32_base_laion400m", "ViT-B-32", "laion400m_e32"),
    ("openclip_vith14_huge_laion2b", "ViT-H-14", "laion2b_s32b_b79k"),
]

for tag, arch, pretrained in EXTRA_MODELS:
    print(f"\n=== {tag} ({arch}, {pretrained}) ===")
    try:
        m, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    except Exception as e:
        print(f"  {tag} 로딩 실패, 건너뜀: {e}")
        continue
    tok = open_clip.get_tokenizer(arch)
    m = m.to(DEVICE).eval()

    def enc_img(paths, batch_size=32, _m=m, _p=preprocess, _tag=tag):
        feats = []
        for i in tqdm(range(0, len(paths), batch_size), desc=f"{_tag} 이미지 인코딩"):
            batch = [_p(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]
            bt = torch.stack(batch).to(DEVICE)
            with torch.no_grad():
                out = _m.encode_image(bt)
                out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0).astype("float32")

    def enc_txt(texts, batch_size=128, _m=m, _tok=tok, _tag=tag):
        feats = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"{_tag} 텍스트 인코딩"):
            tokens = _tok(texts[i:i+batch_size]).to(DEVICE)
            with torch.no_grad():
                out = _m.encode_text(tokens)
                out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0).astype("float32")

    g_emb = enc_img(gallery_paths)
    q_emb = enc_txt(manifest["query_captions"])
    h_emb = enc_img(hub_paths)
    save_model_embeddings(tag, g_emb, q_emb, h_emb)

    del m
    torch.cuda.empty_cache()

# ---- 4. openclip_rn50_base_openai ----
print("\n=== openclip_rn50_base_openai ===")
rn50_model, _, rn50_preprocess = open_clip.create_model_and_transforms("RN50", pretrained="openai")
rn50_tokenizer = open_clip.get_tokenizer("RN50")
rn50_model = rn50_model.to(DEVICE).eval()

def rn50_encode_images(paths, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="openclip_rn50_base_openai 이미지 인코딩"):
        batch = [rn50_preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]
        batch_tensor = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            out = rn50_model.encode_image(batch_tensor)
            out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype("float32")

def rn50_encode_texts(texts, batch_size=128):
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="openclip_rn50_base_openai 텍스트 인코딩"):
        tokens = rn50_tokenizer(texts[i:i+batch_size]).to(DEVICE)
        with torch.no_grad():
            out = rn50_model.encode_text(tokens)
            out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype("float32")

rn50_gallery_emb = rn50_encode_images(gallery_paths)
rn50_query_emb = rn50_encode_texts(manifest["query_captions"])
rn50_hub_emb = rn50_encode_images(hub_paths)
save_model_embeddings("openclip_rn50_base_openai", rn50_gallery_emb, rn50_query_emb, rn50_hub_emb)

elapsed = time.time() - SCRIPT_START
print(f"\n총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
print(f"manifest 및 임베딩 -> {OUT_DIR}")

os._exit(0)

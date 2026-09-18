import os
import sys
import json
import re
import time
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans

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
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/imagebind/hubs")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 16 / 255
STEPS = 300          # CLIP(600)보다 축소 -- 시간 예산 고려
ALPHA = 3 / 255       # step 수를 줄인 만큼 alpha를 살짝 키움
N_TARGET_QUERIES = 100
N_REPS = 10           # 조건당 반복 횟수 (원래 30 -> 10으로 축소)
WORDS = ["dog", "bicycle", "pizza", "cat", "elephant", "umbrella",
         "skateboard", "surfboard", "motorcycle", "kite"]
N_CLUSTERS_TOTAL = 1000
N_CLUSTERS_PICK = 5
SEED = 0

random.seed(SEED)

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
mean_t = torch.tensor(MEAN).view(3, 1, 1).to(DEVICE)
std_t = torch.tensor(STD).view(3, 1, 1).to(DEVICE)
to_tensor = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

print("모델 로딩...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval().to(DEVICE)
for p in model.parameters():
    p.requires_grad_(False)

with open(DATA_META_PATH, "r", encoding="utf-8") as f:
    data_records = json.load(f)
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)

def get_local_image(image_id):
    rec = data_records[image_id % len(data_records)]
    return Image.open(os.path.join(IMG_DIR, rec["filename"])).convert("RGB")

def embed_texts(texts, batch_size=64, desc=None):
    feats = []
    rng = range(0, len(texts), batch_size)
    for i in rng:
        batch = texts[i:i+batch_size]
        inputs = {ModalityType.TEXT: data.load_and_transform_text(batch, DEVICE)}
        with torch.no_grad():
            out = model(inputs)[ModalityType.TEXT]
            out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
        if desc and (i // batch_size) % 50 == 0:
            print(f"  {desc} {i}/{len(texts)}")
    return np.concatenate(feats, axis=0)

def compute_centroid(qt_captions):
    feats = embed_texts(qt_captions)
    c = feats.mean(axis=0, keepdims=True)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return torch.tensor(c, dtype=torch.float32).to(DEVICE)

def run_pgd(c_t, gc_idx):
    pil_img = get_local_image(gc_idx)
    x_clean = to_tensor(pil_img).to(DEVICE)
    x_adv = x_clean.clone().detach().requires_grad_(True)
    for step in range(STEPS):
        normalized = (x_adv - mean_t) / std_t
        out = model({ModalityType.VISION: normalized.unsqueeze(0)})[ModalityType.VISION]
        out = out / out.norm(dim=-1, keepdim=True)
        cos_sim = (out @ c_t.T).squeeze()
        loss = -cos_sim
        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        with torch.no_grad():
            x_adv -= ALPHA * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x_clean - EPS, x_clean + EPS)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = x_adv.detach().requires_grad_(True)
    with torch.no_grad():
        normalized = (x_adv - mean_t) / std_t
        final = model({ModalityType.VISION: normalized.unsqueeze(0)})[ModalityType.VISION]
        final = final / final.norm(dim=-1, keepdim=True)
        final_cos = (final @ c_t.T).item()
    return x_adv.detach().cpu(), final.cpu().numpy().astype("float32"), final_cos

def save_hub(tag, x_adv_cpu, final_feat, final_cos, gc_idx, qt_query_ids, mode, keyword, cluster_id):
    out_img = transforms.ToPILImage()(x_adv_cpu.clamp(0, 1))
    out_img.save(os.path.join(RESULTS_DIR, f"hub_{tag}.png"))
    np.save(os.path.join(RESULTS_DIR, f"hub_{tag}_embedding.npy"), final_feat)
    meta = {
        "mode": mode, "keyword": keyword, "cluster_id": cluster_id,
        "gc_index": gc_idx, "qt_query_ids": qt_query_ids, "n_qt": len(qt_query_ids),
        "eps": EPS, "steps": STEPS, "alpha": ALPHA, "surrogate": "imagebind",
        "final_cos_sim": final_cos,
    }
    with open(os.path.join(RESULTS_DIR, f"hub_{tag}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ---- 클러스터 준비 (크기 제한 없음) ----
CLUSTER_CACHE = os.path.join(EMB_DIR, f"cluster_cache_imagebind_k{N_CLUSTERS_TOTAL}.npz")
if os.path.exists(CLUSTER_CACHE):
    print("캐시된 클러스터링 로드")
    cluster_labels = np.load(CLUSTER_CACHE)["labels"]
else:
    print("전체 쿼리 25,000개 임베딩 중 (ImageBind 텍스트 인코더)...")
    all_captions = [r["caption"] for r in query_records]
    all_q_emb = embed_texts(all_captions, batch_size=128, desc="클러스터링용 쿼리 임베딩")
    print(f"MiniBatchKMeans (K={N_CLUSTERS_TOTAL})...")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS_TOTAL, random_state=0, batch_size=1024, n_init=10).fit(all_q_emb)
    cluster_labels = kmeans.labels_
    np.savez(CLUSTER_CACHE, labels=cluster_labels)

sizes = np.bincount(cluster_labels, minlength=N_CLUSTERS_TOTAL)
# 크기 제한(15~80) 없이, 너무 작은(<3) 것만 제외하고 무작위로 선택
candidates = [c for c in range(N_CLUSTERS_TOTAL) if sizes[c] >= 3]
chosen_clusters = random.sample(candidates, min(N_CLUSTERS_PICK, len(candidates)))
print(f"선택된 클러스터(크기 제한 없음): {chosen_clusters}, 크기={[int(sizes[c]) for c in chosen_clusters]}")

def build_qt_for_cluster(cluster_id):
    idxs = np.where(cluster_labels == cluster_id)[0]
    return [query_records[i] for i in idxs]

# ---- 작업 목록 ----
jobs = []
for w in WORDS:
    jobs.append((f"domain_{w}", "domain", w, None))
for c in chosen_clusters:
    jobs.append((f"cluster{c}", "cluster", None, c))

total_planned = len(jobs) * N_REPS
print(f"\n총 {len(jobs)}개 조건 x {N_REPS}회 = {total_planned}개 hub 생성 예정")
print(f"예상 시간(순수 PGD 기준): 약 {total_planned * (STEPS/300*70) / 60:.0f}분\n")

done = 0
for base_tag, mode, keyword, cluster_id in jobs:
    if mode == "domain":
        pool = [r for r in query_records if keyword.lower() in r["caption"].lower()]
        if not pool:
            print(f"'{keyword}' 관련 caption 없음, 건너뜀")
            continue
    else:
        qt_fixed = build_qt_for_cluster(cluster_id)

    for rep in range(N_REPS):
        t0 = time.time()
        if mode == "domain":
            n = min(N_TARGET_QUERIES, len(pool))
            qt_records = random.sample(pool, n)
        else:
            qt_records = qt_fixed

        qt_query_ids = [r["query_id"] for r in qt_records]
        c_t = compute_centroid([r["caption"] for r in qt_records])

        gc_idx = random.randint(0, len(data_records) - 1)
        tag = f"{base_tag}_gc{gc_idx}_r{rep}"

        x_adv_cpu, final_feat, final_cos = run_pgd(c_t, gc_idx)
        save_hub(tag, x_adv_cpu, final_feat, final_cos, gc_idx, qt_query_ids, mode, keyword, cluster_id)

        done += 1
        elapsed = time.time() - SCRIPT_START
        avg = elapsed / done
        remaining = (total_planned - done) * avg
        print(f"[{done}/{total_planned}] {tag} cos_sim={final_cos:.3f} "
              f"({time.time()-t0:.1f}초) | 누적 {elapsed/60:.1f}분 | 예상잔여 {remaining/60:.1f}분")

print(f"\n=== ImageBind hub 생성 완료 ===")
print(f"총 소요 시간: {(time.time()-SCRIPT_START)/60:.1f}분")
os._exit(0)

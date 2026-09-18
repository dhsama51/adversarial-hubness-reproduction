import os
import re
import json
import glob
import time
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

SCRIPT_START = time.time()

DATA_DIR = os.path.expanduser("~/projects/hubness/data")
IMG_DIR = os.path.join(DATA_DIR, "images")
DATA_META_PATH = os.path.join(DATA_DIR, "metadata.json")
EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip/hubs")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPS = 16 / 255
STEPS = 600
ALPHA = 2 / 255
N_TARGET_QUERIES = 100
N_REPS = 30
N_CLUSTERS_TOTAL = 1000
N_CLUSTERS_PICK = 5
WORDS = ["dog", "bicycle", "pizza", "cat", "elephant", "umbrella",
         "skateboard", "surfboard", "motorcycle", "kite"]

random.seed(0)

with open(DATA_META_PATH, "r", encoding="utf-8") as f:
    data_records = json.load(f)
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    query_records = json.load(f)

print("모델 로딩 중...")
t0 = time.time()
model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
for p in model.parameters():
    p.requires_grad_(False)
print(f"모델 로딩 완료: {time.time() - t0:.1f}초\n")

mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1).to(DEVICE)
std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1).to(DEVICE)
_dummy = Image.new("RGB", (256, 256))
_sample = processor(images=_dummy, return_tensors="pt")
_, _, H, W = _sample["pixel_values"].shape
size_hw = (H, W)
to_tensor = transforms.Compose([
    transforms.Resize(size_hw),
    transforms.CenterCrop(size_hw),
    transforms.ToTensor(),
])

def embed_texts(texts, batch_size=128, desc=None):
    feats_list = []
    it = range(0, len(texts), batch_size)
    if desc:
        it = tqdm(it, desc=desc)
    for i in it:
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs, return_dict=True)
            feats = outputs.pooler_output
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        feats_list.append(feats.cpu().numpy())
    return np.concatenate(feats_list, axis=0)

def get_local_image(image_id):
    rec = data_records[image_id % len(data_records)]
    return Image.open(os.path.join(IMG_DIR, rec["filename"])).convert("RGB")

def compute_centroid(qt_records):
    texts = [r["caption"] for r in qt_records]
    feats = embed_texts(texts)
    c = feats.mean(axis=0, keepdims=True)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return torch.tensor(c, dtype=torch.float32).to(DEVICE)

def run_pgd(c_t, gc_idx):
    pil_img = get_local_image(gc_idx)
    x_clean = to_tensor(pil_img).to(DEVICE)
    x_adv = x_clean.clone().detach().requires_grad_(True)
    for step in range(STEPS):
        normalized = (x_adv - mean) / std
        outputs = model.get_image_features(pixel_values=normalized.unsqueeze(0), return_dict=True)
        feats = outputs.pooler_output
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        cos_sim = (feats @ c_t.T).squeeze()
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
        normalized = (x_adv - mean) / std
        outputs = model.get_image_features(pixel_values=normalized.unsqueeze(0), return_dict=True)
        final_feat = outputs.pooler_output
        final_feat = final_feat / final_feat.norm(p=2, dim=-1, keepdim=True)
        final_cos = (final_feat @ c_t.T).item()
    return x_adv.detach().cpu(), final_feat.cpu().numpy().astype("float32"), final_cos

def save_hub(tag, x_adv_cpu, final_feat, final_cos, gc_idx, qt_query_ids, mode, keyword, cluster_id):
    out_img = transforms.ToPILImage()(x_adv_cpu.clamp(0, 1))
    out_img.save(os.path.join(RESULTS_DIR, f"hub_{tag}.png"))
    np.save(os.path.join(RESULTS_DIR, f"hub_{tag}_embedding.npy"), final_feat)
    meta = {
        "mode": mode, "keyword": keyword, "cluster_id": cluster_id,
        "gc_index": gc_idx, "qt_query_ids": qt_query_ids, "n_qt": len(qt_query_ids),
        "eps": EPS, "steps": STEPS, "alpha": ALPHA, "final_cos_sim": final_cos,
    }
    with open(os.path.join(RESULTS_DIR, f"hub_{tag}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def count_existing(base_tag):
    return glob.glob(os.path.join(RESULTS_DIR, f"hub_{base_tag}_gc*_meta.json")) + \
           glob.glob(os.path.join(RESULTS_DIR, f"hub_{base_tag}_meta.json"))

def existing_gc_indices(base_tag):
    idxs = set()
    for f in glob.glob(os.path.join(RESULTS_DIR, f"hub_{base_tag}_gc*_meta.json")):
        m = re.search(r"_gc(\d+)_meta\.json$", f)
        if m:
            idxs.add(int(m.group(1)))
    return idxs

def next_gc_index(used_gc, start=0):
    idx = start
    while idx in used_gc:
        idx += 1
    used_gc.add(idx)
    return idx

# ---- 조건(condition) 목록 구성 ----
jobs = [("universal", "universal", None, None)]
for w in WORDS:
    jobs.append((f"domain_{w}", "domain", w, None))

CLUSTER_CACHE_PATH = os.path.join(EMB_DIR, f"cluster_cache_k{N_CLUSTERS_TOTAL}.npz")
if os.path.exists(CLUSTER_CACHE_PATH):
    print("캐시된 클러스터링 결과 로드")
    cluster_labels = np.load(CLUSTER_CACHE_PATH)["labels"]
else:
    print(f"전체 쿼리 임베딩 + MiniBatchKMeans(K={N_CLUSTERS_TOTAL})...")
    all_captions = [r["caption"] for r in query_records]
    all_q_emb = embed_texts(all_captions, desc="전체 쿼리 임베딩")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS_TOTAL, random_state=0, batch_size=1024, n_init=10).fit(all_q_emb)
    cluster_labels = kmeans.labels_
    np.savez(CLUSTER_CACHE_PATH, labels=cluster_labels)
    print("클러스터링 결과 캐싱 완료")

sizes = np.bincount(cluster_labels, minlength=N_CLUSTERS_TOTAL)
candidates = [c for c in range(N_CLUSTERS_TOTAL) if 15 <= sizes[c] <= 80]
candidates.sort(key=lambda c: abs(sizes[c] - 25))
chosen_clusters = candidates[:N_CLUSTERS_PICK]
print(f"선택된 클러스터: {chosen_clusters} (크기: {[int(sizes[c]) for c in chosen_clusters]})\n")

for c in chosen_clusters:
    jobs.append((f"cluster{c}", "cluster", None, c))

def build_qt_for_cluster(cluster_id):
    member_idxs = np.where(cluster_labels == cluster_id)[0]
    return [query_records[i] for i in member_idxs]

# ---- 재사용 가능한 기존 결과 반영해 작업량 산정 ----
plan = []
total_planned = 0
print(f"=== 조건별 계획 (목표 {N_REPS}회/조건, 총 {len(jobs)}개 조건) ===")
for base_tag, mode, keyword, cluster_id in jobs:
    n_existing = len(count_existing(base_tag))
    n_needed = max(0, N_REPS - n_existing)
    plan.append((base_tag, mode, keyword, cluster_id, n_needed))
    total_planned += n_needed
    print(f"  [{base_tag}] 기존 {n_existing}개 재사용, 신규 {n_needed}개 생성 예정")

print(f"\n신규 생성 총 {total_planned}회 (PGD만 기준 예상 약 {total_planned*17/60:.1f}분)\n")

# ---- 배치 실행 ----
done_count = 0
used_gc_by_tag = {}

for base_tag, mode, keyword, cluster_id, n_needed in plan:
    if n_needed == 0:
        continue

    if mode == "domain":
        pool = [r for r in query_records if keyword.lower() in r["caption"].lower()]
        assert len(pool) > 0, f"'{keyword}' 포함 caption 없음"
    elif mode == "cluster":
        qt_records_fixed = build_qt_for_cluster(cluster_id)

    used_gc = used_gc_by_tag.setdefault(base_tag, existing_gc_indices(base_tag))

    print(f"\n--- [{base_tag}] {n_needed}회 생성 시작 ---")
    for rep in range(n_needed):
        rep_start = time.time()

        if mode == "universal":
            qt_records = random.sample(query_records, min(N_TARGET_QUERIES, len(query_records)))
        elif mode == "domain":
            n = min(N_TARGET_QUERIES, len(pool))
            qt_records = random.sample(pool, n)
        else:
            qt_records = qt_records_fixed

        qt_query_ids = [r["query_id"] for r in qt_records]
        c_t = compute_centroid(qt_records)

        gc_idx = next_gc_index(used_gc, start=0)
        tag = f"{base_tag}_gc{gc_idx}"

        x_adv_cpu, final_feat, final_cos = run_pgd(c_t, gc_idx)
        save_hub(tag, x_adv_cpu, final_feat, final_cos, gc_idx, qt_query_ids, mode, keyword, cluster_id)

        done_count += 1
        elapsed_total = time.time() - SCRIPT_START
        avg_per_job = elapsed_total / done_count
        remaining = total_planned - done_count
        eta = remaining * avg_per_job

        print(f"[{done_count}/{total_planned}] {tag} 완료 "
              f"(cos_sim={final_cos:.3f}, {time.time()-rep_start:.1f}초) | "
              f"누적 {elapsed_total/60:.1f}분 | 예상 잔여 {eta/60:.1f}분")

print(f"\n=== 배치 생성 완료 ===")
print(f"총 소요 시간: {(time.time()-SCRIPT_START)/60:.1f}분")

os._exit(0)
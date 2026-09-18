import os
import json
import time
import random
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import open_clip
from tqdm import tqdm

SCRIPT_START = time.time()

DATA_DIR = os.path.expanduser("~/projects/hubness/data")
IMG_DIR = os.path.join(DATA_DIR, "images")
DATA_META_PATH = os.path.join(DATA_DIR, "metadata.json")
EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
QUERIES_PATH = os.path.join(EMB_DIR, "queries.json")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/laion_clip/hubs")  # 기존 results/와 분리
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ARCH = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPS = 16 / 255
STEPS = 600
ALPHA = 2 / 255
N_TARGET_QUERIES = 100
SEED = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["universal", "domain", "cluster"], required=True)
    p.add_argument("--keyword", type=str, default=None)
    p.add_argument("--gc_index", type=int, default=None)
    p.add_argument("--n_clusters", type=int, default=1000)
    p.add_argument("--cluster_id", type=int, default=None)
    return p.parse_args()

def get_image_by_index(image_id, data_records):
    rec = next(r for r in data_records if r["id"] == image_id)
    return Image.open(os.path.join(IMG_DIR, rec["filename"])).convert("RGB")

def main():
    args = parse_args()
    random.seed(SEED)

    with open(DATA_META_PATH, "r", encoding="utf-8") as f:
        data_records = json.load(f)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        query_records = json.load(f)

    t_model_start = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ARCH, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_ARCH)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"모델 로딩: {time.time() - t_model_start:.1f}초")

    if args.mode == "universal":
        qt_records = random.sample(query_records, N_TARGET_QUERIES)
        base_tag = "universal"

    elif args.mode == "domain":
        assert args.keyword, "--mode domain 이면 --keyword 필수"
        pool = [r for r in query_records if args.keyword.lower() in r["caption"].lower()]
        assert len(pool) > 0, f"'{args.keyword}' 포함 caption이 없습니다"
        n = min(N_TARGET_QUERIES, len(pool))
        qt_records = random.sample(pool, n)
        base_tag = f"domain_{args.keyword}"
        print(f"'{args.keyword}' 관련 caption {len(pool)}개 중 {n}개를 Qt로 사용")

    else:  # cluster
        from sklearn.cluster import MiniBatchKMeans

        CLUSTER_CACHE_PATH = os.path.join(EMB_DIR, f"cluster_cache_laion_k{args.n_clusters}.npz")
        if os.path.exists(CLUSTER_CACHE_PATH):
            print(f"캐시된 클러스터링 결과 로드: {CLUSTER_CACHE_PATH}")
            cluster_labels = np.load(CLUSTER_CACHE_PATH)["labels"]
        else:
            all_captions = [r["caption"] for r in query_records]
            all_feats_list = []
            for i in tqdm(range(0, len(all_captions), 128), desc="전체 쿼리 임베딩"):
                batch = all_captions[i:i + 128]
                tokens = tokenizer(batch).to(DEVICE)
                with torch.no_grad():
                    feats = model.encode_text(tokens)
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                all_feats_list.append(feats.cpu().numpy())
            all_q_emb = np.concatenate(all_feats_list, axis=0)

            print(f"MiniBatchKMeans 클러스터링 (K={args.n_clusters})...")
            kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=0, batch_size=1024, n_init=10).fit(all_q_emb)
            cluster_labels = kmeans.labels_
            np.savez(CLUSTER_CACHE_PATH, labels=cluster_labels)
            print(f"클러스터링 결과 캐싱 완료 -> {CLUSTER_CACHE_PATH}")

        if args.cluster_id is not None:
            chosen_cluster = args.cluster_id
        else:
            sizes = np.bincount(cluster_labels, minlength=args.n_clusters)
            candidates = [c for c in range(args.n_clusters) if 15 <= sizes[c] <= 80]
            assert candidates, "적당한 크기의 클러스터가 없습니다."
            chosen_cluster = min(candidates, key=lambda c: abs(sizes[c] - 25))

        member_idxs = np.where(cluster_labels == chosen_cluster)[0]
        qt_records = [query_records[i] for i in member_idxs]
        base_tag = f"cluster{chosen_cluster}"
        print(f"클러스터 {chosen_cluster} 크기: {len(qt_records)}개")

    qt_captions = [r["caption"] for r in qt_records]
    qt_query_ids = [r["query_id"] for r in qt_records]
    print(f"[{base_tag}] Qt 크기: {len(qt_captions)}")

    with torch.no_grad():
        tokens = tokenizer(qt_captions).to(DEVICE)
        q_feats = model.encode_text(tokens)
        q_feats = q_feats / q_feats.norm(p=2, dim=-1, keepdim=True)
        c_t = q_feats.mean(dim=0, keepdim=True)
        c_t = c_t / c_t.norm(p=2, dim=-1, keepdim=True)

    gc_idx = args.gc_index if args.gc_index is not None else random.randint(0, len(data_records) - 1)
    print(f"초기 이미지(g_c) id: {gc_idx}")
    tag = f"{base_tag}_gc{gc_idx}"

    pil_img = get_image_by_index(gc_idx, data_records)

    # open_clip preprocess에서 mean/std, 입력 크기 추출
    mean = torch.tensor(preprocess.transforms[-1].mean).view(3, 1, 1).to(DEVICE)
    std = torch.tensor(preprocess.transforms[-1].std).view(3, 1, 1).to(DEVICE)
    size_hw = preprocess.transforms[0].size
    if isinstance(size_hw, int):
        size_hw = (size_hw, size_hw)

    to_tensor = transforms.Compose([
        transforms.Resize(size_hw),
        transforms.CenterCrop(size_hw),
        transforms.ToTensor(),
    ])
    x_clean = to_tensor(pil_img).to(DEVICE)
    x_adv = x_clean.clone().detach().requires_grad_(True)

    t_pgd_start = time.time()
    pbar = tqdm(range(STEPS), desc="PGD 공격 진행")
    for step in pbar:
        normalized = (x_adv - mean) / std
        feats = model.encode_image(normalized.unsqueeze(0))
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

        pbar.set_postfix(cos_sim=f"{cos_sim.item():.4f}")

    print(f"PGD 최적화: {time.time() - t_pgd_start:.1f}초")

    with torch.no_grad():
        normalized = (x_adv - mean) / std
        final_feat = model.encode_image(normalized.unsqueeze(0))
        final_feat = final_feat / final_feat.norm(p=2, dim=-1, keepdim=True)
        final_cos = (final_feat @ c_t.T).item()

    print(f"최종 cos_sim(g_a, c_t) = {final_cos:.4f}")

    out_img = transforms.ToPILImage()(x_adv.detach().cpu().clamp(0, 1))
    out_img.save(os.path.join(RESULTS_DIR, f"hub_{tag}.png"))
    np.save(os.path.join(RESULTS_DIR, f"hub_{tag}_embedding.npy"), final_feat.cpu().numpy().astype("float32"))

    meta = {
        "mode": args.mode, "keyword": args.keyword, "gc_index": gc_idx,
        "qt_query_ids": qt_query_ids, "n_qt": len(qt_query_ids),
        "eps": EPS, "steps": STEPS, "alpha": ALPHA, "final_cos_sim": final_cos,
        "surrogate": f"{MODEL_ARCH}_{PRETRAINED}",
    }
    with open(os.path.join(RESULTS_DIR, f"hub_{tag}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - SCRIPT_START
    print(f"총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
    os._exit(0)

if __name__ == "__main__":
    main()
import os
import json
import glob
import random
import numpy as np

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/openai_clip")
N_REPEATS = 8
HUB_TARGETS = [10, 100]
SEEDS = list(range(100, 100 + N_REPEATS))

gallery_emb = np.load(os.path.join(EMB_DIR, "image_embeddings.npy")).astype("float32")
base_n = gallery_emb.shape[0]

meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hub_*_meta.json")))
hub_tags_all = [os.path.basename(f).replace("hub_", "").replace("_meta.json", "") for f in meta_files]
hub_tags_all = [t for t in hub_tags_all if os.path.exists(os.path.join(RESULTS_DIR, f"hub_{t}_embedding.npy"))]

with open(os.path.join(EMB_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    gallery_records = json.load(f)

for n_hubs_target in HUB_TARGETS:
  OUT_BASE = os.path.expanduser(f"~/projects/hubness/results/openai_clip/ahd_input_repeat_{n_hubs_target}")
  for rep, seed in enumerate(SEEDS):
    random.seed(seed + n_hubs_target)  # 두 비율끼리 겹치지 않게
    out_dir = os.path.join(OUT_BASE, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    hub_sample_tags = random.sample(hub_tags_all, n_hubs_target)
    hub_embs = [np.load(os.path.join(RESULTS_DIR, f"hub_{t}_embedding.npy")).astype("float32") for t in hub_sample_tags]
    hub_emb_matrix = np.concatenate(hub_embs, axis=0)
    combined_emb = np.concatenate([gallery_emb, hub_emb_matrix], axis=0).astype("float32")
    np.save(os.path.join(out_dir, "combined_embeddings.npy"), combined_emb)

    combined_records = []
    for r in gallery_records:
        combined_records.append({"doc_id": f"gallery_{r['id']}", "caption": r["caption"], "is_hub": False, "hub_tag": None})
    for tag in hub_sample_tags:
        combined_records.append({"doc_id": f"hub_{tag}", "caption": None, "is_hub": True, "hub_tag": tag})
    with open(os.path.join(out_dir, "combined_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(combined_records, f, ensure_ascii=False, indent=2)

    print(f"[{n_hubs_target}개/{rep+1}/{N_REPEATS}] seed={seed} hub_tags={hub_sample_tags}")

print(f"\n{N_REPEATS}개 반복 x {len(HUB_TARGETS)}개 비율 데이터 준비 완료")
os._exit(0)

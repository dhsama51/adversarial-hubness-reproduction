import os
import json
import glob
import numpy as np

EMB_DIR = os.path.expanduser("~/projects/hubness/embeddings")
RESULTS_DIR = os.path.expanduser("~/projects/hubness/results/imagebind")
OUT_DIR = os.path.expanduser("~/projects/hubness/results/imagebind/ahd_input")
os.makedirs(OUT_DIR, exist_ok=True)

gallery_emb = np.load(os.path.join(EMB_DIR, "image_embeddings_imagebind.npy")).astype("float32")
base_n = gallery_emb.shape[0]

meta_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hub_*_meta.json")))
hub_tags = [os.path.basename(f).replace("hub_", "").replace("_meta.json", "") for f in meta_files]
hub_tags = [t for t in hub_tags if os.path.exists(os.path.join(RESULTS_DIR, f"hub_{t}_embedding.npy"))]

print(f"gallery {base_n}개 + ImageBind hub {len(hub_tags)}개")

hub_embs = [np.load(os.path.join(RESULTS_DIR, f"hub_{t}_embedding.npy")).astype("float32") for t in hub_tags]
hub_emb_matrix = np.concatenate(hub_embs, axis=0)
combined_emb = np.concatenate([gallery_emb, hub_emb_matrix], axis=0).astype("float32")
np.save(os.path.join(OUT_DIR, "combined_embeddings.npy"), combined_emb)

with open(os.path.join(EMB_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    gallery_records = json.load(f)

combined_records = []
for r in gallery_records:
    combined_records.append({"doc_id": f"gallery_{r['id']}", "caption": r["caption"], "is_hub": False, "hub_tag": None})
for tag in hub_tags:
    combined_records.append({"doc_id": f"hub_{tag}", "caption": None, "is_hub": True, "hub_tag": tag})
with open(os.path.join(OUT_DIR, "combined_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(combined_records, f, ensure_ascii=False, indent=2)

n_total = base_n + len(hub_tags)
print(f"저장 완료: {n_total}개 (hub 비율 {len(hub_tags)/n_total*100:.2f}%)")
print(f"-> {OUT_DIR}")
os._exit(0)

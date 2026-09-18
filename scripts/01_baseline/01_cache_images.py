import os
import json
import time
import itertools
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_START = time.time()

DATA_DIR = os.path.expanduser("~/projects/hubness/data")
IMG_DIR = os.path.join(DATA_DIR, "images")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
os.makedirs(IMG_DIR, exist_ok=True)

DATASET_NAME = "lmms-lab/COCO-Caption2017"
SPLIT = "val"
N_IMAGES = 5000

print("스트리밍 데이터셋 여는 중...")
ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)
stream_iter = iter(ds)

first_example = next(stream_iter)
img_col, cap_col = "image", "answer"

full_stream = itertools.chain([first_example], stream_iter)
limited_stream = itertools.islice(full_stream, N_IMAGES)

records = []
for i, ex in enumerate(tqdm(limited_stream, total=N_IMAGES, desc="이미지 캐싱")):
    img = ex[img_col].convert("RGB")
    fname = f"img_{i:04d}.jpg"
    img.save(os.path.join(IMG_DIR, fname), quality=95)

    raw_captions = ex[cap_col]
    captions = raw_captions if isinstance(raw_captions, list) else [raw_captions]
    records.append({"id": i, "filename": fname, "captions": captions})

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

n_total_captions = sum(len(r["captions"]) for r in records)
print(f"완료: 이미지 {len(records)}장, caption 총 {n_total_captions}개 -> {IMG_DIR}")
print(f"메타데이터 -> {META_PATH}")

elapsed = time.time() - SCRIPT_START
print(f"총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")

os._exit(0)
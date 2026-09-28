"""Phase 0 gate facts on the uploaded dataset: native resolution (gates 03/06), classes, split sizes,
and source-frame duplication (Roboflow names augmented copies `<source>.rf.<hash>.<ext>`)."""

import collections
import pathlib
import sys

import yaml
from PIL import Image

sys.path.insert(0, "/content/ballhawk")
import ballhawk_common as bc

data_yaml = bc.prepare_data()
cfg = yaml.safe_load(pathlib.Path(data_yaml).read_text())
print("classes:", cfg["names"])

sources = {}
for split in ("train", "val", "test"):
    img_dir = pathlib.Path(cfg[split])
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sizes = collections.Counter(Image.open(p).size for p in images)
    counts = collections.Counter()
    for lbl in (img_dir.parent / "labels").glob("*.txt"):
        counts.update(int(line.split()[0]) for line in lbl.read_text().splitlines() if line.strip())
    per_class = {cfg["names"][k]: v for k, v in sorted(counts.items())}
    sources[split] = {p.name.split(".rf.")[0] for p in images}
    print(f"{split}: {len(images)} images from {len(sources[split])} source frames, "
          f"sizes {sizes.most_common(3)}, instances {per_class}")

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    print(f"source frames shared {a}/{b}: {len(sources[a] & sources[b])}")

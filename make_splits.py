"""Build a video-grouped train/val/test split for a Roboflow-export HF dataset and write it as JSON.

The mirrors' own splits put frames from the same clips in both train and test, which inflates
test metrics. Here every source video goes to exactly one split. Filenames are
`<video>_<clip>_<frame>_png.rf.<hash>.jpg`, so the video id is the text before the first `_`.
Run locally once; the JSON is the committed record the Colab build reads.

  python3 make_splits.py                       # detection data -> splits.json
  python3 make_splits.py --repo martinjolif/football-pitch-detection \\
      --revision 73488ede6158cbfe05aa7a8e471fc5041eceba8b --out splits_pitch.json \\
      --force-test 121364 744b27                 # keep Phase 3 click-frame videos out of training
"""

import argparse
import collections
import json
import random
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--repo", default="martinjolif/football-player-detection")
p.add_argument("--revision", default="e8b8cea002692efd74c945fcdad63e729adc5671")
p.add_argument("--out", default="splits.json")
p.add_argument("--force-test", nargs="*", default=[], help="videos that must be in test")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
TARGET = {"test": 0.15, "val": 0.15}  # share of images; train gets the rest

info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{a.repo}/revision/{a.revision}"))
images = [s["rfilename"] for s in info["siblings"] if "/images/" in s["rfilename"]]
per_video = collections.Counter(p.split("/")[-1].split("_")[0] for p in images)

split_videos = {"train": [], "val": [], "test": list(a.force_test)}
filled = {"test": sum(per_video[v] for v in a.force_test), "val": 0}
videos = sorted(set(per_video) - set(a.force_test))
random.Random(a.seed).shuffle(videos)
for v in videos:
    target = next((s for s in ("test", "val") if filled[s] < TARGET[s] * len(images)), "train")
    split_videos[target].append(v)
    if target != "train":
        filled[target] += per_video[v]

for s, vs in split_videos.items():
    print(f"{s}: {len(vs)} videos, {sum(per_video[v] for v in vs)} images")

with open(a.out, "w") as f:
    json.dump({"repo": a.repo, "revision": a.revision, "seed": a.seed, "force_test": a.force_test,
               "videos": {s: sorted(vs) for s, vs in split_videos.items()}}, f, indent=2)

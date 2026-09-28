"""BallHawk seed replication on Kaggle (2x T4): baseline, 04a (scale 0.2), 05b (cls_pw 0.5) x seeds 0, 1, 2.

Settles whether the +0.02 ball gains of 04a and 05b exceed seed noise. All nine runs happen here, seed 0
included, so platform differences (CUDA, drivers) do not mix into the seed noise; seed 0 doubles as a
platform check against the Colab numbers. Built by build.py, which embeds ballhawk_common.py and splits.json.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path("/kaggle/working")
CODE, OUT = ROOT / "code", ROOT / "seeds"
DATA, RAW = pathlib.Path("/kaggle/tmp/data"), pathlib.Path("/kaggle/tmp/raw")
EMBED = {{EMBED}}
CONFIGS = {"baseline": {}, "scale02": {"scale": 0.2}, "clsw05": {"cls_pw": 0.5}}
SEEDS = [0, 1, 2]
JOBS = [(c, s) for s in SEEDS for c in CONFIGS]  # interleaved so both GPUs get a mix


def load_bc():
    sys.path.insert(0, str(CODE))
    import ballhawk_common as bc
    bc.RUNS_DIR, bc.RESULTS_DIR = ROOT / "runs", OUT
    return bc


def ball(per_class):
    return next(c for c in per_class if c["class"] == "ball")


def worker(jobs):
    bc = load_bc()
    data_yaml = str(DATA / "data_fixed.yaml")
    for cfg, seed in jobs:
        name = f"{cfg}_s{seed}"
        best, save_dir, minutes, args = bc.train("yolo11s.pt", data_yaml, name, 640, False, None,
                                                 seed=seed, workers=2, **CONFIGS[cfg])
        val_o, val_pc = bc.evaluate(best, data_yaml, 640, "val")
        test_o, test_pc = bc.evaluate(best, data_yaml, 640, "test")
        rec = {"run": name, "config": cfg, "seed": seed, "overrides": CONFIGS[cfg], "train_minutes": minutes,
               "batch": args["batch"], "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "val": {"overall": val_o, "ball": ball(val_pc)}, "test": {"overall": test_o, "ball": ball(test_pc)}}
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2))
        for w in (save_dir / "weights").glob("*.pt"):
            w.unlink()  # keep the kernel output small; the metrics are what this run is for
        print(f"[gpu {rec['gpu']}] {name}: val ball {rec['val']['ball']['mAP50_95']:.4f} "
              f"test ball {rec['test']['ball']['mAP50_95']:.4f} ({minutes:.1f} min)", flush=True)


def main():
    CODE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    for name, text in EMBED.items():
        (CODE / name).write_text(text)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.163"], check=True)
    bc = load_bc()
    from ultralytics import YOLO
    YOLO("yolo11s.pt")  # download once, before the workers race for it
    bc.prepare_data(DATA, "splits.json", RAW)  # build the video-grouped split once
    import torch
    n_gpu = torch.cuda.device_count()
    print(f"{n_gpu} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}", flush=True)
    t0, procs = time.time(), []
    for g in range(max(n_gpu, 1)):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        mine = JOBS[g::max(n_gpu, 1)]
        procs.append(subprocess.Popen([sys.executable, __file__, "worker", json.dumps(mine)], env=env, cwd=ROOT))
    codes = [p.wait() for p in procs]
    print("workers exited", codes, f"after {(time.time() - t0) / 60:.1f} min", flush=True)

    import statistics
    recs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*_s*.json"))]
    summary = {"runs": len(recs), "expected": len(JOBS), "configs": {}}
    for cfg in CONFIGS:
        rs = [r for r in recs if r["config"] == cfg]
        for split in ("val", "test"):
            vals = [r[split]["ball"]["mAP50_95"] for r in rs]
            if vals:
                summary["configs"].setdefault(cfg, {})[split] = {
                    "per_seed": {r["seed"]: r[split]["ball"]["mAP50_95"] for r in rs},
                    "mean": statistics.mean(vals), "stdev": statistics.stdev(vals) if len(vals) > 1 else None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "worker":
        worker(json.loads(sys.argv[2]))
    else:
        main()

#!/bin/bash
# Allocate the T4 session and install BallHawk code + pinned dependencies on it.
# usage: tools/setup_vm.sh [cpu]      (default T4; `cpu` when GPU quota is exhausted and the job is inference-only)
# The dataset builds itself on first bc.prepare_data().
ROOT=/Users/rohantrivedi/Downloads/Projects/BallHawk
F='new version\|colab update\|update check\|uv tool\|^$\|silence this check'
cd "$ROOT" || exit 1
ACCEL=(--gpu T4); [ "$1" = cpu ] && ACCEL=()
out=$(colab new -s ballhawk "${ACCEL[@]}" 2>&1 | grep -v "$F"); echo "$out"
echo "$out" | grep -q "Session READY" || { echo "allocation failed; aborting setup"; exit 1; }
echo 'import os; os.makedirs("/content/ballhawk", exist_ok=True)' | colab exec -s ballhawk 2>&1 | grep -v "$F"
for f in ballhawk_common.py ballhawk_video.py ballhawk_pitch.py splits.json splits_pitch.json yolo11-p2.yaml run_queue.sh remeasure_fps.py eval_val.py phase0_checks.py notebooks/*.ipynb; do
  colab upload -s ballhawk "$f" "/content/ballhawk/$(basename "$f")" 2>&1 | grep -v "$F" | grep -v Uploaded
done
echo 'import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.163", "papermill", "sahi"], check=True)
import ultralytics; print("ultralytics", ultralytics.__version__)' | colab exec -s ballhawk --timeout 600 2>&1 | grep -v "$F"

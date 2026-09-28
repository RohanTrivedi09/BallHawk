#!/bin/bash
# Retry T4 allocation every 30 min (up to 8 h) — covers both an exhausted free-tier quota and an
# orphaned assignment holding the slot. Once granted, upload the weights the queued notebooks need
# and launch them. Exits 0 once the queue is launched, 1 if no GPU came.
# usage: tools/run_when_gpu.sh "<smoke notebooks>" "<full notebooks>"
ROOT=/Users/rohantrivedi/Downloads/Projects/BallHawk
F='new version\|colab update\|update check\|uv tool\|^$\|silence this check'
SMOKE=$1; FULL=$2
for attempt in $(seq 1 16); do
  if "$ROOT/tools/setup_vm.sh" > /dev/null 2>&1; then
    echo "$(date '+%T') T4 granted on attempt $attempt"
    echo 'import os; [os.makedirs(f"/content/results/{d}", exist_ok=True) for d in ("baseline_yolo11s_640", "res_yolo11s_1280", "pitch_kp_yolo11s")]' \
      | colab exec -s ballhawk 2>&1 | grep -v "$F"
    for f in baseline_yolo11s_640/best.pt res_yolo11s_1280/best.pt pitch_kp_yolo11s/best.pt pitch_kp_yolo11s/pitch_metrics.json; do
      colab upload -s ballhawk "$ROOT/results/$f" "/content/results/$f" 2>&1 | grep -v "$F" | grep -v Uploaded
    done
    echo "import subprocess; subprocess.run(\"cd /content/ballhawk && nohup bash -c 'bash run_queue.sh smoke $SMOKE; bash run_queue.sh full $FULL' > /dev/null 2>&1 &\", shell=True); print('queue launched')" \
      | colab exec -s ballhawk 2>&1 | grep -v "$F"
    exit 0
  fi
  echo "$(date '+%T') no T4 (attempt $attempt/16); retrying in 30 min"
  colab stop -s ballhawk > /dev/null 2>&1
  sleep 1800
done
exit 1

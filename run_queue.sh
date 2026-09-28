#!/bin/bash
# Run experiment notebooks one after another on the Colab VM with papermill.
# usage: run_queue.sh smoke|full NB.ipynb [NB.ipynb ...]
# Per-notebook log: /content/logs/<name>[_smoke].log; progress: /content/logs/queue.log.
# A failed notebook is logged and the queue moves on, so one crash does not idle the GPU.
MODE=$1; shift
cd /content/ballhawk || exit 1
mkdir -p /content/logs /content/executed
for nb in "$@"; do
  name=${nb%.ipynb}
  if [ "$MODE" = smoke ]; then suffix=_smoke; params=(-p SMOKE True); else suffix=; params=(); fi
  echo "$(date -u +%H:%M) START $name$suffix" >> /content/logs/queue.log
  if papermill "$nb" "/content/executed/$name$suffix.ipynb" "${params[@]}" --log-output --cwd /content/ballhawk \
       > "/content/logs/$name$suffix.log" 2>&1; then
    echo "$(date -u +%H:%M) DONE $name$suffix" >> /content/logs/queue.log
  else
    echo "$(date -u +%H:%M) FAIL $name$suffix" >> /content/logs/queue.log
  fi
done
echo "$(date -u +%H:%M) QUEUE COMPLETE $MODE" >> /content/logs/queue.log

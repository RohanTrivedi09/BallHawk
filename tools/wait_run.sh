#!/bin/bash
# Watch a background job on the Colab VM until it ends.
# usage: tools/wait_run.sh <remote log to tail at the end> [interval_s] [process pattern]
# Every poll: appends VM RAM/GPU use to logs/vm_mem.log locally, and runs tools/sync_results.sh
# whenever /content/logs/queue.log gains a line, so a lost VM costs at most the run in progress.
# Exits 0 when no process matches the pattern, 2 if the VM cannot be reached.
LOG=$1; INT=${2:-30}; PAT=${3:-papermill}
ROOT=/Users/rohantrivedi/Downloads/Projects/BallHawk
F='new version\|colab update\|update check\|uv tool\|^$\|silence this check'
mkdir -p "$ROOT/logs"
probe() {
  # [x]yz bracket trick: pgrep -f must not match its own shell command line.
  cat <<EOF | colab exec -s ballhawk 2>&1 | grep -v "$F"
import subprocess
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print("STATE", sh("pgrep -f '[${PAT:0:1}]${PAT:1}' >/dev/null && echo RUNNING || echo DONE"))
print("MEM", sh("free -m | awk '/Mem:/{print \$3\"/\"\$2\" MB RAM\"}'"), sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))
print("QUEUE", sh("wc -l < /content/logs/queue.log 2>/dev/null || echo 0"), sh("tail -1 /content/logs/queue.log 2>/dev/null"))
EOF
}
last_queue=""
while true; do
  out=$(probe)
  state=$(echo "$out" | awk '/^STATE/{print $2}')
  [ -z "$state" ] && { echo "VM check failed (session gone?):"; echo "$out" | tail -5; exit 2; }
  echo "$(date '+%F %T') $(echo "$out" | grep '^MEM') | $(echo "$out" | grep '^QUEUE')" >> "$ROOT/logs/vm_mem.log"
  queue=$(echo "$out" | grep '^QUEUE')
  if [ "$queue" != "$last_queue" ]; then
    "$ROOT/tools/sync_results.sh" > /dev/null
    last_queue=$queue
  fi
  [ "$state" = DONE ] && break
  sleep "$INT"
done
"$ROOT/tools/sync_results.sh" > /dev/null
echo "import subprocess; print(subprocess.run('tail -c 4000 $LOG', shell=True, capture_output=True, text=True).stdout)" | colab exec -s ballhawk 2>&1 | grep -v "$F"

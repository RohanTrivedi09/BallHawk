#!/usr/bin/env bash
# Assemble build/space: everything the BallHawk server needs, in one folder, ready for
#   docker build -t ballhawk build/space                       (local test)
#   hf upload <user>/ballhawk build/space --repo-type space    (Hugging Face Space)
# The repository keeps the shared modules at the root and the site in web/; a Space needs them next to app.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build/space"

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$ROOT"/demo/{app.py,requirements.txt,Dockerfile,README.md} "$OUT"/
cp "$ROOT"/ballhawk_{hud,pitch,tactical,video}.py "$OUT"/
cp -R "$ROOT"/demo/weights "$ROOT"/demo/examples "$OUT"/
cp -R "$ROOT"/web "$OUT"/web
rm -f "$OUT"/web/build.py   # site data is already generated in web/data
printf '**/__pycache__\n**/.DS_Store\n' > "$OUT"/.dockerignore
du -sh "$OUT"
echo "Assembled $OUT"

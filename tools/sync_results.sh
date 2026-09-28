#!/bin/bash
# Pull every non-smoke result file, executed notebook and log from the VM that is not yet local.
cd /Users/rohantrivedi/Downloads/Projects/BallHawk || exit 1
F='new version\|colab update\|update check\|uv tool\|^$\|silence this check'
LIST=$(echo 'import pathlib
for base in ["/content/results", "/content/executed", "/content/logs"]:
    for p in pathlib.Path(base).rglob("*"):
        if p.is_file() and "_smoke" not in str(p): print(p)' | colab exec -s ballhawk 2>&1 | grep '^/content/')
for remote in $LIST; do
  local_path=${remote#/content/}
  [ -f "$local_path" ] && [[ "$local_path" != logs/* ]] && [[ "$local_path" != *.json ]] && continue   # logs and JSON refresh every sync
  mkdir -p "$(dirname "$local_path")"
  colab download -s ballhawk "$remote" "$local_path" 2>&1 | grep -v "$F" | grep -v Downloaded
done
find results executed logs -type f | wc -l

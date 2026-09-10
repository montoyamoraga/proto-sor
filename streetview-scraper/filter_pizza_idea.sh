#!/usr/bin/env bash
# Apply the YOLO object filter to frames_pizza_idea/, writing cutouts to
# frames_pizza_idea_yolo/.
set -euo pipefail
cd "$(dirname "$0")"

./venv/bin/python3 yolo_filter.py \
  --input-dir frames_pizza_idea \
  --output-dir frames_pizza_idea_yolo \
  "$@"

#!/usr/bin/env bash
# CrashTwin environment setup for a NEW Linux machine (Ubuntu 20.04/22.04 tested by CARLA 0.9.15).
# Creates <project>/.venv (Python 3.10), installs the pinned packages, downloads the YOLO weights
# (checked against their SHA-256) and runs the environment check. Safe to run again: it skips what exists.
#
#   bash <project>/crashtwin/scripts/setup_env.sh
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # folder that contains crashtwin/
cd "$PROJ"
echo "Project folder: $PROJ"

if [ ! -x .venv/bin/python ]; then
    echo "Creating .venv with Python 3.10 ..."
    python3.10 -m venv .venv || { echo "Install Python 3.10 first, e.g. sudo apt install python3.10 python3.10-venv"; exit 1; }
fi
PY="$PROJ/.venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r crashtwin/requirements.txt

W="$PROJ/models/yolo/yolo11s.pt"
if [ ! -f "$W" ]; then
    mkdir -p "$(dirname "$W")"
    echo "Downloading YOLO11s weights ..."
    curl -L --fail -o "$W.partial" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
    mv "$W.partial" "$W"
fi
echo "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5  $W" | sha256sum -c -

"$PY" crashtwin/scripts/check_env.py
echo
echo "Next: export CARLA_ROOT=/path/to/CARLA_0.9.15   (add it to ~/.bashrc)"
echo "OpenCV/Open3D need system libraries on minimal installs: sudo apt install libgl1 libglib2.0-0 libgomp1"

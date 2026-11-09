#!/usr/bin/env bash
# Quick end-to-end check: load the UR5e into LIBERO, run the scripted expert on
# two tasks, save videos, and round-trip a policy server.
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
OUT=${1:-runs/smoke}

python -m libero6dof.evaluate --suite libero_spatial --tasks 0,1 --episodes 3 \
  --policy scripted --video-dir "$OUT/videos" --videos-per-task 1 --out "$OUT/scripted.json"

python -m libero6dof.server --port 8765 & SP=$!
sleep 5
python -m libero6dof.evaluate --suite libero_spatial --tasks 0 --episodes 1 --max-steps 20 \
  --policy remote --port 8765 --out "$OUT/remote.json"
kill $SP
echo "Smoke test passed. Videos in $OUT/videos"

#!/usr/bin/env bash
# Policy-side environment (Python 3.12): convert demos to LeRobot datasets, train
# locally, and serve LeRobot checkpoints (SmolVLA, pi0, pi0.5, ...) to the simulator.
# Kept separate from .venv because LIBERO pins numpy<2 and old robosuite.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.12 .venv-lerobot
source .venv-lerobot/bin/activate
uv pip install -e ".[lerobot]"
echo
echo "Done. Activate with:  source $ROOT/.venv-lerobot/bin/activate"
echo "Log in for Hub uploads:  hf auth login"

#!/usr/bin/env bash
# Install libero6dof + LIBERO on Ubuntu 22.04/24.04 (tested headless with EGL).
# Usage: bash scripts/install.sh            (run from the libero6dof folder)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# System GL libraries for offscreen rendering
if command -v apt-get >/dev/null; then
  sudo apt-get install -y libegl1 libgl1 libglfw3 ffmpeg || true
fi

uv venv -p 3.10 .venv
source .venv/bin/activate

[ -d LIBERO ] || git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
uv pip install -e ".[sim]"
uv pip install -e LIBERO --no-deps

# LIBERO asks for paths interactively on first import; write its config up front.
L="$ROOT/LIBERO/libero/libero"
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $ROOT/LIBERO/libero/datasets
assets: $L/assets
EOF

echo
echo "Done. Activate with:  source $ROOT/.venv/bin/activate"
echo "Then run:             bash scripts/smoke_test.sh"
echo "For training/serving VLAs: bash scripts/install_lerobot.sh"

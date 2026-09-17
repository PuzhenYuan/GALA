#!/usr/bin/env bash
# Install into the currently active, dedicated Python 3.10 environment.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip install --no-build-isolation flash-attn==2.7.1.post4
mkdir -p third_party
install_repo() {
    local url="$1" target="$2" revision="$3"
    if [[ ! -d "$target" ]]; then
        git clone "$url" "$target"
        git -C "$target" checkout --detach "$revision"
    elif [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ]]; then
        echo "Expected revision $revision in $target; use a fresh GALA checkout." >&2
        exit 1
    fi
}
install_repo https://github.com/ARISE-Initiative/robosuite.git third_party/robosuite a071383d53568ab798eb315c0e95357911be922d
install_repo https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git third_party/robocasa-gr1-tabletop-tasks 4840e671596f93ca03651524b9f72ffb1aadfeff
if git -C third_party/robocasa-gr1-tabletop-tasks apply --check "$PROJECT_ROOT/patches/robocasa-basket.patch" 2>/dev/null; then
    git -C third_party/robocasa-gr1-tabletop-tasks apply "$PROJECT_ROOT/patches/robocasa-basket.patch"
else
    git -C third_party/robocasa-gr1-tabletop-tasks apply --reverse --check "$PROJECT_ROOT/patches/robocasa-basket.patch"
fi
"$PYTHON_BIN" -m pip install -c requirements.txt -e third_party/robosuite -e third_party/robocasa-gr1-tabletop-tasks
"$PYTHON_BIN" -m pip install --no-deps -e .
"$PYTHON_BIN" third_party/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y
"$PYTHON_BIN" -m pip check

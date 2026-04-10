#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

python_is_supported() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || return 1
    "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)
PY
}

setup_local() {
    "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
    "$ROOT_DIR/.venv/bin/pip" install --upgrade pip
    "$ROOT_DIR/.venv/bin/pip" install -r "$ROOT_DIR/requirements.txt"
    printf 'Local environment ready.\n'
    printf 'Activate it with: source %s/.venv/bin/activate\n' "$ROOT_DIR"
}

setup_container() {
    if ! command -v podman >/dev/null 2>&1; then
        printf 'Python %s is not supported by TensorFlow, and podman is not installed.\n' "$("$PYTHON_BIN" --version 2>/dev/null || printf 'is missing')"
        printf 'Install Python 3.10-3.13 or podman, then rerun setup.sh.\n'
        exit 1
    fi

    podman build -t tensorproject-dev "$ROOT_DIR"
    printf 'Container image ready.\n'
    printf 'Run commands with:\n'
    printf 'podman run --rm -it -v %s:/workspace:Z -w /workspace tensorproject-dev bash\n' "$ROOT_DIR"
}

if python_is_supported; then
    setup_local
else
    setup_container
fi

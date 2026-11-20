#!/usr/bin/env bash
# One-time setup of the Python environment (generators, validation, analysis).
# Run after `source env/modules.sh` so the venv uses the module's Python.
#
#   env/setup_venv.sh            # exact versions from requirements.lock (default)
#   env/setup_venv.sh --loose    # only the direct pins in requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
req=requirements.lock
[ "${1:-}" = --loose ] && req=requirements.txt
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r "$req"
python3 -m pip freeze > .venv/pip-freeze.txt
echo "venv ready: $(python3 --version), packages from $req (frozen list in env/.venv/pip-freeze.txt)"

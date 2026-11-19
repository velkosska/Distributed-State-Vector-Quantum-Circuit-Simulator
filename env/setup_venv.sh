#!/usr/bin/env bash
# One-time setup of the Python environment used by the generators and validation.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

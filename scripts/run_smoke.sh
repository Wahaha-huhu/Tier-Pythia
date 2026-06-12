#!/usr/bin/env bash
# Tiny end-to-end run to verify the environment BEFORE the full run.
# Downloads 4 checkpoints, trains 4 tiny arms (~60 steps each). A few minutes total
# (mostly model download on first run). If this finishes and writes results/SUMMARY.md,
# the full run will work.
set -e
cd "$(dirname "$0")/.."
python run.py smoke
echo
echo "Smoke test done. Inspect results/SUMMARY.md and results/*.png"

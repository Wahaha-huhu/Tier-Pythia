#!/usr/bin/env bash
# Full thesis run: induction window + the 2 tasks x 3 schedules x 3 seeds factorial.
# Expect ~3-4 h on a single A100 (most of it the 18 continued-training arms),
# plus ~5-7 GB of checkpoint downloads on first run.
set -e
cd "$(dirname "$0")/.."
python run.py all \
    --seeds 3 \
    --schedules native_low deep_low rewarm \
    --tasks 1 2 \
    --steps 3000
echo
echo "Full run done. Key artifacts in results/ (see README 'What to send back')."

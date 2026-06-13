#!/usr/bin/env bash
# Full thesis run: induction window + (2 tasks x 3 schedules x 3 seeds) factorial.
# Hop-1 arms run 1500 steps (they saturate fast); Hop-2 arms run 8000 (long runway).
# With L=5 this is ~3 h on a single A100, plus ~5-7 GB of checkpoint downloads first time.
#
# Only launch this AFTER the Hop-2 formation probe (see README) confirms the timescale.
set -e
cd "$(dirname "$0")/.."
python run.py all \
    --seeds 3 \
    --schedules native_low deep_low rewarm \
    --tasks 1 2 \
    --steps 8000
echo
echo "Full run done. Key artifacts in results/ (see README 'What to send back')."

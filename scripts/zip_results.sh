#!/usr/bin/env bash
# Bundle the (small) key results to send back. Excludes any model weights.
set -e
cd "$(dirname "$0")/.."
rm -f results_bundle.zip
zip -r results_bundle.zip results/ -x '*.pt' '*.bin' '*.safetensors' >/dev/null
echo "Created results_bundle.zip ($(du -h results_bundle.zip | cut -f1))"
echo "Download this file and send it back."

"""Collect the key results into a small zip to send back. No large checkpoints.

Gathers every summary.json, log.jsonl, and the calibration and observational json files
under the runs directory, plus a manifest, into one archive.

Usage
  python -m scripts.pack_results --runs-dir runs --out cp_pythia_results.zip
"""

import argparse
import glob
import json
import os
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="cp_pythia_results.zip")
    args = ap.parse_args()

    patterns = ["**/summary.json", "**/log.jsonl", "**/difficulty.json",
                "**/learn*.json", "**/barrier.json", "**/cascade.json"]
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(args.runs_dir, p), recursive=True)
    files = sorted(set(files))

    manifest = {"runs_dir": args.runs_dir, "n_files": len(files), "summaries": []}
    for f in files:
        if f.endswith("summary.json"):
            try:
                manifest["summaries"].append({"path": f, **{k: v for k, v in json.load(open(f)).items()
                                                            if not isinstance(v, (dict, list))}})
            except Exception:
                pass

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=os.path.relpath(f, start=os.path.dirname(args.runs_dir) or "."))
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} with {len(files)} files, {size:.2f} MB")
    print("summaries packed", len(manifest["summaries"]))


if __name__ == "__main__":
    main()

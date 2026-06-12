"""Observational cascade across released Pythia checkpoints. No training.

For each checkpoint, report the top induction score on the repeated-token diagnostic and
the composite hop2 content excess on the synthetic task, to locate the ordering and the
emergence shape. Supporting evidence only, no causal claim.

Usage
  python -m scripts.run_observational --model pythia-160m-deduped --device cuda \
      --steps 0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,143000
"""

import argparse
import json

import numpy as np

from cp_pythia.config import TaskConfig
from cp_pythia.generator import Generator, repeated_token_diag
from cp_pythia.model_io import load_pythia
from cp_pythia.induction import induction_scores, top_heads
from cp_pythia.metrics import eval_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m-deduped")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--content-vocab", type=int, default=64)
    ap.add_argument("--chain-length", type=int, default=8)
    ap.add_argument("--steps", required=True, help="comma-separated checkpoint steps")
    ap.add_argument("--out-dir", default="runs/observational")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    tcfg = TaskConfig(content_vocab=args.content_vocab, chain_length=args.chain_length, k_max=2)
    gen = Generator(tcfg)
    rng = np.random.default_rng(0)
    block = min(16, tcfg.content_vocab // 2)
    rows = []
    for step in [int(s) for s in args.steps.split(",")]:
        model, _, info = load_pythia(args.model, step, args.device, args.dtype, eager=True)
        diag = repeated_token_diag(tcfg, 64, block, rng)
        sc = induction_scores(model, diag, block, args.device, info["n_layers"], info["n_heads"])
        top = top_heads(sc, 1)[0]
        h2 = eval_task(model, gen, "hop2_only", 16, 64, tcfg.content_ids, args.device, rng)
        h1 = eval_task(model, gen, "hop1", 16, 64, tcfg.content_ids, args.device, rng)
        rows.append({"step": step, "induction_top": float(sc[top]), "induction_head": list(top),
                     "hop1_excess": h1["excess_content"], "hop2_excess": h2["excess_content"]})
        print(rows[-1])
        del model
    json.dump(rows, open(f"{args.out_dir}/cascade.json", "w"), indent=2)
    print("ordering, look for induction rising before hop2 excess")


if __name__ == "__main__":
    main()

"""Calibration gates, see plan tier one section 2.6. Run after sanity passes.

Gate one, difficulty. Sweep chain length and hop count and report base hop1 and hop2
content excess, looking for hop1 solvable and hop2 at floor for the base.
Gate two, learnability. A rewarmed run must teach hop2 to confirm a positive arm exists.
Gate three, barrier. A low-rate run against a rewarmed run on hop2, report the
late-to-early style excess ratio that decides whether the full matrix is warranted.

Usage
  python -m scripts.run_calibration --model pythia-160m-deduped --device cuda --stage difficulty
  python -m scripts.run_calibration --model pythia-160m-deduped --device cuda --stage barrier
"""

import argparse
import json

import numpy as np

from cp_pythia.config import TaskConfig, TrainConfig, PYTHIA_LR, default_arms
from cp_pythia.generator import Generator
from cp_pythia.model_io import load_pythia
from cp_pythia.metrics import eval_task
from cp_pythia.train import train


def stage_difficulty(args):
    rng = np.random.default_rng(0)
    rows = []
    for M in [6, 8, 10, 12]:
        for K in [2]:
            tcfg = TaskConfig(content_vocab=args.content_vocab, chain_length=M, k_max=K)
            model, _, info = load_pythia(args.model, args.revision_step, args.device, args.dtype)
            h1 = eval_task(model, Generator(tcfg), "hop1", 16, 64, tcfg, args.device, rng)
            h2 = eval_task(model, Generator(tcfg), "hop2_only", 16, 64, tcfg, args.device, rng)
            rows.append({"M": M, "K": K, "hop1_excess": h1["excess_content"],
                         "hop2_excess": h2["excess_content"],
                         "hop1_acc": h1["acc_content"], "hop2_acc": h2["acc_content"]})
            print(rows[-1])
            del model
    json.dump(rows, open(f"{args.out_dir}/difficulty.json", "w"), indent=2)
    print("look for hop1_excess high and hop2_excess near zero")


def stage_learnability(args):
    tcfg = TaskConfig(content_vocab=args.content_vocab, chain_length=args.chain_length, k_max=2)
    model, _, _ = load_pythia(args.model, args.revision_step, args.device, args.dtype)
    peak = PYTHIA_LR.get(args.model, {"peak": 6e-4})["peak"]
    arms = default_arms(peak, PYTHIA_LR.get(args.model, {"min": 6e-5})["min"])
    tr = TrainConfig(model=args.model, revision_step=args.revision_step, post_steps=args.post_steps,
                     intro_task="compose", out_dir=f"{args.out_dir}/learn_rewarm")
    spath = train(model, tcfg, tr, arms["rewarm"], args.device)
    s = json.load(open(spath))
    print("hop2 tail excess under rewarm", s.get("tail_hop2_only_excess_content"))
    print("if this is not high, reduce difficulty before proceeding")


def stage_barrier(args):
    tcfg = TaskConfig(content_vocab=args.content_vocab, chain_length=args.chain_length, k_max=2)
    peak = PYTHIA_LR.get(args.model, {"peak": 6e-4})["peak"]
    minimum = PYTHIA_LR.get(args.model, {"min": 6e-5})["min"]
    arms = default_arms(peak, minimum)
    res = {}
    for armname in ["low", "rewarm"]:
        model, _, _ = load_pythia(args.model, args.revision_step, args.device, args.dtype)
        tr = TrainConfig(model=args.model, revision_step=args.revision_step, post_steps=args.post_steps,
                         intro_task="compose", out_dir=f"{args.out_dir}/barrier_{armname}")
        spath = train(model, tcfg, tr, arms[armname], args.device)
        res[armname] = json.load(open(spath)).get("tail_hop2_only_excess_content", 0.0)
        del model
    ratio = res["low"] / max(res["rewarm"], 1e-6)
    decision = "proceed_full_matrix" if ratio <= 0.5 else "no_barrier_revisit_difficulty"
    out = {"low_excess": res["low"], "rewarm_excess": res["rewarm"], "low_to_rewarm_ratio": ratio,
           "decision": decision}
    json.dump(out, open(f"{args.out_dir}/barrier.json", "w"), indent=2)
    print(out)


def stage_search(args):
    """Hunt for a positive composition arm. Stage hop1, then train hop2 with hop1 and
    language replay, sweeping a moderate post rate. Reports hop2 excess for each rate.
    """
    from cp_pythia.config import ArmConfig
    from cp_pythia.perplexity import build_blocks
    tcfg = TaskConfig(content_vocab=args.content_vocab, chain_length=args.chain_length, k_max=2)
    grid = [float(x) for x in args.search_lrs.split(",")]
    rows = []
    for lr in grid:
        model, tok, _ = load_pythia(args.model, args.revision_step, args.device, args.dtype)
        lang = build_blocks(tok, n_blocks=256, block_len=512, split="train") if args.replay_lang_frac > 0 else None
        arm = ArmConfig(name="search", post_lr=lr, rewarm_warmup=200)
        tr = TrainConfig(model=args.model, revision_step=args.revision_step,
                         prereq_steps=args.prereq_steps, prereq_lr=args.prereq_lr,
                         post_steps=args.post_steps, intro_task="hop2_only",
                         replay_hop1_frac=args.replay_hop1_frac, replay_lang_frac=args.replay_lang_frac,
                         eval_interval=args.eval_interval, seed=0,
                         out_dir=f"{args.out_dir}/search_lr{lr:.0e}")
        spath = train(model, tcfg, tr, arm, args.device, lang_blocks=lang)
        s = json.load(open(spath))
        rows.append({"post_lr": lr,
                     "hop2_excess": s.get("tail_hop2_only_excess_content"),
                     "hop1_acc": s.get("tail_hop1_acc_content"),
                     "ppl_delta": s.get("tail_ppl_delta_from_base"),
                     "transition": s.get("hop2_transition")})
        print(rows[-1])
        del model
    json.dump(rows, open(f"{args.out_dir}/search.json", "w"), indent=2)
    best = max(rows, key=lambda r: (r["hop2_excess"] or -1))
    print("best rate", best["post_lr"], "hop2 excess", best["hop2_excess"])
    print("if no rate gives a clear positive excess, stage longer or ease the composition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["difficulty", "learnability", "barrier", "search"])
    ap.add_argument("--model", default="pythia-160m-deduped")
    ap.add_argument("--revision-step", type=int, default=143000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--content-vocab", type=int, default=64)
    ap.add_argument("--chain-length", type=int, default=8)
    ap.add_argument("--post-steps", type=int, default=4000)
    ap.add_argument("--prereq-steps", type=int, default=0)
    ap.add_argument("--prereq-lr", type=float, default=1.0e-4)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--replay-hop1-frac", type=float, default=0.0)
    ap.add_argument("--replay-lang-frac", type=float, default=0.0)
    ap.add_argument("--search-lrs", default="3e-5,1e-4,2e-4,3e-4")
    ap.add_argument("--out-dir", default="runs/calib")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    {"difficulty": stage_difficulty, "learnability": stage_learnability,
     "barrier": stage_barrier, "search": stage_search}[args.stage](args)


if __name__ == "__main__":
    main()

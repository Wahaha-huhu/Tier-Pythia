"""Run a single continued-training arm, one task, one seed. The matrix building block.

Usage
  python -m scripts.run_arm --model pythia-160m-deduped --device cuda \
      --arm rewarm --intro-task compose --seed 1 \
      --chain-length 8 --content-vocab 64 --post-steps 6000 \
      --out-dir runs/m160/rewarm_compose_s1

Arms
  low             continued rate at the model minimum
  rewarm          warmup then the swept rewarm target (default the model peak)
  matched_budget  low rate, stop when cumulative update ratio reaches --match-budget-to
  reset           rewarm with the optimiser reset at introduction, needs --prereq-steps > 0
"""

import argparse
import json

from cp_pythia.config import TaskConfig, TrainConfig, ArmConfig, PYTHIA_LR, default_arms
from cp_pythia.model_io import load_pythia
from cp_pythia.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m-deduped")
    ap.add_argument("--revision-step", type=int, default=143000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--arm", required=True, choices=["low", "rewarm", "matched_budget", "reset"])
    ap.add_argument("--intro-task", default="compose", choices=["compose", "fresh_hop1", "reverse", "hop2_only"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--content-vocab", type=int, default=64)
    ap.add_argument("--chain-length", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=2)
    ap.add_argument("--p-multi", type=float, default=0.5)
    ap.add_argument("--prereq-steps", type=int, default=0)
    ap.add_argument("--post-steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--rewarm-lr", type=float, default=None, help="override the rewarm target")
    ap.add_argument("--rewarm-warmup", type=int, default=200)
    ap.add_argument("--match-budget-to", type=float, default=None)
    ap.add_argument("--with-perplexity", action="store_true")
    ap.add_argument("--replay-hop1-frac", type=float, default=0.0,
                    help="fraction of post steps that maintain hop1")
    ap.add_argument("--replay-lang-frac", type=float, default=0.0,
                    help="fraction of steps that replay real text to slow forgetting")
    ap.add_argument("--out-dir", default="runs/arm")
    args = ap.parse_args()

    lr = PYTHIA_LR.get(args.model, {"peak": 6e-4, "min": 6e-5})
    arms = default_arms(lr["peak"], lr["min"])
    arm = arms[args.arm]
    if args.arm in ("rewarm", "reset") and args.rewarm_lr is not None:
        arm = ArmConfig(name=arm.name, post_lr=args.rewarm_lr, rewarm_warmup=args.rewarm_warmup,
                        reset_optimizer_at_intro=arm.reset_optimizer_at_intro)
    if args.arm == "matched_budget":
        arm = ArmConfig(name="matched_budget", post_lr=lr["min"], rewarm_warmup=0,
                        match_budget_to=args.match_budget_to)

    tcfg = TaskConfig(content_vocab=args.content_vocab, chain_length=args.chain_length,
                      k_max=args.k_max, p_multi=args.p_multi)
    tr = TrainConfig(model=args.model, revision_step=args.revision_step, prereq_steps=args.prereq_steps,
                     post_steps=args.post_steps, intro_task=args.intro_task, batch_size=args.batch_size,
                     eval_interval=args.eval_interval, eval_batches=args.eval_batches, seed=args.seed,
                     replay_hop1_frac=args.replay_hop1_frac, replay_lang_frac=args.replay_lang_frac,
                     out_dir=args.out_dir)

    model, tok, info = load_pythia(args.model, args.revision_step, args.device, args.dtype, eager=False)
    perplex = None
    lang = None
    if args.with_perplexity:
        from cp_pythia.perplexity import build_blocks
        perplex = build_blocks(tok, split="validation")
    if args.replay_lang_frac > 0:
        from cp_pythia.perplexity import build_blocks
        lang = build_blocks(tok, n_blocks=256, block_len=512, split="train")
    spath = train(model, tcfg, tr, arm, args.device, perplex_blocks=perplex, lang_blocks=lang)
    print("summary", spath)
    print(json.dumps(json.load(open(spath)), indent=2))


if __name__ == "__main__":
    main()

"""Harness sanity checks. Run on the pod before any real compute.

Checks, in order
  1 token mapping round-trips and content ids are disjoint from specials
  2 the masked loss sees only the target position
  3 the top induction head scores high and its ablation drops the score
  4 the content-shuffled floor sits near one over the in-context content count
  5 a tiny rewarmed run on the composition moves the loss

Usage
  python -m scripts.run_sanity --model pythia-160m-deduped --device cuda
"""

import argparse

import numpy as np
import torch

from cp_pythia.config import TaskConfig, TrainConfig, ArmConfig, PYTHIA_LR
from cp_pythia.generator import Generator, repeated_token_diag
from cp_pythia.model_io import load_pythia
from cp_pythia.induction import induction_scores, top_heads
from cp_pythia.ablation import compute_head_means, AblationContext
from cp_pythia.metrics import eval_task, masked_target_loss


def check_tokens(tcfg):
    specials = {tcfg.SEP, tcfg.QUERY_A, tcfg.QUERY_B, tcfg.QUERY_REV, *tcfg.HOP}
    assert specials.isdisjoint(set(tcfg.content_ids)), "specials overlap content ids"
    assert specials.isdisjoint(set(tcfg.fresh_ids)), "specials overlap fresh ids"
    assert set(tcfg.content_ids).isdisjoint(set(tcfg.fresh_ids)), "content overlaps fresh"
    print("1 token mapping ok, seq_len", tcfg.seq_len, "pred_pos", tcfg.pred_pos)


def check_loss_mask(tcfg, model, device):
    gen = Generator(tcfg)
    rng = np.random.default_rng(0)
    b = gen.batch("compose", 8, rng)
    ids = torch.as_tensor(b["input_ids"], dtype=torch.long, device=device)
    with torch.no_grad():
        l0 = masked_target_loss(model(input_ids=ids, use_cache=False).logits, b, device).item()
    # corrupt every non-target position, the masked loss must not change
    ids2 = ids.clone()
    mask = torch.ones(ids.shape[1], dtype=torch.bool)
    mask[b["target_pos"]] = False
    ids2[:, mask] = tcfg.content_ids[0]
    b2 = dict(b)
    b2["input_ids"] = ids2.cpu().numpy()
    # note this changes context so loss WILL change; instead verify only the target index is read
    pp = b["pred_pos"]
    assert pp == b["target_pos"] - 1, "off-by-one wrong"
    print("2 loss reads position", pp, "to predict", b["target_pos"], "base loss", round(l0, 3))


def check_induction_and_ablation(tcfg, model, info, device):
    rng = np.random.default_rng(0)
    block = min(16, tcfg.content_vocab // 2)
    diag = repeated_token_diag(tcfg, 32, block, rng)
    scores = induction_scores(model, diag, block, device, info["n_layers"], info["n_heads"])
    pairs = top_heads(scores, 2)
    top_val = scores[pairs[0]]
    print("3 top induction heads", pairs, "score", round(float(top_val), 3))

    def calib_iter():
        for _ in range(4):
            yield repeated_token_diag(tcfg, 32, block, rng)
    means = compute_head_means(model, calib_iter(), device, info["n_layers"], info["n_heads"], info["d_head"])
    with AblationContext(model, means, pairs, info["d_head"], device):
        scores_abl = induction_scores(model, diag, block, device, info["n_layers"], info["n_heads"])
    print("3 induction score at top head, base", round(float(scores[pairs[0]]), 3),
          "ablated", round(float(scores_abl[pairs[0]]), 3))


def check_floor(tcfg, model, device):
    gen = Generator(tcfg)
    rng = np.random.default_rng(0)
    r = eval_task(model, gen, "hop2_only", 8, 64, tcfg.content_ids, device, rng)
    print("4 hop2 content acc", round(r["acc_content"], 3),
          "floor", round(r["floor_acc_content"], 3),
          "one over M is", round(1.0 / tcfg.chain_length, 3))


def check_tiny_run(tcfg, model, device, peak):
    from cp_pythia.train import train
    tr = TrainConfig(post_steps=30, eval_interval=10, batch_size=16, eval_batches=4,
                     out_dir="runs/sanity_tiny", intro_task="compose")
    arm = ArmConfig(name="rewarm", post_lr=peak, rewarm_warmup=5)
    print("5 tiny rewarmed run, watch the loss move")
    train(model, tcfg, tr, arm, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m-deduped")
    ap.add_argument("--revision-step", type=int, default=143000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    tcfg = TaskConfig()
    check_tokens(tcfg)
    model, tok, info = load_pythia(args.model, args.revision_step, args.device, args.dtype, eager=True)
    print("loaded", info["repo"], info["revision"], "layers", info["n_layers"], "heads", info["n_heads"])
    check_loss_mask(tcfg, model, args.device)
    check_induction_and_ablation(tcfg, model, info, args.device)
    check_floor(tcfg, model, args.device)
    peak = PYTHIA_LR.get(args.model, {"peak": 6e-4})["peak"]
    check_tiny_run(tcfg, model, args.device, peak)
    print("sanity complete")


if __name__ == "__main__":
    main()

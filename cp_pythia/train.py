"""Continued-training loop with the rate arms. Torch.

Optional prerequisite phase, then introduction of the composition or a control task,
then the post-introduction phase under one rate arm. Logs accuracies, prerequisite
retention, floors, pretraining perplexity, and the cumulative update-to-weight ratio.
"""

import json
import os
import time

import numpy as np
import torch

from .generator import Generator
from .metrics import eval_task, masked_target_loss


def lr_at(post_step, arm):
    if arm.rewarm_warmup > 0 and post_step < arm.rewarm_warmup:
        return arm.post_lr * float(post_step + 1) / float(arm.rewarm_warmup)
    return arm.post_lr


@torch.no_grad()
def _global_param_norm(params):
    sq = torch.zeros((), device=params[0].device, dtype=torch.float32)
    for p in params:
        sq += p.detach().float().pow(2).sum()
    return sq.sqrt()


@torch.no_grad()
def _delta_norm(params, snapshot):
    sq = torch.zeros((), device=params[0].device, dtype=torch.float32)
    for p, s in zip(params, snapshot):
        sq += (p.detach().float() - s).pow(2).sum()
    return sq.sqrt()


def _snapshot(params):
    return [p.detach().float().clone() for p in params]


def evaluate(model, gen, tcfg, eval_tasks, n_batches, batch_size, device, rng, perplex=None):
    out = {}
    for task in eval_tasks:
        r = eval_task(model, gen, task, n_batches, batch_size, tcfg.content_ids, device, rng)
        for k, v in r.items():
            out[f"{task}/{k}"] = v
    if perplex is not None:
        from .perplexity import perplexity
        out.update({f"ppl/{k}": v for k, v in perplexity(model, perplex, device).items()})
    return out


def train(model, tcfg, tr, arm, device, eval_tasks=None, perplex_blocks=None, log=print):
    """Run one continued-training arm. Returns the path to the written summary."""
    os.makedirs(tr.out_dir, exist_ok=True)
    rng = np.random.default_rng(tr.seed)
    gen = Generator(tcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    if eval_tasks is None:
        eval_tasks = ["hop1", "hop2_only"]
        if tr.intro_task in ("fresh_hop1", "reverse"):
            eval_tasks.append(tr.intro_task)

    opt = torch.optim.AdamW(params, lr=tr.prereq_lr, betas=(tr.beta1, tr.beta2),
                            weight_decay=tr.weight_decay)

    log_path = os.path.join(tr.out_dir, "log.jsonl")
    logf = open(log_path, "w")
    snap = _snapshot(params)
    cum_update = 0.0
    base_ppl = None

    def do_eval(step, phase, lr):
        nonlocal cum_update, snap, base_ppl
        wnorm = _global_param_norm(params).item()
        dnorm = _delta_norm(params, snap).item()
        ratio = dnorm / max(wnorm, 1e-8)
        cum_update += ratio
        snap = _snapshot(params)
        metrics = evaluate(model, gen, tcfg, eval_tasks, tr.eval_batches, tr.batch_size,
                           device, rng, perplex=perplex_blocks)
        if base_ppl is None and "ppl/ppl" in metrics:
            base_ppl = metrics["ppl/ppl"]
        rec = {"step": step, "phase": phase, "lr": lr, "interval_update_ratio": ratio,
               "cum_update_ratio": cum_update, "weight_norm": wnorm}
        rec.update(metrics)
        if base_ppl is not None and "ppl/ppl" in metrics:
            rec["ppl/delta_from_base"] = metrics["ppl/ppl"] - base_ppl
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        log(f"[{phase} {step}] lr {lr:.2e} cum_upd {cum_update:.3f} "
            f"hop1 {metrics.get('hop1/acc_content', float('nan')):.3f} "
            f"hop2 {metrics.get('hop2_only/excess_content', float('nan')):+.3f}")
        return rec

    # baseline evaluation before any update
    do_eval(0, "base", 0.0)

    # phase one, optional prerequisite training on hop1
    gstep = 0
    for ps in range(tr.prereq_steps):
        gstep += 1
        for g in opt.param_groups:
            g["lr"] = tr.prereq_lr
        _train_step(model, gen, "hop1", tr.batch_size, device, rng, opt, tr.grad_clip)
        if gstep % tr.eval_interval == 0:
            do_eval(gstep, "prereq", tr.prereq_lr)

    # introduction, optional optimiser reset
    if arm.reset_optimizer_at_intro:
        opt = torch.optim.AdamW(params, lr=arm.post_lr, betas=(tr.beta1, tr.beta2),
                                weight_decay=tr.weight_decay)
    snap = _snapshot(params)  # cumulative budget is measured from introduction
    cum_update = 0.0

    # phase two, post-introduction training under the arm rate
    for pstep in range(tr.post_steps):
        gstep += 1
        lr = lr_at(pstep, arm)
        for g in opt.param_groups:
            g["lr"] = lr
        _train_step(model, gen, tr.intro_task, tr.batch_size, device, rng, opt, tr.grad_clip)
        if gstep % tr.eval_interval == 0:
            rec = do_eval(gstep, "post", lr)
            if arm.match_budget_to is not None and rec["cum_update_ratio"] >= arm.match_budget_to:
                log(f"matched budget {arm.match_budget_to:.3f} reached at step {gstep}, stopping")
                break

    logf.close()
    summary = _summarise(log_path, tcfg, tr, arm)
    spath = os.path.join(tr.out_dir, "summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    return spath


def _train_step(model, gen, task, batch_size, device, rng, opt, clip):
    model.train()
    b = gen.batch(task, batch_size, rng)
    ids = torch.as_tensor(b["input_ids"], dtype=torch.long, device=device)
    logits = model(input_ids=ids, use_cache=False).logits
    loss = masked_target_loss(logits, b, device)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()


def _tail(values, frac=0.1):
    n = max(1, int(len(values) * frac))
    return float(np.mean(values[-n:]))


def _summarise(log_path, tcfg, tr, arm):
    recs = [json.loads(l) for l in open(log_path)]
    post = [r for r in recs if r["phase"] == "post"]
    take = post if post else recs
    out = {
        "model": tr.model, "arm": arm.name, "seed": tr.seed, "intro_task": tr.intro_task,
        "post_lr": arm.post_lr, "prereq_steps": tr.prereq_steps,
        "task": tcfg.to_dict(),
        "final_step": recs[-1]["step"] if recs else 0,
    }
    for key in ["hop1/acc_content", "hop2_only/acc_content", "hop2_only/excess_content",
                "hop2_only/floor_acc_content"]:
        if take and key in take[-1]:
            out["tail_" + key.replace("/", "_")] = _tail([r[key] for r in take if key in r])
    if take and "cum_update_ratio" in take[-1]:
        out["final_cum_update_ratio"] = take[-1]["cum_update_ratio"]
    if take and "ppl/delta_from_base" in take[-1]:
        out["tail_ppl_delta_from_base"] = _tail([r["ppl/delta_from_base"] for r in take
                                                 if "ppl/delta_from_base" in r])
    # transition width on hop2 excess, ten to ninety per cent of final
    ex = [(r["step"], r.get("hop2_only/excess_content")) for r in recs
          if r.get("hop2_only/excess_content") is not None]
    out["hop2_transition"] = _transition_width(ex)
    return out


def _transition_width(step_excess):
    if not step_excess:
        return None
    final = step_excess[-1][1]
    if final <= 0.05:
        return {"final_excess": final, "acquired": False}
    lo, hi = 0.1 * final, 0.9 * final
    t10 = next((s for s, e in step_excess if e >= lo), None)
    t90 = next((s for s, e in step_excess if e >= hi), None)
    return {"final_excess": final, "acquired": True, "t10": t10, "t90": t90,
            "width": (t90 - t10) if (t10 is not None and t90 is not None) else None}

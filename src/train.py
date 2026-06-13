"""Continued-training of a late Pythia checkpoint for one experimental arm.

One arm = (hop, schedule, seed). The checkpoint is the maximally-decayed model; we
continue training on the synthetic task under a constant LR (short warmup) set by the
schedule. Crucially we ALWAYS evaluate BOTH Hop-1 and Hop-2 throughout training, so the
curves show selectivity (Hop-1 rises, Hop-2 may not) and recoverability (Hop-2 jumps
under rewarm). Because HF checkpoints carry no optimiser state, our 'rewarm' arm is a
rewarm+RESET by construction -- which the toy showed is equivalent to plain rewarm.
"""
import numpy as np
import torch

from .model_utils import load_model
from .eval import evaluate
from .sharpness import top_hessian_eigenvalue
from .induction import icl_gap_only, induction_track_metrics


def _make_warmup_constant(optimizer, warmup):
    def fn(step):
        return (step + 1) / warmup if step < warmup else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def train_arm(cfg, task, hop, lr, tag, seed, measure_sharpness=False, measure_induction=False,
              ablate_topk=0, ablate_mode="induction", measure_distance=False, return_model=False):
    """One arm = (hop, lr, seed). `tag` is a label used for filenames/grouping.
    measure_sharpness -> track top Hessian eigenvalue (finite-difference HVP).
    measure_induction -> track behavioral ICL gap + attention-based induction score.
    ablate_topk>0     -> functionally knock out k heads of THIS checkpoint before training
                         (causal test; heads stay dead). ablate_mode='induction' cuts the
                         top-k induction heads; 'random' cuts a matched random control set.
    measure_distance  -> track L2 weight movement ||theta_t - theta_0|| each eval.
    return_model      -> keep the trained model + theta_0 snapshot in the result
                         (for the interpolation/mode-connectivity command)."""
    torch.manual_seed(seed)
    model = load_model(cfg, dtype=torch.float32)  # sdpa, same as the sweep; FD-HVP needs no eager

    ablated = []
    abl_handles = []
    if ablate_topk and ablate_topk > 0:
        from .ablation import select_ablation_heads, apply_head_ablation
        ablated, ranked = select_ablation_heads(model, cfg, ablate_topk, mode=ablate_mode, seed=seed)
        abl_handles = apply_head_ablation(model, ablated)
        ind_top = [lh for (lh, _sc) in ranked[:ablate_topk]]
        print(f"    [ablate:{ablate_mode}] knocking out {ablated}  "
              f"(induction top-{ablate_topk}={ind_top})")
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=cfg.adam_betas, eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    sch = _make_warmup_constant(opt, cfg.warmup_steps)
    rng = np.random.default_rng(seed)

    theta0 = theta0_norm = None
    if measure_distance or return_model:
        theta0 = [p.detach().clone() for p in model.parameters()]
        theta0_norm = float(torch.sqrt(sum((p0.pow(2).sum() for p0 in theta0))).item())

    s_input = s_labels = None
    if measure_sharpness:
        srng = np.random.default_rng(cfg.eval_seed + 999)
        sb = task.batch(cfg.sharpness_batch, hop, srng)   # fixed batch, comparable across steps/LRs
        s_input = sb["input_ids"].to(cfg.device)
        s_labels = sb["labels"].to(cfg.device)

    total_steps = cfg.steps_for(hop)
    curve = []
    for step in range(total_steps):
        b = task.batch(cfg.batch_size, hop, rng)
        input_ids = b["input_ids"].to(cfg.device)
        labels = b["labels"].to(cfg.device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sch.step()

        if (step % cfg.eval_every == 0) or (step == total_steps - 1):
            e1 = evaluate(model, task, cfg, cfg.train_eval_batches, hop=1)
            e2 = evaluate(model, task, cfg, cfg.train_eval_batches, hop=2)
            point = dict(
                step=step, loss=float(loss.item()),
                hop1_acc=e1["acc"], hop1_excess=e1["excess"],
                hop2_acc=e2["acc"], hop2_excess=e2["excess"],
            )
            if measure_sharpness:
                lam = top_hessian_eigenvalue(model, s_input, s_labels,
                                             n_iter=cfg.sharpness_n_iter,
                                             eps_rel=cfg.sharpness_eps_rel)
                point["lambda_max"] = lam
                point["eta_lambda"] = lr * lam
            if measure_induction:
                icl, mh, t5 = induction_track_metrics(model, cfg, ablated_heads=ablated)
                point["icl_gap"] = icl
                point["max_head_induction"] = mh
                point["top5_head_induction"] = t5
            if measure_distance:
                d2 = sum(((p.detach() - p0).pow(2).sum() for p, p0 in zip(model.parameters(), theta0)))
                wd = float(torch.sqrt(d2).item())
                point["weight_dist"] = wd
                point["weight_dist_rel"] = wd / theta0_norm if theta0_norm else float("nan")
            curve.append(point)
            model.train()
            extra = f"  lam {point['lambda_max']:.1f}  eta*lam {point['eta_lambda']:.2f}" if measure_sharpness else ""
            extra += f"  icl {point['icl_gap']:+.2f} maxhd {point['max_head_induction']:.2f}" if measure_induction else ""
            extra += f"  ||dθ|| {point['weight_dist']:.2f}" if measure_distance else ""
            print(
                f"    [{tag:>12} hop{hop} s{seed}] step {step:>5}  loss {loss.item():7.3f}  "
                f"h1_acc {e1['acc']:.3f}  h2_acc {e2['acc']:.3f}{extra}"
            )

    # Final, larger eval. Logit lens computed on Hop-2 (the interesting mode).
    f1 = evaluate(model, task, cfg, cfg.final_eval_batches, hop=1)
    f2 = evaluate(model, task, cfg, cfg.final_eval_batches, hop=2, do_lens=True)

    result = dict(
        hop=hop, schedule=tag, tag=tag, seed=seed, lr=lr,
        curve=curve, final_hop1=f1, final_hop2=f2,
        ablated_heads=ablated,
    )
    if return_model:
        result["model"] = model
        result["theta0"] = theta0
        return result
    for h in abl_handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return result

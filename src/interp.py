"""Linear-interpolation (mode-connectivity) probe for the parameter-distance hypothesis.

Given the starting checkpoint weights theta0 and the post-acquisition weights theta1, walk
the straight line theta(a) = (1-a) theta0 + a theta1 and evaluate the Hop-2 loss/accuracy at
each a. The shape of that path operationalises "reachability":

  monotone (no bump)         -> the solution is essentially downhill from the checkpoint;
                                low effective distance, linearly mode-connected. Consistent
                                with the recoverable / no-irreversibility reading.
  loss bump / accuracy dip   -> the optimiser had to cross a barrier to reach the solution;
                                the signature a genuinely hard critical period would leave.

Also supports interpolating between TWO acquired solutions (e.g. from step512 vs step1000)
to ask whether different starting points land in the same basin (no barrier) or different
ones (barrier) -- the cross-checkpoint path-dependence test.

Weights are saved/loaded as ordered parameter lists (model.parameters() order), so loading
is index-consistent without relying on state_dict key matching.
"""
import numpy as np
import torch

from .eval import evaluate
from .induction import induction_track_metrics


def save_param_list(params, path):
    torch.save([p.detach().cpu() for p in params], path)


def load_param_list(path, device):
    return [t.to(device) for t in torch.load(path, map_location=device)]


def snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


@torch.no_grad()
def _set_params(model, vecs):
    for p, v in zip(model.parameters(), vecs):
        p.copy_(v)


@torch.no_grad()
def _hop2_loss(model, task, cfg, n_batches, seed_offset=7):
    rng = np.random.default_rng(cfg.eval_seed + seed_offset)
    ls = []
    for _ in range(n_batches):
        b = task.batch(cfg.batch_size, 2, rng)
        ids = b["input_ids"].to(cfg.device)
        lab = b["labels"].to(cfg.device)
        ls.append(float(model(input_ids=ids, labels=lab).loss.item()))
    return float(np.mean(ls))


def interpolate_eval(model, task, cfg, theta0, theta1, alphas, label=""):
    """Evaluate Hop-1/Hop-2 accuracy, Hop-2 loss, and general induction along the line."""
    rows = []
    for a in alphas:
        _set_params(model, [(1 - a) * p0 + a * p1 for p0, p1 in zip(theta0, theta1)])
        model.eval()
        e1 = evaluate(model, task, cfg, cfg.train_eval_batches, hop=1)
        e2 = evaluate(model, task, cfg, cfg.train_eval_batches, hop=2)
        h2loss = _hop2_loss(model, task, cfg, cfg.train_eval_batches)
        icl, mh, _ = induction_track_metrics(model, cfg)
        rows.append(dict(alpha=float(a), hop1_acc=e1["acc"], hop2_acc=e2["acc"],
                         hop2_loss=h2loss, icl_gap=icl, max_head_induction=mh))
        print(f"    [interp {label}] a={a:.2f}  h2_acc {e2['acc']:.3f}  "
              f"h2_loss {h2loss:.3f}  icl {icl:+.2f}  maxhd {mh:.2f}")
    return rows


def barrier_metrics(rows):
    """Quantify the path. loss_barrier > 0 means an interior loss bump above the worse
    endpoint; acc_dip > 0 means interior accuracy falls below the worse endpoint."""
    a = np.array([r["alpha"] for r in rows])
    loss = np.array([r["hop2_loss"] for r in rows])
    acc = np.array([r["hop2_acc"] for r in rows])
    interior = (a > 0) & (a < 1)
    end_loss = max(loss[0], loss[-1])
    end_acc = min(acc[0], acc[-1])
    loss_barrier = float(loss[interior].max() - end_loss) if interior.any() else 0.0
    acc_dip = float(end_acc - acc[interior].min()) if interior.any() else 0.0
    return dict(loss_barrier=loss_barrier, acc_dip=acc_dip,
                endpoint_loss=[float(loss[0]), float(loss[-1])],
                endpoint_acc=[float(acc[0]), float(acc[-1])])

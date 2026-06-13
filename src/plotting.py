"""Plotting helpers. Pure matplotlib (Agg), no seaborn."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_induction(rows, out_path):
    rows = sorted(rows, key=lambda r: r["step"])
    steps = [max(r["step"], 0.5) for r in rows]  # 0 -> 0.5 so it shows on log axis
    icl = [r["icl_gap"] for r in rows]
    sec = [r["second_loss"] for r in rows]
    mh = [r["max_head_induction"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, icl, "o-", label="ICL gap (first - second loss)")
    ax[0].plot(steps, sec, "s--", color="gray", label="second-copy loss")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel("nats")
    ax[0].set_title("In-context copying forms")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    if not all(np.isnan(mh)):
        ax[1].plot(steps, mh, "o-", color="C3", label="max-head induction score")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("training step")
    ax[1].set_ylabel("attention to induction position")
    ax[1].set_title("Strongest induction head")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_curves(curves_by_group, out_path):
    """curves_by_group[(hop, schedule)] = dict(steps, mean, std) of hop-matching excess."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    titles = {1: "Hop-1 (lookup primitive) excess", 2: "Hop-2 (composition) excess"}
    colors = {"native_low": "C0", "deep_low": "C1", "rewarm": "C2"}
    for col, hop in enumerate((1, 2)):
        a = ax[col]
        for (h, sch), d in sorted(curves_by_group.items()):
            if h != hop:
                continue
            steps = np.array(d["steps"])
            mean = np.array(d["mean"])
            std = np.array(d["std"])
            c = colors.get(sch, None)
            a.plot(steps, mean, "-o", ms=3, color=c, label=sch)
            a.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15)
        a.axhline(0.0, color="k", lw=0.8, ls=":")
        a.set_title(titles[hop])
        a.set_xlabel("continued-training step")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("accuracy - floor (excess)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sweep(points, out_path, lens_points=None):
    """points = [{lr, mean, std}] of final Hop-2 excess; optional lens_points = [{lr, C}]."""
    points = sorted(points, key=lambda p: p["lr"])
    lrs = [p["lr"] for p in points]
    mean = [p["mean"] for p in points]
    std = [p["std"] for p in points]
    ncol = 2 if lens_points else 1
    fig, ax = plt.subplots(1, ncol, figsize=(11 if lens_points else 6, 4), squeeze=False)
    a = ax[0][0]
    a.errorbar(lrs, mean, yerr=std, fmt="o-", capsize=3, color="C0")
    a.set_xscale("log")
    a.axhline(0.0, color="k", ls=":", lw=0.8)
    a.set_xlabel("continued-training learning rate")
    a.set_ylabel("final Hop-2 excess (acc - floor)")
    a.set_title("Composition forms only within an LR band")
    a.grid(alpha=0.3)
    if lens_points:
        lp = sorted(lens_points, key=lambda p: p["lr"])
        a2 = ax[0][1]
        a2.plot([p["lr"] for p in lp], [p["C"] for p in lp], "o-", color="C3")
        a2.set_xscale("log")
        a2.set_xlabel("continued-training learning rate")
        a2.set_ylabel("max Hop-2 answer-C logit lens")
        a2.set_title("Reorganisation vs LR")
        a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sharpness(curves_by_lr, out_path):
    """curves_by_lr[lr] = dict(steps, lambda_max, eta_lambda, h2_acc)."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for lr, d in sorted(curves_by_lr.items()):
        steps = np.array(d["steps"])
        lab = f"{lr:g}"
        ax[0].plot(steps, d["lambda_max"], "-o", ms=3, label=lab)
        ax[1].plot(steps, d["eta_lambda"], "-o", ms=3, label=lab)
        ax[2].plot(steps, d["h2_acc"], "-o", ms=3, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_title("Top Hessian eigenvalue (sharpness)")
    ax[0].set_ylabel("lambda_max")
    ax[1].axhline(2.0, color="k", ls="--", lw=1, label="GD edge (2/eta)")
    ax[1].set_title("eta * lambda_max  (Adam threshold differs)")
    ax[1].set_ylabel("eta * lambda_max")
    ax[2].axhline(0.167, color="gray", ls=":", lw=0.8)
    ax[2].set_title("Hop-2 accuracy (for alignment)")
    ax[2].set_ylabel("accuracy")
    for a in ax:
        a.set_xlabel("continued-training step")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_lens(lens_by_schedule, out_path):
    """lens_by_schedule[schedule] = dict(C=[per-layer], B=[per-layer])  (Hop-2)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"native_low": "C0", "deep_low": "C1", "rewarm": "C2"}
    for sch, d in sorted(lens_by_schedule.items()):
        layers = np.arange(len(d["C"]))
        ax[0].plot(layers, d["C"], "-o", ms=3, color=colors.get(sch), label=sch)
        if "B" in d:
            ax[1].plot(layers, d["B"], "-o", ms=3, color=colors.get(sch), label=sch)
    ax[0].set_title("Answer C decodable by logit lens (Hop-2)")
    ax[1].set_title("Intermediate B decodable by logit lens (Hop-2)")
    for a in ax:
        a.set_xlabel("layer (0 = embeddings)")
        a.set_ylabel("candidate-restricted decode acc")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

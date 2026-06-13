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


def plot_track(steps, icl_gap, max_head, hop2_acc, out_path, title="continued training"):
    """Two convergent probes of GENERAL induction across continued training.
    Panel A (mechanistic): strongest induction-head attention vs composition accuracy
      -- both 0-1, so a clean overlay; if the prefix-match HEAD is destroyed/repurposed
      as the composition forms, this line falls.
    Panel B (behavioral): ICL gap (nats) vs composition accuracy."""
    steps = np.array(steps)
    mh = np.array(max_head, dtype=float)
    have_attn = not np.all(np.isnan(mh))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Panel A: attention-level induction (0-1) vs composition (0-1) ---
    if have_attn:
        axA.plot(steps, mh, "-o", ms=3, color="C3",
                 label="max-head induction attention")
    axA.plot(steps, hop2_acc, "-s", ms=3, color="C0", label="Hop-2 acc (composition)")
    axA.axhline(0.167, color="gray", ls=":", lw=0.8)
    axA.set_ylim(-0.05, 1.05)
    axA.set_xlabel("continued-training step")
    axA.set_ylabel("attention score / accuracy")
    axA.set_title("Attention-level induction vs composition"
                  if have_attn else "Attention unavailable (SDPA refused attns)")
    axA.grid(alpha=0.3)
    axA.legend(fontsize=8, loc="center right")

    # --- Panel B: behavioral ICL gap (nats) vs composition (twin axis) ---
    axB.plot(steps, icl_gap, "-o", ms=3, color="C3", label="ICL gap (nats)")
    axB.set_xlabel("continued-training step")
    axB.set_ylabel("ICL gap (nats)", color="C3")
    axB.tick_params(axis="y", labelcolor="C3")
    axB.axhline(0.0, color="gray", ls=":", lw=0.8)
    axB.grid(alpha=0.3)
    axB2 = axB.twinx()
    axB2.plot(steps, hop2_acc, "-s", ms=3, color="C0", label="Hop-2 acc")
    axB2.set_ylabel("Hop-2 accuracy", color="C0")
    axB2.tick_params(axis="y", labelcolor="C0")
    axB2.set_ylim(-0.02, 1.02)
    axB.set_title("Behavioral induction (ICL gap) vs composition")

    fig.suptitle(f"Does the composition reuse or cannibalize induction? ({title})")
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


def plot_distance(arms, out_path, jump_thresh=0.6):
    """Weight movement vs the parameter-distance hypothesis.
    Panel A: ||theta_t - theta_0|| over training, one line per LR, with a star at the
      Hop-2 acquisition step -- does acquisition track a characteristic distance, and does
      a blocked (high-LR) arm travel FAR without acquiring (distance not sufficient)?
    Panel B: Hop-2 accuracy vs distance traversed -- if acquisition happens at a common
      distance across LRs, the curves stack vertically at the same x."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.5))
    for a in sorted(arms, key=lambda r: r["lr"]):
        c = a["curve"]
        steps = np.array([p["step"] for p in c])
        dist = np.array([p.get("weight_dist", np.nan) for p in c])
        acc = np.array([p["hop2_acc"] for p in c])
        lab = f"lr={a['lr']:g}"
        line, = axA.plot(steps, dist, "-o", ms=2.5, label=lab)
        crossed = np.where(acc >= jump_thresh)[0]
        if crossed.size:
            j = crossed[0]
            axA.plot(steps[j], dist[j], "*", ms=14, color=line.get_color())
        axB.plot(dist, acc, "-o", ms=2.5, color=line.get_color(), label=lab)
    axA.set_xlabel("continued-training step")
    axA.set_ylabel(r"$\|\theta_t-\theta_0\|_2$")
    axA.set_title("Weight movement (star = Hop-2 acquisition)")
    axA.grid(alpha=0.3)
    axA.legend(fontsize=8)
    axB.axhline(0.167, color="gray", ls=":", lw=0.8)
    axB.set_xlabel(r"distance traversed $\|\theta_t-\theta_0\|_2$")
    axB.set_ylabel("Hop-2 accuracy")
    axB.set_ylim(-0.02, 1.02)
    axB.set_title("Does acquisition happen at a characteristic distance?")
    axB.grid(alpha=0.3)
    axB.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_interp(rows, out_path, title="", barrier=None):
    """Linear-interpolation path between theta_0 (a=0) and theta_final (a=1)."""
    a = np.array([r["alpha"] for r in rows])
    loss = np.array([r["hop2_loss"] for r in rows])
    acc = np.array([r["hop2_acc"] for r in rows])
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(a, loss, "-o", ms=4, color="C3", label="Hop-2 loss")
    ax1.set_xlabel(r"interpolation $\alpha$  ($\theta_0 \to \theta_{\rm final}$)")
    ax1.set_ylabel("Hop-2 loss (nats)", color="C3")
    ax1.tick_params(axis="y", labelcolor="C3")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(a, acc, "-s", ms=4, color="C0", label="Hop-2 acc")
    ax2.axhline(0.167, color="gray", ls=":", lw=0.8)
    ax2.set_ylabel("Hop-2 accuracy", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax2.set_ylim(-0.02, 1.02)
    sub = ""
    if barrier is not None:
        sub = f"  (loss barrier {barrier['loss_barrier']:+.2f} nats, acc dip {barrier['acc_dip']:+.2f})"
    ax1.set_title(f"Interpolation path {title}{sub}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_generalize(rows, out_path, title="", in_dist_acc=None):
    """Horizontal bars of accuracy per OOD variant, with the per-variant chance floor marked.
    Near 1.0 => the systematic algorithm transfers; near the floor => a distribution-specific
    heuristic that doesn't generalize on that axis."""
    rows = list(rows)
    names = [r["name"] for r in rows]
    acc = np.array([r["acc"] for r in rows])
    floor = np.array([r["floor"] for r in rows])
    y = np.arange(len(rows))[::-1]            # first row at top
    colors = ["#3b7dd8" if r.get("hop", 2) == 2 else "#7a7a7a" for r in rows]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(rows) + 1.2)))
    ax.barh(y, acc, color=colors, height=0.6, zorder=2)
    ax.scatter(floor, y, marker="|", s=400, color="crimson", zorder=3, label="chance floor")
    for yi, a in zip(y, acc):
        ax.text(min(a + 0.02, 0.98), yi, f"{a:.2f}", va="center", fontsize=8)
    if in_dist_acc is not None:
        ax.axvline(in_dist_acc, color="green", ls="--", lw=1, label=f"in-dist acc {in_dist_acc:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("accuracy (full-vocab argmax == answer)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

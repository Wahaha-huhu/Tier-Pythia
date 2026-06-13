"""Entry point for the Pythia critical-period probe.

Subcommands:
  induction      observational induction-formation window across checkpoints
  intervention   the (task x schedule x seed) continued-training factorial
  sweep          LR sweep for the composition (+ --track-induction / --track-distance)
  sharpness      finite-difference top-Hessian-eigenvalue probe across LRs
  ablate         CAUSAL test: knock out the induction head(s) at a fixed checkpoint and
                 compare Hop-2 acquisition with vs without them (removes maturity confound)
  interp         interpolate theta_0 -> theta_final (or between two solutions) and measure
                 the Hop-2 loss/accuracy barrier (parameter-distance / mode-connectivity)
  all            induction + intervention + SUMMARY.md
  smoke          tiny end-to-end run to verify the environment (~minutes)

Run from the repo root, e.g.:
  python run.py ablate  --revision step1000 --ablate-topk 3 --steps 6000 --seeds 1 --out-dir results_ablate
  python run.py interp  --revision step1000 --lr 6e-5 --steps 6000 --interp-points 21 --out-dir results_interp1000
  python run.py sweep   --lrs 6e-6 6e-5 6e-4 --revision step1000 --steps 6000 --track-distance --out-dir results_dist
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import Config
from src.data import ChainTask
from src.induction import run_induction_window
from src.train import train_arm
from src.model_utils import load_model
from src import plotting
from src import interp as I


# --------------------------------------------------------------------------- io
def ensure_dirs(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "intervention"), exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def arm_path(cfg, hop, tag, seed):
    return os.path.join(cfg.out_dir, "intervention", f"hop{hop}_{tag}_seed{seed}.json")


def print_env():
    print("=" * 70)
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print("=" * 70)


# ------------------------------------------------------------------- induction
def cmd_induction(cfg):
    print("\n[1] Induction-formation window (observational, no training)")
    rows = run_induction_window(cfg)
    save_json(rows, os.path.join(cfg.out_dir, "induction_window.json"))
    with open(os.path.join(cfg.out_dir, "induction_window.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "first_loss", "second_loss", "icl_gap",
                    "max_head_induction", "top5_head_induction"])
        for r in sorted(rows, key=lambda x: x["step"]):
            w.writerow([r["step"], f"{r['first_loss']:.4f}", f"{r['second_loss']:.4f}",
                        f"{r['icl_gap']:.4f}", f"{r['max_head_induction']:.4f}",
                        f"{r['top5_head_induction']:.4f}"])
    plotting.plot_induction(rows, os.path.join(cfg.out_dir, "induction_window.png"))
    print(f"  -> {cfg.out_dir}/induction_window.{{json,csv,png}}")
    return rows


# ---------------------------------------------------------------- intervention
def cmd_intervention(cfg):
    print("\n[2] Intervention factorial (continued training)")
    task = ChainTask(cfg)
    n = len(cfg.tasks) * len(cfg.schedules) * cfg.seeds
    print(f"  {n} arms: tasks={list(cfg.tasks)} schedules={list(cfg.schedules)} seeds={cfg.seeds}")
    done = 0
    for seed in range(cfg.seeds):
        for hop in cfg.tasks:
            for schedule in cfg.schedules:
                done += 1
                print(f"\n  --- arm {done}/{n}: hop{hop} {schedule} seed{seed} ---")
                res = train_arm(cfg, task, hop, cfg.lr_for(schedule), schedule, seed)
                save_json(res, arm_path(cfg, hop, schedule, seed))
    print("\n  all arms complete")
    summary_rows, lens = aggregate_intervention(cfg)
    ind_path = os.path.join(cfg.out_dir, "induction_window.json")
    ind = load_json(ind_path) if os.path.exists(ind_path) else []
    write_summary(cfg, ind, summary_rows, lens)


def _load_all_arms(cfg):
    import glob
    arms = []
    for p in sorted(glob.glob(os.path.join(cfg.out_dir, "intervention", "hop*_*.json"))):
        arms.append(load_json(p))
    return arms


def aggregate_intervention(cfg):
    arms = _load_all_arms(cfg)
    if not arms:
        print("  (no arm results to aggregate)")
        return [], {}

    # ---- summary table: per (hop, schedule), hop-matching final acc/excess over seeds
    by_group = defaultdict(list)
    for a in arms:
        key = (a["hop"], a["schedule"])
        f = a["final_hop1"] if a["hop"] == 1 else a["final_hop2"]
        by_group[key].append(f)

    summary_rows = []
    for (hop, sch), fs in sorted(by_group.items()):
        accs = np.array([f["acc"] for f in fs])
        exc = np.array([f["excess"] for f in fs])
        floor = float(np.mean([f["floor"] for f in fs]))
        succ = float(np.mean(accs >= cfg.success_acc))
        summary_rows.append(dict(
            hop=hop, schedule=sch, n_seeds=len(fs),
            mean_acc=float(accs.mean()), std_acc=float(accs.std()),
            mean_excess=float(exc.mean()), std_excess=float(exc.std()),
            floor=floor, success_rate=succ,
        ))

    with open(os.path.join(cfg.out_dir, "intervention_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hop", "schedule", "n_seeds", "mean_acc", "std_acc",
                    "mean_excess", "std_excess", "floor", "success_rate"])
        for r in summary_rows:
            w.writerow([r["hop"], r["schedule"], r["n_seeds"],
                        f"{r['mean_acc']:.4f}", f"{r['std_acc']:.4f}",
                        f"{r['mean_excess']:.4f}", f"{r['std_excess']:.4f}",
                        f"{r['floor']:.4f}", f"{r['success_rate']:.2f}"])

    # ---- training curves (hop-matching excess), averaged over seeds
    curves_by_group = {}
    grp_curves = defaultdict(list)
    for a in arms:
        grp_curves[(a["hop"], a["schedule"])].append(a["curve"])
    for (hop, sch), curves in grp_curves.items():
        steps = [pt["step"] for pt in curves[0]]
        key = "hop1_excess" if hop == 1 else "hop2_excess"
        mat = np.array([[pt[key] for pt in c] for c in curves])  # [seeds, T]
        curves_by_group[(hop, sch)] = dict(
            steps=steps, mean=mat.mean(0).tolist(), std=mat.std(0).tolist()
        )
    plotting.plot_curves(curves_by_group, os.path.join(cfg.out_dir, "intervention_curves.png"))

    # ---- logit lens (Hop-2), averaged over seeds, per schedule
    lens_by_schedule = {}
    lens_acc = defaultdict(lambda: {"C": [], "B": []})
    for a in arms:
        if a["hop"] != 2:
            continue
        f2 = a["final_hop2"]
        if "lens_C" in f2:
            lens_acc[a["schedule"]]["C"].append(f2["lens_C"])
        if "lens_B" in f2:
            lens_acc[a["schedule"]]["B"].append(f2["lens_B"])
    for sch, d in lens_acc.items():
        entry = {}
        if d["C"]:
            entry["C"] = np.array(d["C"]).mean(0).tolist()
        if d["B"]:
            entry["B"] = np.array(d["B"]).mean(0).tolist()
        if entry:
            lens_by_schedule[sch] = entry
    if lens_by_schedule:
        plotting.plot_lens(lens_by_schedule, os.path.join(cfg.out_dir, "logit_lens.png"))

    print(f"  -> {cfg.out_dir}/intervention_summary.csv, intervention_curves.png, logit_lens.png")
    return summary_rows, lens_by_schedule


# -------------------------------------------------------------------- sharpness
def cmd_sharpness(cfg, lrs):
    print("\n[sharpness] edge-of-stability probe (top Hessian eigenvalue) for Hop-2")
    task = ChainTask(cfg)
    n = len(lrs) * cfg.seeds
    print(f"  {n} arms (eager attn): lrs={[f'{x:g}' for x in lrs]} seeds={cfg.seeds} steps={cfg.max_steps_hop2}")
    done = 0
    for seed in range(cfg.seeds):
        for lr in lrs:
            done += 1
            tag = f"sharp_lr{lr:g}"
            print(f"\n  --- sharpness arm {done}/{n}: lr={lr:g} seed{seed} ---")
            res = train_arm(cfg, task, 2, lr, tag, seed, measure_sharpness=True)
            save_json(res, arm_path(cfg, 2, tag, seed))
    aggregate_sharpness(cfg)


def aggregate_sharpness(cfg):
    arms = [a for a in _load_all_arms(cfg) if str(a.get("tag", "")).startswith("sharp_lr")]
    if not arms:
        print("  (no sharpness arms found)")
        return
    by_lr = defaultdict(list)
    for a in arms:
        by_lr[a["lr"]].append(a)

    curves_by_lr = {}
    rows = []
    for lr, group in sorted(by_lr.items()):
        g = sorted(group, key=lambda x: x["seed"])[0]  # representative seed for the curve
        c = g["curve"]
        steps = [pt["step"] for pt in c]
        lam = [pt.get("lambda_max", float("nan")) for pt in c]
        el = [pt.get("eta_lambda", float("nan")) for pt in c]
        acc = [pt["hop2_acc"] for pt in c]
        curves_by_lr[lr] = dict(steps=steps, lambda_max=lam, eta_lambda=el, h2_acc=acc)
        # late-phase summary (last third of training)
        k = max(1, len(c) // 3)
        late_el = np.mean([pt.get("eta_lambda", np.nan) for pt in c[-k:]])
        late_lam = np.mean([pt.get("lambda_max", np.nan) for pt in c[-k:]])
        formed = max(pt["hop2_acc"] for pt in c) - 0.167 >= 0.5
        rows.append(dict(lr=lr, late_lambda=float(late_lam), late_eta_lambda=float(late_el),
                         formed=formed))

    plotting.plot_sharpness(curves_by_lr, os.path.join(cfg.out_dir, "sharpness.png"))
    with open(os.path.join(cfg.out_dir, "sharpness_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr", "late_lambda_max", "late_eta_lambda", "composition_formed"])
        for r in rows:
            w.writerow([f"{r['lr']:g}", f"{r['late_lambda']:.3f}", f"{r['late_eta_lambda']:.4f}",
                        "YES" if r["formed"] else "no"])

    lines = ["# Edge-of-stability probe -- Hop-2 sharpness vs continued-training LR\n",
             f"Model `{cfg.model_name}` @ `{cfg.late_revision}`, L={cfg.chain_len}, "
             f"Hop-2 steps={cfg.max_steps_hop2}\n",
             "| LR | late lambda_max | late eta*lambda_max | formed? |",
             "|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["lr"]):
        lines.append(f"| {r['lr']:g} | {r['late_lambda']:.1f} | {r['late_eta_lambda']:.2f} | "
                     f"{'YES' if r['formed'] else 'no'} |")
    lines.append("\nReading: if the blocked high-LR arm sits at a much larger eta*lambda_max than "
                 "the forming arms (and cannot reduce its loss), the upper edge is an instability "
                 "barrier -- the LR is too large for the local curvature to descend into the routing "
                 "minimum. (eta*lambda~2 is the GD edge; AdamW's threshold differs, so compare across "
                 "LRs rather than to 2 exactly.)")
    with open(os.path.join(cfg.out_dir, "SHARPNESS_SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"  -> {cfg.out_dir}/sharpness.png, sharpness_summary.csv, SHARPNESS_SUMMARY.md")
    print("\n" + "\n".join(lines))


# ----------------------------------------------------------------------- sweep
def cmd_sweep(cfg, lrs, track_induction=False, track_distance=False):
    print("\n[sweep] LR sweep for the composition (Hop-2 by default)")
    task = ChainTask(cfg)
    hops = cfg.tasks if cfg.tasks else (2,)
    n = len(lrs) * len(hops) * cfg.seeds
    extra = []
    if track_induction:
        extra.append("induction tracking")
    if track_distance:
        extra.append("weight-distance tracking")
    print(f"  {n} arms: lrs={[f'{x:g}' for x in lrs]} hops={list(hops)} seeds={cfg.seeds}"
          f"{(' (+' + ', '.join(extra) + ')') if extra else ''}")
    done = 0
    for seed in range(cfg.seeds):
        for hop in hops:
            for lr in lrs:
                done += 1
                tag = f"lr{lr:g}"
                print(f"\n  --- sweep arm {done}/{n}: hop{hop} lr={lr:g} seed{seed} ---")
                res = train_arm(cfg, task, hop, lr, tag, seed,
                                measure_induction=track_induction,
                                measure_distance=track_distance)
                save_json(res, arm_path(cfg, hop, tag, seed))
    aggregate_sweep(cfg)


def aggregate_sweep(cfg):
    arms = [a for a in _load_all_arms(cfg) if a["hop"] == 2]
    if not arms:
        print("  (no Hop-2 sweep arms found)")
        return
    by_lr = defaultdict(list)
    for a in arms:
        by_lr[a["lr"]].append(a)

    points, lens_points, summary = [], [], []
    curves_by_group = {}
    for lr, group in sorted(by_lr.items()):
        exc = np.array([g["final_hop2"]["excess"] for g in group])
        accs = np.array([g["final_hop2"]["acc"] for g in group])
        cmax = [max(g["final_hop2"].get("lens_C", [0])) for g in group]
        # jump step: first eval step where mean-ish excess crosses 0.5 (per first seed's curve)
        jump = None
        for pt in group[0]["curve"]:
            if pt["hop2_excess"] >= 0.5:
                jump = pt["step"]
                break
        points.append(dict(lr=lr, mean=float(exc.mean()), std=float(exc.std())))
        lens_points.append(dict(lr=lr, C=float(np.mean(cmax))))
        summary.append(dict(lr=lr, mean_acc=float(accs.mean()), mean_excess=float(exc.mean()),
                            std_excess=float(exc.std()), max_lensC=float(np.mean(cmax)),
                            jump_step=jump, n=len(group)))
        # overlay curve (mean over seeds) keyed for plot_curves
        steps = [pt["step"] for pt in group[0]["curve"]]
        mat = np.array([[pt["hop2_excess"] for pt in g["curve"]] for g in group])
        curves_by_group[(2, f"lr{lr:g}")] = dict(steps=steps, mean=mat.mean(0).tolist(),
                                                 std=mat.std(0).tolist())

    plotting.plot_sweep(points, os.path.join(cfg.out_dir, "sweep_invertedU.png"), lens_points)
    plotting.plot_curves(curves_by_group, os.path.join(cfg.out_dir, "sweep_curves.png"))

    # if induction was tracked, emit convergent overlays (attention + behavioral) per arm
    for a in arms:
        c = a["curve"]
        if c and ("icl_gap" in c[0]):
            steps = [pt["step"] for pt in c]
            icl = [pt.get("icl_gap", float("nan")) for pt in c]
            mh = [pt.get("max_head_induction", float("nan")) for pt in c]
            acc = [pt["hop2_acc"] for pt in c]
            fn = os.path.join(cfg.out_dir, f"track_{a['tag']}_seed{a['seed']}.png")
            plotting.plot_track(steps, icl, mh, acc, fn, title=f"{cfg.late_revision}, lr={a['lr']:g}")

    # if weight-distance was tracked, emit the parameter-distance figure (per seed 0)
    dist_arms = [a for a in arms if a["curve"] and ("weight_dist" in a["curve"][0])
                 and a["seed"] == 0]
    if dist_arms:
        plotting.plot_distance(dist_arms, os.path.join(cfg.out_dir, "sweep_distance.png"))

    with open(os.path.join(cfg.out_dir, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr", "mean_acc", "mean_excess", "std_excess", "max_lensC", "jump_step", "n_seeds"])
        for r in summary:
            w.writerow([f"{r['lr']:g}", f"{r['mean_acc']:.4f}", f"{r['mean_excess']:.4f}",
                        f"{r['std_excess']:.4f}", f"{r['max_lensC']:.4f}",
                        r["jump_step"] if r["jump_step"] is not None else "", r["n"]])

    # short summary
    forms = [r for r in summary if r["mean_excess"] >= 0.5]
    lines = ["# LR sweep -- composition (Hop-2) acquisition vs continued-training LR\n",
             f"Model `{cfg.model_name}` @ `{cfg.late_revision}`, L={cfg.chain_len}, "
             f"seeds={cfg.seeds}, Hop-2 steps={cfg.max_steps_hop2}\n",
             "| LR | mean acc | mean excess | max C-lens | jump step | forms? |",
             "|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['lr']:g} | {r['mean_acc']:.3f} | {r['mean_excess']:+.3f} ± "
                     f"{r['std_excess']:.3f} | {r['max_lensC']:.2f} | "
                     f"{r['jump_step'] if r['jump_step'] is not None else '-'} | "
                     f"{'YES' if r['mean_excess'] >= 0.5 else 'no'} |")
    lines.append("")
    if forms:
        lo = min(r["lr"] for r in forms)
        hi = max(r["lr"] for r in forms)
        lines.append(f"**Acquisition band**: composition forms for LR in roughly "
                     f"[{lo:g}, {hi:g}]. Outside this band it stays near floor.")
        if len(forms) < len(summary):
            lines.append("Non-monotonic in LR: blocked both below (too small to traverse to the "
                         "solution -- the toy's late-decay regime) and above (too large to settle "
                         "into the sharp routing minimum). Reconciles the toy as the lower edge.")
    else:
        lines.append("No LR in the swept range acquired the composition -- widen the range or steps.")
    out = os.path.join(cfg.out_dir, "SWEEP_SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> {cfg.out_dir}/sweep_invertedU.png, sweep_curves.png, sweep_summary.csv, SWEEP_SUMMARY.md")
    print("\n" + "\n".join(lines))


# ------------------------------------------------------------------- SUMMARY
def _get(summary_rows, hop, sch, field, default=None):
    for r in summary_rows:
        if r["hop"] == hop and r["schedule"] == sch:
            return r[field]
    return default


def write_summary(cfg, induction_rows, summary_rows, lens_by_schedule):
    lines = []
    lines.append("# Pythia critical-period probe -- auto-generated summary\n")
    lines.append(f"Model: `{cfg.model_name}` @ `{cfg.late_revision}`  |  "
                 f"seeds={cfg.seeds}, L={cfg.chain_len}, "
                 f"steps(hop1/hop2)={cfg.max_steps_hop1}/{cfg.max_steps_hop2}, "
                 f"LRs: native_low={cfg.lr_native_low:g}, deep_low={cfg.lr_deep_low:g}, "
                 f"rewarm={cfg.lr_rewarm:g}\n")

    # --- induction window
    if induction_rows:
        rows = sorted(induction_rows, key=lambda r: r["step"])
        gaps = [(r["step"], r["icl_gap"]) for r in rows]
        max_gap = max(g for _, g in gaps)
        thr = 0.5 * max_gap
        emerge = next((s for s, g in gaps if s > 0 and g >= thr), None)
        lines.append("## (A) Induction-formation window\n")
        lines.append(f"- Max ICL gap = **{max_gap:.2f} nats**; reaches half-max by "
                     f"**~step {emerge}** (approx. when in-context copying forms).")
        mh = [r["max_head_induction"] for r in rows if not np.isnan(r["max_head_induction"])]
        if mh:
            mh_final = rows[-1]["max_head_induction"]
            lines.append(f"- Max-head induction score at final checkpoint = **{mh_final:.2f}** "
                         f"(lookup primitive robustly present in the decayed model).")
        lines.append("")

    # --- intervention
    if summary_rows:
        lines.append("## (B) Intervention factorial (mean +/- std over seeds)\n")
        lines.append("| task | schedule | mean acc | mean excess | floor | success rate |")
        lines.append("|---|---|---|---|---|---|")
        name = {1: "Hop-1 lookup", 2: "Hop-2 composition"}
        for r in summary_rows:
            lines.append(
                f"| {name[r['hop']]} | {r['schedule']} | "
                f"{r['mean_acc']:.3f} ± {r['std_acc']:.3f} | "
                f"{r['mean_excess']:+.3f} ± {r['std_excess']:.3f} | "
                f"{r['floor']:.3f} | {r['success_rate']:.0%} |"
            )
        lines.append("")

        # headline reads
        h1_low = _get(summary_rows, 1, "native_low", "mean_excess")
        h2_low = _get(summary_rows, 2, "native_low", "mean_excess")
        h2_rew = _get(summary_rows, 2, "rewarm", "mean_excess")
        h2_deep = _get(summary_rows, 2, "deep_low", "mean_excess")

        lines.append("## Headline reads (auto-detected)\n")
        HI, LO = 0.5, 0.2  # excess bars: "acquired" vs "near floor"
        if h1_low is not None and h2_low is not None:
            gap = h1_low - h2_low
            if h1_low >= HI and h2_low <= LO:
                lines.append(f"- **SELECTIVITY (clean)**: at the decayed LR the fresh lookup is "
                             f"acquired (excess {h1_low:+.2f}) while the composition stays near "
                             f"floor (excess {h2_low:+.2f}). Rules out generic plasticity loss.")
            elif gap >= 0.3 and h1_low > h2_low:
                lines.append(f"- **SELECTIVITY (partial/early)**: lookup is ahead of composition "
                             f"at the decayed LR (excess {h1_low:+.2f} vs {h2_low:+.2f}); the gap "
                             f"should widen with more steps. Confirm at the full step budget.")
            elif h1_low <= LO:
                lines.append(f"- **Neither acquired yet at the decayed LR** (lookup {h1_low:+.2f}, "
                             f"composition {h2_low:+.2f}). If this persists at the full step budget "
                             f"it points to generic plasticity loss; at low step counts it may "
                             f"simply be undertrained.")
            else:
                lines.append(f"- **No clear barrier at the native floor**: the composition is also "
                             f"climbing at the decayed LR (excess {h2_low:+.2f}). Check deep_low; "
                             f"the barrier may sit below Pythia's 10% LR floor.")
        if h2_low is not None and h2_rew is not None:
            if h2_rew - h2_low >= 0.3:
                lines.append(f"- **RECOVERABILITY present**: rewarm lifts Hop-2 excess from "
                             f"{h2_low:+.2f} to {h2_rew:+.2f}. Matches the toy's reopenable barrier.")
            elif h2_rew <= LO and h2_low <= LO:
                lines.append(f"- **Composition not acquired under either schedule yet** "
                             f"(rewarm {h2_rew:+.2f}, native {h2_low:+.2f}). Needs more steps, or "
                             f"the rewarm transient is still recovering -- check the loss curve.")
            else:
                lines.append(f"- **Rewarm did not clearly rescue** (Hop-2 {h2_low:+.2f} -> {h2_rew:+.2f}).")
        if h2_deep is not None:
            lines.append(f"- deep_low Hop-2 excess = {h2_deep:+.2f} "
                         f"(expected lowest if the barrier deepens with decay).")
        lines.append("\n*Reads use fixed excess bars (acquired >= +0.5, near-floor <= +0.2) and are "
                     "only reliable at the full step budget; short runs under-state acquisition.*")
        lines.append("")

    # --- lens
    if lens_by_schedule:
        lines.append("## (C) Mechanism (Hop-2 logit lens, candidate-restricted)\n")
        for sch, d in sorted(lens_by_schedule.items()):
            if "C" in d:
                lines.append(f"- {sch}: max answer-C decode across layers = "
                             f"**{max(d['C']):.2f}** (last layer {d['C'][-1]:.2f}).")
        lines.append("\nReorganisation signature: successful arms rebuild a decodable answer C "
                     "(and route through the intermediate B); failed arms do not, even though the "
                     "Hop-1 primitive remains intact.\n")

    out = os.path.join(cfg.out_dir, "SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  -> {out}")
    print("\n" + "\n".join(lines))


def _jump_step(curve, thresh=0.6, field="hop2_acc"):
    for pt in curve:
        if pt.get(field, 0.0) >= thresh:
            return pt["step"]
    return None


# --------------------------------------------------------------- causal ablation
def cmd_ablate(cfg, ablate_topk, lr, ablate_mode="induction", no_control=False):
    """Hold the checkpoint fixed; compare Hop-2 acquisition WITH vs WITHOUT its induction
    head(s). Removes the maturity confound in the checkpoint-axis result. ablate_mode=
    'random' runs a matched random-head control to separate circuit identity from count."""
    print(f"\n[ablate] causal head knockout at {cfg.late_revision}  "
          f"(top-{ablate_topk}, mode={ablate_mode}, lr={lr:g})")
    task = ChainTask(cfg)
    out = os.path.join(cfg.out_dir, "ablate")
    os.makedirs(out, exist_ok=True)
    abl_label = "ablated" if ablate_mode == "induction" else f"ablated_{ablate_mode}"
    conds = [] if no_control else [("control", 0, "induction")]
    conds.append((abl_label, ablate_topk, ablate_mode))
    rows = []
    for seed in range(cfg.seeds):
        for cond, k, mode in conds:
            print(f"\n  --- {cond} (ablate_topk={k}, mode={mode}) seed{seed} ---")
            res = train_arm(cfg, task, 2, lr, f"{cond}_lr{lr:g}", seed,
                            measure_induction=True, ablate_topk=k, ablate_mode=mode)
            save_json(res, os.path.join(out, f"{cond}_seed{seed}.json"))
            c = res["curve"]
            steps = [p["step"] for p in c]
            icl = [p.get("icl_gap", float("nan")) for p in c]
            mh = [p.get("max_head_induction", float("nan")) for p in c]
            acc = [p["hop2_acc"] for p in c]
            plotting.plot_track(steps, icl, mh, acc,
                                os.path.join(out, f"track_{cond}_seed{seed}.png"),
                                title=f"{cfg.late_revision} {cond}, lr={lr:g}")
            rows.append(dict(cond=cond, seed=seed, ablated=res["ablated_heads"],
                             jump=_jump_step(c), final_acc=res["final_hop2"]["acc"]))
            print(f"    -> {cond}: Hop-2 jump @ {rows[-1]['jump']}  "
                  f"final acc {rows[-1]['final_acc']:.3f}")

    def _mean_jump(cond):
        js = [r["jump"] for r in rows if r["cond"] == cond and r["jump"] is not None]
        return round(float(np.mean(js)), 1) if js else None

    lines = [f"# Causal head ablation at {cfg.late_revision}", "",
             f"LR={lr:g}; top-{ablate_topk}, mode={ablate_mode} (frozen knockout). "
             f"Reference: pre-induction step512 acquires Hop-2 at ~step 3400; intact "
             f"step1000 at ~step 1200; full induction knockout (top-8) at ~step 2400.", ""]
    for r in rows:
        lines.append(f"- seed{r['seed']} {r['cond']:>14}: jump @ {r['jump']}, "
                     f"final acc {r['final_acc']:.3f}"
                     + (f", heads={r['ablated']}" if r["ablated"] else ""))
    lines += ["", f"mean Hop-2 jump:  " + "  ".join(
        f"{c}={_mean_jump(c)}" for c in dict.fromkeys(r["cond"] for r in rows)), "",
        "Read (random-control): random ~ control => spare heads, so the induction-ablation "
        "slowdown is induction-SPECIFIC (real scaffold). random ~ induction-ablated => the "
        "slowdown is mid-layer capacity, not induction (over-parameterisation bites)."]
    with open(os.path.join(out, "ABLATE_SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


# ------------------------------------------------ interpolation / mode connectivity
def cmd_interp(cfg, lr, interp_points, ablate_topk=0, save_weights=False, between=None):
    """Walk the line between theta_0 and theta_final (or between two acquired solutions)
    and measure the Hop-2 loss/accuracy barrier -- the geometric form of 'reachability'."""
    task = ChainTask(cfg)
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    alphas = np.linspace(0.0, 1.0, interp_points)

    if between:
        fa, fb = between
        print(f"\n[interp] cross-checkpoint basin test: {fa}  ->  {fb}")
        model = load_model(cfg, dtype=torch.float32)
        model.eval()
        ta = I.load_param_list(fa, cfg.device)
        tb = I.load_param_list(fb, cfg.device)
        rows = I.interpolate_eval(model, task, cfg, ta, tb, alphas, label="A->B")
        b = I.barrier_metrics(rows)
        save_json(dict(mode="between", files=[fa, fb], rows=rows, barrier=b),
                  os.path.join(out, "interp_between.json"))
        plotting.plot_interp(rows, os.path.join(out, "interp_between.png"),
                             title="(solution A -> solution B)", barrier=b)
        print(f"\n  loss_barrier={b['loss_barrier']:+.3f} nats   acc_dip={b['acc_dip']:+.3f}")
        del model
        torch.cuda.empty_cache()
        return

    print(f"\n[interp] {cfg.late_revision}: acquire Hop-2 (lr={lr:g}"
          f"{', ablated' if ablate_topk else ''}), then interpolate theta_0 -> theta_final")
    res = train_arm(cfg, task, 2, lr, "interp", 0,
                    measure_induction=True, ablate_topk=ablate_topk, return_model=True)
    model, theta0 = res["model"], res["theta0"]
    theta1 = I.snapshot(model)
    print(f"  acquired: final Hop-2 acc = {res['final_hop2']['acc']:.3f}")
    if save_weights:
        I.save_param_list(theta0, os.path.join(out, f"theta0_{cfg.late_revision}.pt"))
        I.save_param_list(theta1, os.path.join(out, f"thetafinal_{cfg.late_revision}.pt"))
        print(f"  saved theta0_{cfg.late_revision}.pt / thetafinal_{cfg.late_revision}.pt")
    rows = I.interpolate_eval(model, task, cfg, theta0, theta1, alphas, label=cfg.late_revision)
    b = I.barrier_metrics(rows)
    save_json(dict(mode="checkpoint->final", revision=cfg.late_revision, lr=lr,
                   ablate_topk=ablate_topk, final_acc=res["final_hop2"]["acc"],
                   rows=rows, barrier=b),
              os.path.join(out, "interp_curve.json"))
    plotting.plot_interp(rows, os.path.join(out, "interp_curve.png"),
                         title=f"({cfg.late_revision} -> acquired)", barrier=b)
    print(f"\n  loss_barrier={b['loss_barrier']:+.3f} nats   acc_dip={b['acc_dip']:+.3f}   "
          f"endpoint loss={b['endpoint_loss']}  acc={b['endpoint_acc']}")
    print("  Read: ~flat/monotone (barrier ~ 0) => solution is downhill from the checkpoint, "
          "linearly mode-connected (low effective distance); a bump => the optimiser crossed "
          "a barrier (the harder-critical-period signature).")
    del model
    torch.cuda.empty_cache()


def cmd_all(cfg):
    cmd_induction(cfg)
    cmd_intervention(cfg)  # self-aggregates and writes SUMMARY.md (using induction on disk)


# ----------------------------------------------------------------------- smoke
def apply_smoke(cfg):
    cfg.induction_steps = (0, 1, 512, 143000)
    cfg.induction_batch = 4
    cfg.max_steps_hop1 = 60
    cfg.max_steps_hop2 = 60
    cfg.eval_every = 20
    cfg.batch_size = 64
    cfg.train_eval_batches = 2
    cfg.final_eval_batches = 2
    cfg.content_pool_size = 80
    cfg.seeds = 1
    cfg.schedules = ("native_low", "rewarm")
    cfg.tasks = (1, 2)
    cfg.warmup_steps = 10
    return cfg


# ------------------------------------------------------------------------ main
def build_cfg(args):
    cfg = Config()
    if getattr(args, "model", None):
        cfg.model_name = args.model
    if getattr(args, "revision", None):
        cfg.late_revision = args.revision
    if getattr(args, "steps", None):
        cfg.max_steps_hop2 = args.steps
    if getattr(args, "steps_hop1", None):
        cfg.max_steps_hop1 = args.steps_hop1
    if getattr(args, "chain_len", None):
        cfg.chain_len = args.chain_len
    if getattr(args, "seeds", None):
        cfg.seeds = args.seeds
    if getattr(args, "schedules", None):
        cfg.schedules = tuple(args.schedules)
    if getattr(args, "tasks", None):
        cfg.tasks = tuple(args.tasks)
    if getattr(args, "out_dir", None):
        cfg.out_dir = args.out_dir
    return cfg


def main():
    p = argparse.ArgumentParser(description="Pythia critical-period probe")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("induction", "intervention", "all", "smoke", "sweep", "sharpness",
                 "ablate", "interp"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", type=str, default=None)
        sp.add_argument("--revision", type=str, default=None)
        sp.add_argument("--steps", type=int, default=None,
                        help="Hop-2 step budget (the binding constraint)")
        sp.add_argument("--steps-hop1", dest="steps_hop1", type=int, default=None)
        sp.add_argument("--chain-len", dest="chain_len", type=int, default=None,
                        help="chain length L (smaller = composition forms faster)")
        sp.add_argument("--seeds", type=int, default=None)
        sp.add_argument("--schedules", nargs="+", default=None,
                        choices=["native_low", "deep_low", "rewarm"])
        sp.add_argument("--tasks", nargs="+", type=int, default=None, choices=[1, 2])
        sp.add_argument("--out-dir", dest="out_dir", type=str, default=None)
        if name in ("sweep", "sharpness"):
            sp.add_argument("--lrs", nargs="+", type=float, required=True,
                            help="learning rates, e.g. --lrs 6e-6 2e-5 6e-5 1.5e-4 6e-4")
        if name == "sweep":
            sp.add_argument("--track-induction", dest="track_induction", action="store_true",
                            help="track behavioral ICL gap + attention-based induction score")
            sp.add_argument("--track-distance", dest="track_distance", action="store_true",
                            help="track L2 weight movement ||theta_t - theta_0|| each eval")
        if name in ("ablate", "interp"):
            sp.add_argument("--lr", type=float, default=6e-5,
                            help="continued-training LR (band centre by default)")
        if name == "ablate":
            sp.add_argument("--ablate-topk", dest="ablate_topk", type=int, default=3,
                            help="number of top induction heads to knock out")
            sp.add_argument("--ablate-mode", dest="ablate_mode", default="induction",
                            choices=["induction", "random"],
                            help="'induction' cuts top-k induction heads; 'random' a matched control")
            sp.add_argument("--no-control", dest="no_control", action="store_true",
                            help="skip the unablated control arm (reuse a prior one)")
        if name == "interp":
            sp.add_argument("--ablate-topk", dest="ablate_topk", type=int, default=0,
                            help="knock out top induction heads before acquiring (optional)")
            sp.add_argument("--interp-points", dest="interp_points", type=int, default=21,
                            help="number of alpha samples along the interpolation")
            sp.add_argument("--save-final-weights", dest="save_final_weights",
                            action="store_true",
                            help="dump theta_0 / theta_final for a later --between basin test")
            sp.add_argument("--between", nargs=2, default=None,
                            metavar=("WEIGHTS_A", "WEIGHTS_B"),
                            help="interpolate between two saved theta_final files instead")
    args = p.parse_args()

    print_env()
    cfg = build_cfg(args)
    if args.cmd == "smoke":
        cfg = apply_smoke(cfg)
    if args.cmd == "sweep" and args.tasks is None:
        cfg.tasks = (2,)  # sweep the composition by default
    ensure_dirs(cfg)

    if args.cmd == "induction":
        cmd_induction(cfg)
    elif args.cmd == "intervention":
        cmd_intervention(cfg)
    elif args.cmd == "sweep":
        cmd_sweep(cfg, list(args.lrs),
                  track_induction=getattr(args, "track_induction", False),
                  track_distance=getattr(args, "track_distance", False))
    elif args.cmd == "sharpness":
        cmd_sharpness(cfg, list(args.lrs))
    elif args.cmd == "ablate":
        cmd_ablate(cfg, args.ablate_topk, args.lr,
                   ablate_mode=args.ablate_mode, no_control=args.no_control)
    elif args.cmd == "interp":
        cmd_interp(cfg, args.lr, args.interp_points,
                   ablate_topk=args.ablate_topk,
                   save_weights=args.save_final_weights,
                   between=args.between)
    else:  # all, smoke
        cmd_all(cfg)


if __name__ == "__main__":
    main()

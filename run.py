"""Entry point for the Pythia critical-period probe.

Subcommands:
  induction      observational induction-formation window across checkpoints
  intervention   the (task x schedule x seed) continued-training factorial
  all            induction + intervention + SUMMARY.md
  smoke          tiny end-to-end run to verify the environment (~minutes)

Run from the repo root, e.g.:
  python run.py all
  python run.py intervention --seeds 3 --schedules native_low deep_low rewarm --steps 3000
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
from src import plotting


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


def arm_path(cfg, hop, schedule, seed):
    return os.path.join(cfg.out_dir, "intervention", f"hop{hop}_{schedule}_seed{seed}.json")


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
                res = train_arm(cfg, task, hop, schedule, seed)
                save_json(res, arm_path(cfg, hop, schedule, seed))
    print("\n  all arms complete")
    aggregate_intervention(cfg)


def _load_all_arms(cfg):
    arms = []
    for seed in range(cfg.seeds):
        for hop in cfg.tasks:
            for schedule in cfg.schedules:
                p = arm_path(cfg, hop, schedule, seed)
                if os.path.exists(p):
                    arms.append(load_json(p))
    return arms


def aggregate_intervention(cfg):
    arms = _load_all_arms(cfg)
    if not arms:
        print("  (no arm results to aggregate)")
        return

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
                 f"seeds={cfg.seeds}, max_steps={cfg.max_steps}, "
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
        if h1_low is not None and h2_low is not None:
            if h1_low > 0.5 and h2_low < 0.2:
                lines.append(f"- **SELECTIVITY present**: at the decayed LR the fresh lookup "
                             f"learns (excess {h1_low:+.2f}) while the composition stays near "
                             f"floor (excess {h2_low:+.2f}). Rules out generic plasticity loss.")
            elif h1_low <= 0.5:
                lines.append(f"- **No selectivity**: even the fresh lookup struggles at the "
                             f"decayed LR (excess {h1_low:+.2f}) -> looks like generic plasticity "
                             f"loss, not composition-specific.")
            else:
                lines.append(f"- **No barrier at native floor**: the composition also learns at "
                             f"the decayed LR (excess {h2_low:+.2f}). Check deep_low below; the "
                             f"barrier may sit below Pythia's 10% LR floor.")
        if h2_low is not None and h2_rew is not None:
            if h2_rew - h2_low > 0.3:
                lines.append(f"- **RECOVERABILITY present**: rewarm lifts Hop-2 excess from "
                             f"{h2_low:+.2f} to {h2_rew:+.2f}. Matches the toy's reopenable barrier.")
            else:
                lines.append(f"- **Rewarm did not clearly rescue** (Hop-2 {h2_low:+.2f} -> {h2_rew:+.2f}).")
        if h2_deep is not None:
            lines.append(f"- deep_low Hop-2 excess = {h2_deep:+.2f} "
                         f"(expected lowest if the barrier deepens with decay).")
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


def cmd_all(cfg):
    ind = cmd_induction(cfg)
    cmd_intervention(cfg)
    agg = aggregate_intervention(cfg)
    summary_rows, lens = agg if agg else ([], {})
    write_summary(cfg, ind, summary_rows, lens)


# ----------------------------------------------------------------------- smoke
def apply_smoke(cfg):
    cfg.induction_steps = (0, 1, 512, 143000)
    cfg.induction_batch = 4
    cfg.max_steps = 60
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
        cfg.max_steps = args.steps
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
    for name in ("induction", "intervention", "all", "smoke"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", type=str, default=None)
        sp.add_argument("--revision", type=str, default=None)
        sp.add_argument("--steps", type=int, default=None)
        sp.add_argument("--seeds", type=int, default=None)
        sp.add_argument("--schedules", nargs="+", default=None,
                        choices=["native_low", "deep_low", "rewarm"])
        sp.add_argument("--tasks", nargs="+", type=int, default=None, choices=[1, 2])
        sp.add_argument("--out-dir", dest="out_dir", type=str, default=None)
    args = p.parse_args()

    print_env()
    cfg = build_cfg(args)
    if args.cmd == "smoke":
        cfg = apply_smoke(cfg)
    ensure_dirs(cfg)

    if args.cmd == "induction":
        cmd_induction(cfg)
    elif args.cmd == "intervention":
        cmd_intervention(cfg)
    else:  # all, smoke
        cmd_all(cfg)


if __name__ == "__main__":
    main()

# Pythia critical-period probe

A small, self-contained transfer test of the toy-model finding — that a late/decayed
learning-rate schedule imposes a **selective, recoverable acquirability barrier** on
*dependent compositions* — in a real pretrained model (Pythia-160m). Designed to run on a
single A100.

It is a **transfer probe, not a scaling proof**: the question is whether the toy mechanism
is an artifact of extreme simplicity or recurs on a real model with real vocabulary.

## What it does

**(A) Observational — date the induction window.** Load Pythia-160m at ~19 training
checkpoints and measure when in-context copying forms (loss gap on a repeated-random-token
sequence; plus a max-head induction-attention score). This confirms the lookup primitive is
robustly present in the final, maximally-decayed checkpoint — the toy's premise (primitive
present, dependent skill withheld).

**(B) Intervention — continue-train the decayed checkpoint.** Starting from `step143000`,
continue training on a synthetic in-context chain-retrieval task rendered in Pythia's own
vocabulary (no new embedding rows), across a factorial:

| axis | values |
|---|---|
| task | Hop-1 (lookup primitive) · Hop-2 (composition, must route through the intermediate) |
| schedule (LR) | `native_low`=6e-5 (the decayed model's own floor) · `deep_low`=6e-6 (below the floor) · `rewarm`=6e-4 (back to peak) |
| seed | 3 |

Both hops are evaluated throughout every run. Metric is **excess = accuracy − chance floor**.
For Hop-2 we also run a per-layer **logit lens** to see whether a successful arm rebuilds a
decodable answer (and routes through the intermediate B) — the toy's reorganisation signature.

> Note on optimiser state: public Pythia checkpoints carry no optimiser state, so `rewarm`
> here is a rewarm **with reset**. The toy showed rewarm ≈ rewarm+reset, which is exactly what
> licenses this.

## Predicted pattern (the headline)

- Hop-1 / `native_low` → **succeeds** (selectivity: a fresh simple lookup still learns at the decayed LR)
- Hop-2 / `native_low` → **fails** (the barrier)
- Hop-2 / `rewarm`     → **succeeds** (recoverability)

## Setup (RunPod)

Use a **PyTorch 2.x / CUDA 12** image. Then:

```bash
cd pythia-critical-period
pip install -r requirements.txt
```

No HF token needed (Pythia is public). First run downloads ~5–7 GB of checkpoints into the
HF cache; make sure the volume has ~15 GB free.

## Commands

```bash
# 0) sanity-check the environment (a few minutes, mostly downloads)
bash scripts/run_smoke.sh

# 1) HEADLINE EXPERIMENT -- LR sweep for the composition.
#    Maps the inverted-U: the Hop-2 composition forms only within a learning-rate
#    band (blocked both above and below). ~2.5-3 h for 5 LRs x 3 seeds.
python run.py sweep --lrs 6e-6 2e-5 6e-5 1.5e-4 6e-4 --seeds 3 \
    --steps 7000 --out-dir results_sweep
#    -> results_sweep/sweep_invertedU.png, sweep_curves.png, SWEEP_SUMMARY.md

# 2) (optional) full named-schedule factorial incl. the cheap Hop-1 selectivity arms
bash scripts/run_full.sh

# 3) bundle results to send back
zip -r results_bundle.zip results_sweep >/dev/null   # or: bash scripts/zip_results.sh
```

Knobs: `--chain-len 4` (composition forms even faster), `--steps 12000` (if a low-LR arm
is still climbing at the end), wider `--lrs` to locate the band edges precisely.

Other sub-runs:

```bash
python run.py induction                                   # just the induction window
python run.py all --model EleutherAI/pythia-410m          # bigger model
```

## What to send back

Everything in `results/` is small (JSON/CSV/PNG; no weights). `results_bundle.zip` is what to
return. Key thesis artifacts inside it:

- `induction_window.png` / `.csv` — when in-context copying forms (Act-3 motivation)
- `intervention_summary.csv` — the selectivity + recoverability table (the headline numbers)
- `intervention_curves.png` — Hop-2 barrier vs rescue, with Hop-1 as the selectivity baseline
- `logit_lens.png` — per-layer answer/intermediate decode (mechanism transfer)
- `SUMMARY.md` — auto-generated headline reads with the actual numbers, paste-ready

## Interpreting the outcome (four canonical cases)

The design pre-registers *signatures*, not a conclusion. `SUMMARY.md` auto-detects which case
you landed in; the mapping:

1. **Toy replicates** — Hop-1/native_low high, Hop-2/native_low ≈ floor, Hop-2/rewarm high →
   schedule-induced, reopenable barrier specific to dependent compositions.
2. **Generic plasticity loss** — even Hop-1/native_low struggles → the dissociation is *not*
   composition-specific; the compositional framing loses support.
3. **No barrier at the native floor** — Hop-2/native_low also succeeds → Pythia's 10% LR floor
   sits *above* the threshold; look at `deep_low` to see where the barrier reappears.
4. **Irreversible component** — Hop-2 stays low even under `rewarm` → an entrenchment the toy
   lacked (would be a genuinely new finding).

## Repo layout

```
config.py            all knobs (model, LRs, steps, task params)
run.py               entry point: induction | intervention | all | smoke
src/data.py          synthetic chain-retrieval task (fixed-length, target-only loss)
src/model_utils.py   load Pythia checkpoints via HF (faithful GPT-NeoX params)
src/induction.py     induction-window metrics across checkpoints
src/train.py         continued-training loop for one arm
src/eval.py          accuracy/floor/excess, candidate mass, logit lens
src/plotting.py      figures
scripts/             run_smoke.sh, run_full.sh, zip_results.sh
```

## Notes / knobs worth knowing

- `config.py` is the single source of truth; CLI flags override the common ones.
- Model weights are **not saved** by default (keeps the bundle tiny). If you later want
  per-arm checkpoints for deeper mechanistic work, that's a small addition to `train_arm`.
- Sequences are fixed-length (28 tokens for chain_len=8), so batches need no padding — large
  batch sizes are cheap; raise `batch_size` if the A100 is underused.

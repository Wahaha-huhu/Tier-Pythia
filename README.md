# cp pythia. Conditional and compositional reachability on Pythia

This repository tests, on a real pretrained Pythia model, the claim the controlled toy
established. A decayed schedule creates a selective and recoverable barrier to acquiring
a new composition through continued training. The composition is a two hop chain lookup
that reuses a single hop lookup prerequisite. The test intervenes on continued training,
since Pythia cannot have its pretraining schedule re run. The headline is the
interventional acquirability matrix. An observational cascade across released checkpoints
is included as supporting evidence and makes no causal claim.

## What is inside

The package cp_pythia holds the chain retrieval generator mapped onto Pythia vocabulary
ids, the masked target loss and accuracy with the next token off by one, the induction
and key slot scores from returned attention weights, mean ablation of heads, a held out
pretraining perplexity probe, and the continued training loop with the rate arms. The
scripts run the sanity checks, the calibration gates, a single arm, the observational
cascade, and the result packing. The generator and metric logic are tested offline in
tests, run them first.

## Setup

Install torch matched to the pod CUDA, then the rest.

```bash
pip install -r requirements.txt
python -m pytest tests -q            # or, python tests/test_generator.py
```

All commands below are run from the repository root, which puts cp_pythia and scripts on
the path. No editable install is required, though pip install -e . also works.

## Run sequence

First the harness sanity checks, which download the model and verify the wiring before any
real compute.

```bash
python -m scripts.run_sanity --model pythia-160m-deduped --device cuda
```

Then the three calibration gates. Difficulty looks for a chain length where hop one is
solved by the base but hop two sits at floor. Learnability confirms a rewarmed run can
teach hop two. The barrier gate confirms the low rate is impaired relative to rewarm
before the full matrix is launched.

```bash
python -m scripts.run_calibration --model pythia-160m-deduped --device cuda --stage difficulty
python -m scripts.run_calibration --model pythia-160m-deduped --device cuda --stage learnability --chain-length 8
python -m scripts.run_calibration --model pythia-160m-deduped --device cuda --stage barrier --chain-length 8
```

Then the core matrix. Each cell is one arm, one intro task, one seed. The composition is
intro task compose. The two selectivity controls are fresh_hop1, reuse leaning, and
reverse, formation leaning. Run the low and rewarm arms across seeds for the composition
and both controls, then the matched budget and reset arms for the composition.

```bash
# composition, low and rewarm, three seeds
for SEED in 1 2 3; do
  python -m scripts.run_arm --model pythia-160m-deduped --device cuda --arm low \
    --intro-task compose --seed $SEED --chain-length 8 --post-steps 6000 \
    --with-perplexity --out-dir runs/m160/low_compose_s$SEED
  python -m scripts.run_arm --model pythia-160m-deduped --device cuda --arm rewarm \
    --intro-task compose --seed $SEED --chain-length 8 --post-steps 6000 \
    --with-perplexity --out-dir runs/m160/rewarm_compose_s$SEED
done

# selectivity controls, low and rewarm, three seeds
for SEED in 1 2 3; do
  for TASK in fresh_hop1 reverse; do
    python -m scripts.run_arm --model pythia-160m-deduped --device cuda --arm low \
      --intro-task $TASK --seed $SEED --chain-length 8 --post-steps 6000 \
      --out-dir runs/m160/low_${TASK}_s$SEED
    python -m scripts.run_arm --model pythia-160m-deduped --device cuda --arm rewarm \
      --intro-task $TASK --seed $SEED --chain-length 8 --post-steps 6000 \
      --out-dir runs/m160/rewarm_${TASK}_s$SEED
  done
done
```

Read the rewarm composition summaries for the final cumulative update ratio, then run the
matched budget arm with that value so the low rate gets the same post introduction budget.

```bash
# substitute the rewarm final_cum_update_ratio you read from a rewarm summary
for SEED in 1 2 3; do
  python -m scripts.run_arm --model pythia-160m-deduped --device cuda --arm matched_budget \
    --intro-task compose --seed $SEED --chain-length 8 --post-steps 40000 \
    --match-budget-to 3.0 --out-dir runs/m160/matched_compose_s$SEED
done
```

The rewarm sweep maps the threshold by overriding the rewarm target.

```bash
for FRAC in 0.02 0.05 0.1 0.25 0.5 1.0; do
  python -m scripts.run_arm --model pythia-160m-deduped --device cuda --arm rewarm \
    --intro-task compose --seed 1 --chain-length 8 --post-steps 6000 \
    --rewarm-lr $(python -c "print(6e-4*$FRAC)") --out-dir runs/m160/sweep_${FRAC}_s1
done
```

The observational cascade, supporting evidence only.

```bash
python -m scripts.run_observational --model pythia-160m-deduped --device cuda \
  --steps 0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,143000
```

## Send the results back

The pack script collects every summary, log, and calibration and cascade json into one
small archive with a manifest. No model weights are included.

```bash
python -m scripts.pack_results --runs-dir runs --out cp_pythia_results.zip
```

Send cp_pythia_results.zip. The summaries carry the tail accuracies, the hop two excess
over floor, the cumulative update ratio, the transition width, and the perplexity change
from base for every cell, which is what the thesis figures are built from.

## Notes

The off by one is handled, the target at the final position is read from the logits at the
position before it. Eager attention is forced for the analysis so attention weights are
returned. The ablation hook uses the GPTNeoX output projection, and if a newer transformers
version renames it the model_io fallback finds it by shape. Verify the model peak and
minimum learning rate against the Pythia configuration before trusting the default rate
values in config. The reset arm needs a prerequisite phase to have an optimiser state to
reset, so use it with prereq steps greater than zero, otherwise continued training already
starts from a fresh optimiser.

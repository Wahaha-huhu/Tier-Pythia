"""Central configuration. All experiment knobs live here.

Pythia-160m schedule facts used below:
  peak LR = 6e-4, final (cosine) LR = 6e-5  (i.e. decays to 10% of peak),
  143000 total steps. The final published checkpoint (step143000) is therefore
  the maximally-decayed model -- the right substrate for a "late signal arrival"
  test, mirroring the toy setting (primitive present, dependent skill withheld).
"""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Config:
    # ---- Model ----
    model_name: str = "EleutherAI/pythia-160m"
    late_revision: str = "step143000"      # decayed model = continued-training start point
    device: str = "cuda"

    # ---- Synthetic chain-retrieval task ----
    # Content symbols are EXISTING token IDs (no new embedding rows). Bindings are
    # randomised per example, so the task is pure in-context retrieval (no memorisation).
    content_pool_size: int = 200
    pool_lo: int = 256                     # skip byte-level/very-common low IDs
    pool_hi: int = 50000                   # stay well within real-token range (<50254)
    # markers: QUERY, HOP_1, HOP_2, SEP  (rare but REAL trained token rows, <50254)
    marker_ids: Tuple[int, int, int, int] = (50250, 50251, 50252, 50253)
    chain_len: int = 5                     # L; chain has L+1 distinct tokens, L edges.
    #                                        Reduced from 8: shorter in-context search lets the
    #                                        Hop-2 composition form in far fewer steps (the
    #                                        phenomenon is not L-specific). Floor = 1/(L+1).
    n_distractors: int = 0                  # extra off-chain [key,val] edges mixed into context
    #                                        (0 = original task; >0 raises difficulty / capacity
    #                                        pressure; also an axis of the generalization battery)
    randomize_tokens: bool = False          # if True, draw each example's chain tokens fresh from
    #                                        the FULL vocab range instead of the fixed 200-pool, so
    #                                        the model cannot memorise token identities and must
    #                                        learn a token-agnostic lookup (i.e. use induction).
    pool_seed: int = 0                     # fixes the content pool identically across all runs

    # ---- Learning-rate schedules (the core manipulation) ----
    lr_native_low: float = 6e-5            # what the decayed model's own schedule gives
    lr_deep_low: float = 6e-6             # keep decaying below Pythia's floor (toward toy regime)
    lr_rewarm: float = 6e-4              # rewarm to peak. NB: fresh optimiser == rewarm+RESET,
    #                                       which the toy showed is equivalent to plain rewarm.
    warmup_steps: int = 200                # smooth ramp; the 10x jump to rewarm peak spikes
    #                                        the new-task loss without an adequate warmup

    # ---- Continued-training loop ----
    # Per-task budgets: Hop-1 saturates fast (~500 steps), Hop-2 needs a long runway to
    # pass through the pre-jump phase and (if it does) make the accuracy jump.
    max_steps_hop1: int = 1500
    max_steps_hop2: int = 8000
    batch_size: int = 256
    grad_clip: float = 1.0
    weight_decay: float = 0.01             # matches Pythia
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    eval_every: int = 150
    train_eval_batches: int = 4            # light eval during training (for curves)
    eval_batch_size: int = 128

    # ---- Final eval (+ logit lens) ----
    final_eval_batches: int = 16
    eval_seed: int = 12345                 # fixed -> identical eval set for every arm/timepoint

    # ---- Sharpness / edge-of-stability probe ----
    sharpness_batch: int = 64              # fixed batch for the Hessian estimate
    sharpness_n_iter: int = 20             # power-iteration steps for top eigenvalue
    sharpness_eps_rel: float = 3e-3        # finite-difference step, relative to ||theta||

    # ---- Observational induction window ----
    # All steps below exist as HF branches for Pythia (step0; powers of two to 512; then by 1000).
    induction_steps: Tuple[int, ...] = (
        0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000,
    )
    induction_T: int = 64                  # repeated-random-token block length (sequence = 2T)
    induction_batch: int = 16
    induction_seed: int = 7

    # ---- Factorial selection (overridable from CLI) ----
    tasks: Tuple[int, ...] = (1, 2)                              # 1 = lookup primitive, 2 = composition
    schedules: Tuple[str, ...] = ("native_low", "deep_low", "rewarm")
    seeds: int = 3

    # ---- Output ----
    out_dir: str = "results"

    # success threshold (accuracy) for the summary's success-rate column
    success_acc: float = 0.90

    def lr_for(self, schedule: str) -> float:
        return {
            "native_low": self.lr_native_low,
            "deep_low": self.lr_deep_low,
            "rewarm": self.lr_rewarm,
        }[schedule]

    def steps_for(self, hop: int) -> int:
        return self.max_steps_hop1 if hop == 1 else self.max_steps_hop2

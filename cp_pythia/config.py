"""Frozen configuration for the Pythia conditional-compositional reachability experiments.

All experiment knobs live here so they can be frozen and recorded for the thesis.
Token ids are mapped onto existing Pythia vocabulary so no embedding is added.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


# Pythia per-model peak and minimum learning rates and warmup.
# VERIFY against the Pythia configuration for the exact model you load.
PYTHIA_LR: Dict[str, Dict[str, float]] = {
    "pythia-160m-deduped": {"peak": 6.0e-4, "min": 6.0e-5, "pretrain_warmup": 1430},
    "pythia-410m-deduped": {"peak": 3.0e-4, "min": 3.0e-5, "pretrain_warmup": 1430},
    "pythia-1b-deduped": {"peak": 3.0e-4, "min": 3.0e-5, "pretrain_warmup": 1430},
}

PYTHIA_FINAL_STEP = 143000
PYTHIA_REPO_PREFIX = "EleutherAI/"


@dataclass
class TaskConfig:
    """The chain-retrieval task, mapped to Pythia vocabulary ids."""

    content_vocab: int = 64          # size of the main content-token pool
    chain_length: int = 8            # M, number of distinct content tokens per chain
    k_max: int = 2                   # maximum hop count for the composition
    p_multi: float = 0.5             # fraction of multi-hop queries after introduction
    prepend_bos: bool = True         # prepend the tokenizer eos/bos id 0

    # token-id layout in Pythia vocabulary (all < 50304, disjoint blocks)
    eos_id: int = 0                  # Pythia <|endoftext|>, reused as bos
    special_base: int = 900          # SEP and markers occupy a small block here
    content_base: int = 1000         # main content pool ids
    fresh_base: int = 5000           # disjoint fresh content pool for the reuse control

    def __post_init__(self):
        b = self.special_base
        # fixed special-token ids
        self.SEP = b + 0
        self.QUERY_A = b + 1          # forward retrieval, main pool
        self.QUERY_B = b + 2          # forward retrieval, fresh pool (reuse control)
        self.QUERY_REV = b + 3        # reverse retrieval (new-pattern control)
        self.HOP = [b + 10 + i for i in range(self.k_max + 1)]  # HOP[h] for h in 0..k_max
        self.content_ids = list(range(self.content_base, self.content_base + self.content_vocab))
        self.fresh_ids = list(range(self.fresh_base, self.fresh_base + self.content_vocab))

    @property
    def seq_len(self) -> int:
        # M-1 bindings of 3 tokens, then marker, hop, start, target = 4 tokens
        return 3 * (self.chain_length - 1) + 4 + (1 if self.prepend_bos else 0)

    @property
    def pred_pos(self) -> int:
        # the query start-content position, whose logits predict the target
        return self.seq_len - 2

    @property
    def target_pos(self) -> int:
        return self.seq_len - 1

    def to_dict(self):
        d = asdict(self)
        d.update(seq_len=self.seq_len, pred_pos=self.pred_pos, target_pos=self.target_pos)
        return d


@dataclass
class ArmConfig:
    """A continued-training rate arm. See the plan, tier one section 2.4."""

    name: str                        # "low" | "rewarm" | "matched_budget" | "reset"
    post_lr: float                   # constant post-introduction rate (target)
    rewarm_warmup: int = 0           # linear warmup steps to post_lr (0 for the low arm)
    reset_optimizer_at_intro: bool = False
    match_budget_to: float = None    # for matched_budget, the cumulative update ratio to reach


@dataclass
class TrainConfig:
    model: str = "pythia-160m-deduped"
    revision_step: int = PYTHIA_FINAL_STEP

    # staged prerequisite phase (set prereq_steps = 0 if the base already solves hop1)
    prereq_steps: int = 0
    prereq_lr: float = 1.0e-4

    # post-introduction phase
    post_steps: int = 6000
    intro_task: str = "compose"      # "compose" | "fresh_hop1" | "reverse" | "hop2_only"

    batch_size: int = 64
    eval_interval: int = 50
    eval_batches: int = 16
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    seed: int = 0
    dtype: str = "bfloat16"

    # replay to keep the prerequisite and the language model alive during the hop2 phase
    replay_hop1_frac: float = 0.0
    replay_lang_frac: float = 0.0

    out_dir: str = "runs/run"


def default_arms(peak: float, minimum: float) -> Dict[str, ArmConfig]:
    """Default arm set built from a model's peak and minimum rates."""
    return {
        "low": ArmConfig(name="low", post_lr=minimum, rewarm_warmup=0),
        "rewarm": ArmConfig(name="rewarm", post_lr=peak, rewarm_warmup=200),
        "matched_budget": ArmConfig(name="matched_budget", post_lr=minimum, rewarm_warmup=0,
                                    match_budget_to=None),
        "reset": ArmConfig(name="reset", post_lr=peak, rewarm_warmup=200,
                           reset_optimizer_at_intro=True),
    }


# Default rewarm sweep as fractions of the model peak (see plan section 2.5).
REWARM_SWEEP_FRACTIONS: List[float] = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]

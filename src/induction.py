"""Observational part: date the induction-formation window in Pythia.

On a repeated-random-token sequence  [t_0..t_{T-1}, t_0..t_{T-1}]  the second copy is
perfectly predictable by an induction (prefix-match-and-copy) mechanism.

  icl_gap          = mean loss on first-copy targets  -  mean loss on second-copy targets
                     (grows sharply once in-context copying forms; the primary signal)
  max_head_induction / top5  = attention paid by the strongest head(s) to the induction
                     position  (q -> q - T + 1)  in the repeated region (needs eager attn).

This confirms the developmental window is dateable in the exact model we then perturb,
and that the lookup primitive is robustly present in the late checkpoint.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model_utils import load_model


@torch.no_grad()
def icl_gap_only(model, cfg):
    """Loss-based in-context-copying signal on a repeated-random-token sequence.
    Uses logits only (no output_attentions), so it is SDPA-safe and can be called
    cheaply during continued training without disturbing it. This is the same primary
    signal as the induction window: it tracks whether GENERAL induction (prefix-match
    -and-copy) is present, independent of the synthetic chain format."""
    T, B = cfg.induction_T, cfg.induction_batch
    rng = np.random.default_rng(cfg.induction_seed)
    base = rng.integers(cfg.pool_lo, cfg.pool_hi, size=(B, T))
    seq = np.concatenate([base, base], axis=1)
    input_ids = torch.tensor(seq, dtype=torch.long, device=cfg.device)
    was_training = model.training
    model.eval()
    out = model(input_ids=input_ids)
    model.train() if was_training else None
    logits = out.logits
    V = logits.shape[-1]
    sl = logits[:, :-1, :]
    lab = input_ids[:, 1:]
    lt = F.cross_entropy(sl.reshape(-1, V), lab.reshape(-1), reduction="none").reshape(B, 2 * T - 1)
    first = lt[:, 0:T - 1].mean().item()
    second = lt[:, T:2 * T - 1].mean().item()
    return first - second


@torch.no_grad()
def induction_metrics_for_model(model, cfg):
    T, B = cfg.induction_T, cfg.induction_batch
    rng = np.random.default_rng(cfg.induction_seed)
    base = rng.integers(cfg.pool_lo, cfg.pool_hi, size=(B, T))
    seq = np.concatenate([base, base], axis=1)               # [B, 2T]
    input_ids = torch.tensor(seq, dtype=torch.long, device=cfg.device)

    out = model(input_ids=input_ids, output_attentions=True)
    logits = out.logits
    V = logits.shape[-1]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    lt = F.cross_entropy(
        shift_logits.reshape(-1, V), shift_labels.reshape(-1), reduction="none"
    ).reshape(B, 2 * T - 1)
    first = lt[:, 0 : T - 1].mean().item()                   # first-copy targets t=1..T-1
    second = lt[:, T : 2 * T - 1].mean().item()              # second-copy targets t=T+1..2T-1
    icl_gap = first - second

    max_head, top5 = float("nan"), float("nan")
    try:
        attns = out.attentions                               # tuple len L, each [B, H, 2T, 2T]
        qs = torch.arange(T, 2 * T, device=cfg.device)
        ks = qs - T + 1
        head_scores = []
        for A in attns:
            head_scores.append(A[:, :, qs, ks].mean(dim=(0, 2)))  # [H]
        hs = torch.cat(head_scores)                          # [L*H]
        max_head = hs.max().item()
        top5 = hs.topk(min(5, hs.numel())).values.mean().item()
    except Exception:  # noqa: BLE001 -- some attn impls return None; loss metric still stands
        pass

    return dict(
        first_loss=first, second_loss=second, icl_gap=icl_gap,
        max_head_induction=max_head, top5_head_induction=top5,
    )


@torch.no_grad()
def induction_track_metrics(model, cfg):
    """In-training probe of GENERAL induction, returning BOTH signals from one forward:
      icl_gap   -- behavioral: second-copy minus first-copy loss advantage (nats)
      max_head  -- mechanistic: strongest single head's attention to the induction
                   position (q -> q-T+1) in the repeated region; 0-1
      top5      -- mean of the five strongest such heads
    Requests output_attentions=True, which transparently falls back to eager for THIS
    forward only (training stays on SDPA, dynamics unchanged). If a given transformers
    build refuses output_attentions under SDPA and raises, we retry a plain forward so
    the behavioral signal always survives and max_head/top5 are reported as NaN.

    The probe tokens are drawn from [pool_lo, pool_hi); only ~0.4% coincide with the
    200-token chain content pool, so this measures induction on essentially-disjoint
    tokens -- i.e. whether GENERAL prefix-match-and-copy survives, not chain lookup."""
    T, B = cfg.induction_T, cfg.induction_batch
    rng = np.random.default_rng(cfg.induction_seed)
    base = rng.integers(cfg.pool_lo, cfg.pool_hi, size=(B, T))
    seq = np.concatenate([base, base], axis=1)
    input_ids = torch.tensor(seq, dtype=torch.long, device=cfg.device)
    was_training = model.training
    model.eval()

    max_head, top5 = float("nan"), float("nan")
    try:
        out = model(input_ids=input_ids, output_attentions=True)
        attns = getattr(out, "attentions", None)
        if attns is not None and attns[0] is not None:
            qs = torch.arange(T, 2 * T, device=cfg.device)
            ks = qs - T + 1
            hs = torch.cat([A[:, :, qs, ks].mean(dim=(0, 2)) for A in attns])  # [L*H]
            max_head = hs.max().item()
            top5 = hs.topk(min(5, hs.numel())).values.mean().item()
    except Exception:  # noqa: BLE001 -- SDPA build refused output_attentions; retry plain
        out = model(input_ids=input_ids)

    logits = out.logits
    V = logits.shape[-1]
    lt = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, V), input_ids[:, 1:].reshape(-1), reduction="none"
    ).reshape(B, 2 * T - 1)
    icl_gap = lt[:, 0 : T - 1].mean().item() - lt[:, T : 2 * T - 1].mean().item()

    if was_training:
        model.train()
    return icl_gap, max_head, top5


def run_induction_window(cfg):
    rows = []
    for step in cfg.induction_steps:
        rev = f"step{step}"
        try:
            model = load_model(cfg, revision=rev, attn_impl="eager", dtype=torch.float32)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {rev}: {e}")
            continue
        m = induction_metrics_for_model(model, cfg)
        m["step"] = step
        rows.append(m)
        print(
            f"  step {step:>7}: icl_gap={m['icl_gap']:+.3f}  "
            f"second_loss={m['second_loss']:.3f}  max_head={m['max_head_induction']:.3f}"
        )
        del model
        torch.cuda.empty_cache()
    return rows

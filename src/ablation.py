"""Causal induction-head ablation at a FIXED checkpoint.

The checkpoint-axis result compared step512 (pre-induction) with step1000 (post-induction)
but those differ in MATURITY as well as in whether an induction head exists. To isolate the
causal role of the induction circuit we instead hold the checkpoint fixed (step1000) and
remove the induction head(s), then run the identical Hop-2 continued-training.

Ablation = mask the per-head block of the attention output-projection input, so the head
contributes nothing to the residual stream. Because the head is thereby disconnected from
the loss, it also receives no gradient and stays ablated for the whole run (a frozen
knockout, not a one-shot zeroing the optimiser could undo). The head's attention WEIGHTS
are still computed (so we exclude it explicitly when probing induction), but its OV/output
path is dead -- the standard way to knock out a head's causal contribution.

Interpretation of the resulting Hop-2 speed:
  ablated step1000 ~ step512 (slow)   -> the induction circuit WAS the head-start; the
                                         composition reorganises *from* it (load-bearing).
  ablated step1000 ~ step1000 (fast)  -> the head-start comes from general maturity, not
                                         the induction head; the head was incidental.
"""
import numpy as np
import torch


def _layers_and_dims(model):
    cfg = model.config
    hidden = cfg.hidden_size
    nheads = cfg.num_attention_heads
    hd = hidden // nheads
    # Pythia == GPTNeoXForCausalLM
    base = getattr(model, "gpt_neox", None) or getattr(model, "transformer", None)
    layers = base.layers if hasattr(base, "layers") else base.h
    return layers, nheads, hd


def _output_proj(attn):
    for name in ("dense", "o_proj", "out_proj"):
        m = getattr(attn, name, None)
        if m is not None:
            return m
    raise AttributeError("could not locate attention output projection")


@torch.no_grad()
def rank_induction_heads(model, cfg):
    """Return [((layer, head), score), ...] sorted by induction-attention score
    (mean attention to the prefix-match position q -> q-T+1 on a repeated-random probe),
    strongest first. Uses output_attentions (eager fallback for this forward only)."""
    T, B = cfg.induction_T, cfg.induction_batch
    rng = np.random.default_rng(cfg.induction_seed)
    base = rng.integers(cfg.pool_lo, cfg.pool_hi, size=(B, T))
    seq = np.concatenate([base, base], axis=1)
    ids = torch.tensor(seq, dtype=torch.long, device=cfg.device)
    was = model.training
    model.eval()
    out = model(input_ids=ids, output_attentions=True)
    attns = out.attentions
    qs = torch.arange(T, 2 * T, device=cfg.device)
    ks = qs - T + 1
    ranked = []
    for L, A in enumerate(attns):
        hs = A[:, :, qs, ks].mean(dim=(0, 2))   # [H]
        for h in range(hs.numel()):
            ranked.append(((L, h), float(hs[h].item())))
    if was:
        model.train()
    ranked.sort(key=lambda x: -x[1])
    return ranked


def select_ablation_heads(model, cfg, topk, mode="induction", seed=0):
    """Return (heads_to_ablate, ranked). mode='induction' -> the top-k induction heads.
    mode='random' -> a MATCHED control: for each induction head in a layer, pick a random
    OTHER head in that same layer (same count, same per-layer profile, disjoint from the
    induction set), so the comparison isolates circuit identity from head count/depth."""
    ranked = rank_induction_heads(model, cfg)
    induction_heads = [lh for (lh, _sc) in ranked[:topk]]
    if mode == "induction":
        return induction_heads, ranked
    rng = np.random.default_rng(seed)
    nheads = model.config.num_attention_heads
    ind_set = set(induction_heads)
    chosen = []
    for (L, _h) in induction_heads:                      # one random head per induction slot, same layer
        opts = [(L, hh) for hh in range(nheads) if (L, hh) not in ind_set and (L, hh) not in chosen]
        chosen.append(opts[int(rng.integers(len(opts)))])
    return chosen, ranked


def apply_head_ablation(model, head_list):
    """Functionally knock out the given (layer, head) pairs for the model's lifetime.
    Returns hook handles (keep them alive for the whole run)."""
    layers, nheads, hd = _layers_and_dims(model)
    by_layer = {}
    for (L, h) in head_list:
        by_layer.setdefault(int(L), []).append(int(h))

    def make_hook(heads):
        def hook(module, args):
            x = args[0].clone()
            for h in heads:
                x[..., h * hd:(h + 1) * hd] = 0.0
            return (x,) + tuple(args[1:])
        return hook

    handles = []
    for L, heads in by_layer.items():
        proj = _output_proj(layers[L].attention)
        handles.append(proj.register_forward_pre_hook(make_hook(heads)))
    return handles

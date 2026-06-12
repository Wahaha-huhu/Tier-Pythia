"""Mean ablation of attention heads. Torch, GPTNeoX layout.

A head's contribution is replaced by that head's global mean over a calibration set,
applied before the attention output projection where each head occupies a contiguous
slice of width d_head. A single global mean is used, not a per-position mean, so the
fixed layout is not leaked.
"""

import numpy as np
import torch

from .model_io import output_projection


@torch.no_grad()
def compute_head_means(model, calib_ids_iter, device, n_layers, n_heads, d_head):
    """Mean per (layer, head) of the input to each output projection over a calibration set."""
    bucket = {l: [] for l in range(n_layers)}
    handles = []

    def make_capture(l):
        def hook(module, args):
            (x,) = args                                  # [B, S, n_heads*d_head]
            bucket[l].append(x.detach().float().reshape(-1, n_heads, d_head).cpu())
        return hook

    for l in range(n_layers):
        handles.append(output_projection(model, l).register_forward_pre_hook(make_capture(l)))
    try:
        for ids in calib_ids_iter:
            ids = torch.as_tensor(ids, dtype=torch.long, device=device)
            model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(bucket[l]).mean(0) for l in bucket}    # each [n_heads, d_head]


class AblationContext:
    """Context manager that mean-ablates a set of (layer, head) pairs during evaluation."""

    def __init__(self, model, head_means, pairs, d_head, device):
        self.model = model
        self.head_means = head_means
        self.d_head = d_head
        self.device = device
        # group heads by layer
        self.by_layer = {}
        for l, h in pairs:
            self.by_layer.setdefault(l, []).append(h)
        self.handles = []

    def _make(self, l, heads):
        d = self.d_head
        means = self.head_means[l].to(self.device)
        def hook(module, args):
            (x,) = args
            x = x.clone()
            for h in heads:
                x[..., h * d:(h + 1) * d] = means[h].to(x.dtype)
            return (x,)
        return hook

    def __enter__(self):
        for l, heads in self.by_layer.items():
            self.handles.append(output_projection(self.model, l).register_forward_pre_hook(self._make(l, heads)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def random_matched_pairs(pairs, n_layers, n_heads, rng):
    """A random set of (layer, head) pairs of the same size, disjoint from pairs."""
    chosen = set(pairs)
    out = []
    while len(out) < len(pairs):
        cand = (int(rng.integers(0, n_layers)), int(rng.integers(0, n_heads)))
        if cand not in chosen:
            chosen.add(cand)
            out.append(cand)
    return out

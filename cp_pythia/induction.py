"""Induction score on repeated-token diagnostics, and the task key-slot score. Torch.

The induction score for head (l, h) is the attention from a token to the position right
after its previous occurrence, read from the returned attention weights. The key-slot
score reads the attention from the query start position to the value-slot position of the
key-slot binding, on task-structured single-hop sequences.
"""

import numpy as np
import torch


@torch.no_grad()
def head_attentions(model, ids, device):
    """Return the per-layer attention weights for a batch. Requires eager attention."""
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    out = model(input_ids=ids, output_attentions=True, use_cache=False)
    if out.attentions is None:
        raise RuntimeError("model returned no attentions; load with attn_implementation='eager'")
    return out.attentions  # tuple length n_layers, each [B, n_heads, S, S]


@torch.no_grad()
def induction_scores(model, diag_ids, block_len, device, n_layers, n_heads):
    """Mean prefix-matching attention per head over a repeated-token diagnostic."""
    attns = head_attentions(model, diag_ids, device)
    n = block_len
    # query at position n+i attends to position i+1 (token after the prior occurrence)
    q_idx = torch.arange(n, 2 * n - 1, device=device)        # n .. 2n-2
    t_idx = torch.arange(1, n, device=device)                # 1 .. n-1
    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    for l in range(n_layers):
        A = attns[l].float()                                  # [B, H, S, S]
        vals = A[:, :, q_idx, t_idx]                          # [B, H, n-1]
        scores[l] = vals.mean(dim=(0, 2)).cpu().numpy()
    return scores


def top_heads(scores, k):
    """Return the top-k (layer, head) pairs by score, descending."""
    flat = [((l, h), scores[l, h]) for l in range(scores.shape[0]) for h in range(scores.shape[1])]
    flat.sort(key=lambda x: x[1], reverse=True)
    return [lh for lh, _ in flat[:k]]


@torch.no_grad()
def keyslot_scores(model, batch, device, n_layers, n_heads):
    """Mean attention from the query start position to the key-slot value position per head.

    batch must be a single-hop forward batch so value_slot_pos is set for every row.
    """
    vsp = np.asarray(batch["value_slot_pos"])
    keep = vsp >= 0
    ids = batch["input_ids"][keep]
    vsp = torch.as_tensor(vsp[keep], dtype=torch.long, device=device)
    pp = batch["pred_pos"]
    attns = head_attentions(model, ids, device)
    rows = torch.arange(ids.shape[0], device=device)
    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    for l in range(n_layers):
        A = attns[l].float()                                  # [B, H, S, S]
        Ap = A[:, :, pp, :]                                    # [B, H, S], select the query position
        vals = Ap[rows, :, vsp]                                # [B, H], gather the value-slot column
        scores[l] = vals.mean(dim=0).cpu().numpy()
    return scores

"""Masked target loss and accuracy with the next-token off-by-one. Torch.

A causal model's logits at position t predict the token at t+1, so the target at the
final position P is read from logits at P-1, the query start-content position.
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_accuracy(model, batch, content_ids, device, amp_dtype=None):
    """Return strict full-vocab accuracy, content-restricted accuracy, and target loss."""
    ids = torch.as_tensor(batch["input_ids"], dtype=torch.long, device=device)
    tgt = torch.as_tensor(batch["target"], dtype=torch.long, device=device)
    pp = batch["pred_pos"]
    ctx = torch.as_tensor(np.asarray(content_ids), dtype=torch.long, device=device)

    ctx_mgr = torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype) if amp_dtype else _null()
    with ctx_mgr:
        logits = model(input_ids=ids, use_cache=False).logits[:, pp, :].float()  # [B, V]

    loss = F.cross_entropy(logits, tgt).item()
    acc_full = (logits.argmax(-1) == tgt).float().mean().item()
    sub = logits[:, ctx]                                   # [B, C]
    pred_tok = ctx[sub.argmax(-1)]
    acc_content = (pred_tok == tgt).float().mean().item()
    return {"acc_full": acc_full, "acc_content": acc_content, "loss": loss}


@torch.no_grad()
def eval_task(model, gen, task, n_batches, batch_size, content_ids, device, rng, amp_dtype=None):
    """Average accuracy over n_batches, plus the content-shuffled floor for the same task."""
    real, floor = [], []
    for _ in range(n_batches):
        b = gen.batch(task, batch_size, rng, shuffle=False)
        real.append(eval_accuracy(model, b, content_ids, device, amp_dtype))
        bf = gen.batch(task, batch_size, rng, shuffle=True)
        floor.append(eval_accuracy(model, bf, content_ids, device, amp_dtype))
    out = {k: float(np.mean([r[k] for r in real])) for k in real[0]}
    fl = {f"floor_{k}": float(np.mean([r[k] for r in floor])) for k in floor[0]}
    out.update(fl)
    out["excess_content"] = out["acc_content"] - out["floor_acc_content"]
    return out


def masked_target_loss(logits, batch, device):
    """Training loss, cross-entropy at the target position only. logits [B, S, V]."""
    tgt = torch.as_tensor(batch["target"], dtype=torch.long, device=device)
    pp = batch["pred_pos"]
    return F.cross_entropy(logits[:, pp, :].float(), tgt)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False

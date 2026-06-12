"""Held-out pretraining perplexity probe. Torch and datasets.

Detects forgetting during continued training so a rewarm that only acquires the
composition by damaging the base model is flagged rather than counted as recovery.
Only the regression relative to the base checkpoint matters, so any fixed neutral
corpus suffices. A Pile-deduped validation slice is the principled choice.
"""

import numpy as np
import torch
import torch.nn.functional as F


def build_blocks(tokenizer, n_blocks=64, block_len=512, dataset="wikitext", config="wikitext-103-raw-v1"):
    """Tokenise a fixed neutral corpus into fixed-length blocks, once, and cache the ids."""
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split="validation")
    text = "\n\n".join(t for t in ds["text"] if t and not t.isspace())
    ids = tokenizer(text, return_tensors="np")["input_ids"][0]
    need = n_blocks * block_len
    if len(ids) < need:
        reps = need // len(ids) + 1
        ids = np.tile(ids, reps)
    ids = ids[:need].reshape(n_blocks, block_len)
    return ids.astype(np.int64)


@torch.no_grad()
def perplexity(model, blocks, device, micro_batch=8):
    """Mean negative log likelihood per token and its exponential over the fixed blocks."""
    nlls, ntok = 0.0, 0
    for i in range(0, blocks.shape[0], micro_batch):
        b = torch.as_tensor(blocks[i:i + micro_batch], dtype=torch.long, device=device)
        logits = model(input_ids=b, use_cache=False).logits[:, :-1, :].float()
        tgt = b[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
        nlls += loss.item()
        ntok += tgt.numel()
    mean_nll = nlls / max(ntok, 1)
    return {"nll": mean_nll, "ppl": float(np.exp(mean_nll))}

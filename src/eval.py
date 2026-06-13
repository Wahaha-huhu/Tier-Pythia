"""Evaluation metrics.

Primary behavioural metric (matches the toy): EXCESS over a chance floor.
  acc   = P(argmax next-token at x_start == correct answer C)
  floor = 1 / |distinct in-context content tokens|   (uniform guess among candidates)
  excess = acc - floor

Mechanism transfer (ports toy Exp 4-6):
  cand_mass = probability mass the final logits place on in-context content tokens
              (the "candidate-set / format" stage that improves before routing).
  logit lens = per-layer candidate-restricted argmax at the x_start position; we record
              how often it equals the intermediate B and the answer C across depth.
"""
import numpy as np
import torch


@torch.no_grad()
def evaluate_variant(model, task, cfg, hop, n_batches, pool=None, L=None,
                     n_distractors=None, seed_offset=0):
    """Accuracy on an OOD variant of the task (held-out tokens / longer chains / distractors).
    Primary metric is full-vocab argmax accuracy == answer; floor = 1/(L+1) for reference."""
    model.eval()
    rng = np.random.default_rng(cfg.eval_seed + hop + seed_offset)
    tot = correct = 0
    floor_sum = mass_sum = 0.0
    for _ in range(n_batches):
        b = task.batch(cfg.eval_batch_size, hop, rng, pool=pool, L=L, n_distractors=n_distractors)
        ids = b["input_ids"].to(cfg.device)
        C = b["C_ids"].to(cfg.device)
        cand = b["cand_ids"].to(cfg.device)
        logits = model(input_ids=ids[:, :-1]).logits[:, -1, :]
        pred = logits.argmax(-1)
        correct += (pred == C).sum().item()
        tot += C.numel()
        floor_sum += (1.0 / cand.shape[1]) * C.numel()
        mass_sum += logits.softmax(-1).gather(1, cand).sum(1).sum().item()
    acc = correct / tot
    floor = floor_sum / tot
    return dict(acc=acc, floor=floor, excess=acc - floor, cand_mass=mass_sum / tot, n=tot)


@torch.no_grad()
def evaluate(model, task, cfg, n_batches, hop, do_lens=False):
    model.eval()
    rng = np.random.default_rng(cfg.eval_seed + hop)  # identical eval set every call

    tot = 0
    correct = 0
    floor_sum = 0.0
    mass_sum = 0.0
    lens_B = None
    lens_C = None
    lens_n = 0

    for _ in range(n_batches):
        b = task.batch(cfg.eval_batch_size, hop, rng)
        input_ids = b["input_ids"].to(cfg.device)
        C_ids = b["C_ids"].to(cfg.device)
        B_ids = b["B_ids"].to(cfg.device)
        cand_ids = b["cand_ids"].to(cfg.device)

        trunc = input_ids[:, :-1]                       # context + query, ending at x_start
        out = model(input_ids=trunc, output_hidden_states=do_lens)
        final_logits = out.logits[:, -1, :]             # next-token prediction at x_start
        pred = final_logits.argmax(-1)

        correct += (pred == C_ids).sum().item()
        tot += C_ids.numel()
        floor_sum += (1.0 / cand_ids.shape[1]) * C_ids.numel()

        probs = final_logits.softmax(-1)
        mass_sum += probs.gather(1, cand_ids).sum(1).sum().item()

        if do_lens:
            hs = out.hidden_states                      # tuple len (num_layers + 1)
            ln = model.gpt_neox.final_layer_norm
            W = model.embed_out
            if lens_B is None:
                Lp = len(hs)
                lens_B = [0] * Lp
                lens_C = [0] * Lp
            for li, h in enumerate(hs):
                lg = W(ln(h[:, -1, :]))                  # logit lens at x_start: [B, V]
                cl = lg.gather(1, cand_ids)              # restrict to candidates: [B, k]
                loc = cl.argmax(1)
                pid = cand_ids.gather(1, loc.unsqueeze(1)).squeeze(1)
                lens_C[li] += (pid == C_ids).sum().item()
                if hop == 2:
                    lens_B[li] += (pid == B_ids).sum().item()
            lens_n += trunc.shape[0]

    acc = correct / tot
    floor = floor_sum / tot
    res = dict(acc=acc, floor=floor, excess=acc - floor, cand_mass=mass_sum / tot, n=tot)
    if do_lens:
        res["lens_C"] = [c / lens_n for c in lens_C]
        if hop == 2:
            res["lens_B"] = [c / lens_n for c in lens_B]
    return res

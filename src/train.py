"""Continued-training of a late Pythia checkpoint for one experimental arm.

One arm = (hop, schedule, seed). The checkpoint is the maximally-decayed model; we
continue training on the synthetic task under a constant LR (short warmup) set by the
schedule. Crucially we ALWAYS evaluate BOTH Hop-1 and Hop-2 throughout training, so the
curves show selectivity (Hop-1 rises, Hop-2 may not) and recoverability (Hop-2 jumps
under rewarm). Because HF checkpoints carry no optimiser state, our 'rewarm' arm is a
rewarm+RESET by construction -- which the toy showed is equivalent to plain rewarm.
"""
import numpy as np
import torch

from .model_utils import load_model
from .eval import evaluate


def _make_warmup_constant(optimizer, warmup):
    def fn(step):
        return (step + 1) / warmup if step < warmup else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def train_arm(cfg, task, hop, schedule, seed):
    torch.manual_seed(seed)
    model = load_model(cfg, dtype=torch.float32)  # sdpa for speed; lens uses hidden_states (impl-agnostic)
    model.train()

    lr = cfg.lr_for(schedule)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=cfg.adam_betas, eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    sch = _make_warmup_constant(opt, cfg.warmup_steps)
    rng = np.random.default_rng(seed)

    total_steps = cfg.steps_for(hop)
    curve = []
    for step in range(total_steps):
        b = task.batch(cfg.batch_size, hop, rng)
        input_ids = b["input_ids"].to(cfg.device)
        labels = b["labels"].to(cfg.device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sch.step()

        if (step % cfg.eval_every == 0) or (step == total_steps - 1):
            e1 = evaluate(model, task, cfg, cfg.train_eval_batches, hop=1)
            e2 = evaluate(model, task, cfg, cfg.train_eval_batches, hop=2)
            curve.append(dict(
                step=step, loss=float(loss.item()),
                hop1_acc=e1["acc"], hop1_excess=e1["excess"],
                hop2_acc=e2["acc"], hop2_excess=e2["excess"],
            ))
            model.train()
            print(
                f"    [{schedule:>10} hop{hop} s{seed}] step {step:>5}  loss {loss.item():7.3f}  "
                f"h1_acc {e1['acc']:.3f}  h2_acc {e2['acc']:.3f}"
            )

    # Final, larger eval. Logit lens computed on Hop-2 (the interesting mode).
    f1 = evaluate(model, task, cfg, cfg.final_eval_batches, hop=1)
    f2 = evaluate(model, task, cfg, cfg.final_eval_batches, hop=2, do_lens=True)

    result = dict(
        hop=hop, schedule=schedule, seed=seed, lr=lr,
        curve=curve, final_hop1=f1, final_hop2=f2,
    )
    del model
    torch.cuda.empty_cache()
    return result

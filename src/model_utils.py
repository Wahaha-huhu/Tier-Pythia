"""Load Pythia checkpoints with HuggingFace (the actual GPT-NeoX parameterisation).

We deliberately avoid any reparameterisation (no LN folding etc.) so that
"continued training of a real checkpoint" is faithful. Content/marker symbols are
fed as raw token IDs, so the tokenizer is never needed.
"""
import torch
from transformers import GPTNeoXForCausalLM


def load_model(cfg, revision=None, attn_impl=None, dtype=torch.float32):
    """Load a Pythia checkpoint.

    attn_impl=None  -> try 'sdpa' (fast) then fall back to 'eager'.
    attn_impl='eager' -> required when you need output_attentions (induction analysis).
    """
    rev = revision if revision is not None else cfg.late_revision
    impls = [attn_impl] if attn_impl is not None else ["sdpa", "eager"]
    last_err = None
    for impl in impls:
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                cfg.model_name, revision=rev, torch_dtype=dtype, attn_implementation=impl
            )
            model.to(cfg.device)
            model.config.use_cache = False
            return model
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err

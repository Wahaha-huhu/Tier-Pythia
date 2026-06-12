"""Load a Pythia checkpoint and introspect its shape. Torch and transformers.

Eager attention is forced so that attention weights are returned for the induction score.
"""

import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from .config import PYTHIA_REPO_PREFIX


_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_pythia(model_name, revision_step, device="cuda", dtype="bfloat16", eager=True):
    """Return (model, tokenizer, info) for EleutherAI/<model_name> at step<revision_step>."""
    repo = PYTHIA_REPO_PREFIX + model_name
    rev = f"step{revision_step}"
    kwargs = dict(revision=rev, torch_dtype=_DTYPE[dtype])
    if eager:
        kwargs["attn_implementation"] = "eager"
    model = GPTNeoXForCausalLM.from_pretrained(repo, **kwargs).to(device)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    info = {
        "n_layers": model.config.num_hidden_layers,
        "n_heads": model.config.num_attention_heads,
        "d_model": model.config.hidden_size,
        "d_head": model.config.hidden_size // model.config.num_attention_heads,
        "vocab": model.config.vocab_size,
        "repo": repo,
        "revision": rev,
    }
    return model, tok, info


def attention_module(model, layer):
    """Return the attention submodule for a layer. Verify against your transformers version."""
    return model.gpt_neox.layers[layer].attention


def output_projection(model, layer):
    """Return the attention output projection whose input is the concatenated per-head context."""
    attn = attention_module(model, layer)
    if hasattr(attn, "dense"):
        return attn.dense
    # fallback for renamed projections, find the linear with in_features == hidden_size
    hidden = model.config.hidden_size
    for name, mod in attn.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.in_features == hidden and mod.out_features == hidden:
            if "query_key_value" not in name:
                return mod
    raise RuntimeError("could not locate the attention output projection; inspect the module")

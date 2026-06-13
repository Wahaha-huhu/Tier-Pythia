"""Edge-of-stability probe: top Hessian eigenvalue of the (masked-CE) loss.

We estimate lambda_max(H) by power iteration with Hessian-vector products
(double backprop). This characterizes the curvature of the region the optimizer
is traversing. Tracked over continued training at several LRs, it tests whether the
upper edge of the acquisition band is an instability barrier: at a too-large LR the
optimizer cannot descend into the sharp routing minimum (loss oscillates), which
should coincide with a large eta * lambda_max.

Notes:
- Computed in fp32, no autocast (double-backward is finicky under bf16 autocast).
- Requires eager attention: the SDPA/flash kernels generally do NOT support the
  second-order derivative, so the caller must load the model with attn_implementation="eager".
- Power iteration returns the dominant eigenvalue by magnitude; near the relevant
  regions of these losses it is the large positive top eigenvalue (sharpness).
- The classical "eta * lambda_max ~ 2" edge is a GD result. AdamW preconditions the
  geometry, so treat the raw product as a comparative diagnostic across LRs, not as an
  exact threshold (the Adam threshold differs and depends on beta2). The clean
  preconditioned-sharpness version is a natural follow-up.
"""
import torch


def _dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def _norm(a):
    return _dot(a, a).clamp_min(0).sqrt()


@torch.enable_grad()
def top_hessian_eigenvalue(model, input_ids, labels, n_iter=20, tol=1e-3):
    params = [p for p in model.parameters() if p.requires_grad]
    vs = [torch.randn_like(p) for p in params]
    nv = _norm(vs)
    vs = [v / nv for v in vs]

    was_training = model.training
    model.eval()  # deterministic; Pythia has no dropout anyway

    eig = 0.0
    for i in range(n_iter):
        loss = model(input_ids=input_ids, labels=labels).loss
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        gv = _dot(grads, vs)
        Hv = torch.autograd.grad(gv, params, retain_graph=False, allow_unused=True)
        Hv = [h.detach() if h is not None else torch.zeros_like(p) for h, p in zip(Hv, params)]
        eig_new = _dot(Hv, vs).item()                 # Rayleigh quotient v^T H v (v unit)
        nh = _norm(Hv)
        if nh.item() == 0.0:
            eig = eig_new
            break
        vs = [h / nh for h in Hv]
        if i > 0 and abs(eig_new - eig) < tol * (abs(eig_new) + 1e-8):
            eig = eig_new
            break
        eig = eig_new

    if was_training:
        model.train()
    return float(eig)

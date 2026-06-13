"""Edge-of-stability probe: top Hessian eigenvalue of the (masked-CE) loss.

lambda_max(H) is estimated by power iteration where each Hessian-vector product is
computed by a CENTRAL FINITE DIFFERENCE of the gradient:

    H v  ~=  ( grad L(theta + eps v) - grad L(theta - eps v) ) / (2 eps)

This deliberately avoids exact double-backprop: the second derivative of GPTNeoX's
causal-masked softmax attention (the -inf masking) is numerically ill-conditioned and
yields garbage. The finite-difference version needs only first-order gradients, so it
(a) is numerically robust and (b) works under the SDPA attention kernel -- the same
kernel used everywhere else -- so it does NOT change training dynamics.

Safety: parameters are perturbed in place and ALWAYS restored (finally block), so the
measurement cannot corrupt the run it is measuring. Computed in fp32, no autocast.

lambda_max characterizes the sharpness of the region the optimizer is traversing.
The classical 'eta * lambda_max ~ 2' edge is a GD result; AdamW preconditions the
geometry (different, beta2-dependent threshold), so treat eta*lambda_max as a
COMPARATIVE diagnostic across LRs, not an exact threshold.
"""
import torch


def _dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def _norm(a):
    return _dot(a, a).clamp_min(0).sqrt()


@torch.enable_grad()
def top_hessian_eigenvalue(model, input_ids, labels, n_iter=20, eps_rel=3e-3, tol=1e-2):
    params = [p for p in model.parameters() if p.requires_grad]
    was_training = model.training
    model.eval()  # deterministic; Pythia has no dropout

    def grad_at_current():
        loss = model(input_ids=input_ids, labels=labels).loss
        g = torch.autograd.grad(loss, params, allow_unused=True)  # first-order only
        return [gi.detach() if gi is not None else torch.zeros_like(p)
                for gi, p in zip(g, params)]

    # random unit init direction
    vs = [torch.randn_like(p) for p in params]
    nv = _norm(vs)
    vs = [v / nv for v in vs]

    # finite-difference step, scaled to the parameter norm
    theta_norm = _norm([p.detach() for p in params]).item()
    eps = eps_rel * (theta_norm + 1e-12)

    orig = [p.detach().clone() for p in params]
    eig = 0.0
    try:
        for i in range(n_iter):
            with torch.no_grad():
                for p, v in zip(params, vs):
                    p.add_(eps * v)                 # theta + eps v
            g_plus = grad_at_current()
            with torch.no_grad():
                for p, v in zip(params, vs):
                    p.add_(-2.0 * eps * v)          # theta - eps v
            g_minus = grad_at_current()
            with torch.no_grad():
                for p, o in zip(params, orig):
                    p.copy_(o)                      # restore theta
            Hv = [(gp - gm) / (2.0 * eps) for gp, gm in zip(g_plus, g_minus)]
            eig_new = _dot(Hv, vs).item()           # Rayleigh quotient v^T H v (v unit)
            nh = _norm(Hv)
            if nh.item() == 0.0:
                eig = eig_new
                break
            vs = [h / nh for h in Hv]
            if i > 0 and abs(eig_new - eig) < tol * (abs(eig_new) + 1e-8):
                eig = eig_new
                break
            eig = eig_new
    finally:
        with torch.no_grad():
            for p, o in zip(params, orig):
                p.copy_(o)                          # ALWAYS restore -> cannot corrupt training
        if was_training:
            model.train()
    return float(eig)

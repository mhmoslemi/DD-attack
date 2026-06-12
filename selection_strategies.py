"""
selection_strategies.py

A menu of base-selection criteria for the selection ablation. Every criterion
scores the base-class candidates and returns the global indices of the N_p
"best" bases, under one convention: SMALLER score = better base, so all criteria
share a single torch.topk(..., largest=False) path.

Grouping (the axis the ablation is really about):
  target-AWARE  (use the target x_t):
    pixel_l2  : input-space L2 distance to x_t            (naive baseline)
    feat_l2   : penultimate-feature L2 to x_t             (= 'ours' with lam=0)
    feat_cos  : penultimate-feature cosine to x_t
    grad_cos  : last-layer loss-gradient cosine to x_t's adversarial gradient
    ours      : feat_l2 + lam * margin   (the proposed rule)
    anti      : 'ours' reversed -> worst bases            (sanity control)
  target-AGNOSTIC (ignore x_t, score generic importance):
    gradnorm  : last-layer gradient norm  (GraNd, Paul et al. 2021)
    el2n      : ||softmax - onehot||_2     (EL2N, Paul et al. 2021)
    margin    : low margin toward y_adv    (the proposed rule's margin term alone)
  control:
    random    : uniform over the base class

grad_cos / gradnorm / el2n use the closed-form LAST-LAYER gradient (no autograd):
for features f, logits z=Wf+b, softmax p, label e=onehot(y), the last-layer grad
is [vec((p-e) f^T); (p-e)], so
  ||grad|| = ||p-e|| * sqrt(||f||^2 + 1)
  <grad_a, grad_b> = <a,b> (<f_a,f_b> + 1),   a=p_a-e, b=p_b-e
which is exact and cheap.

All surrogate-based scores are standardized per surrogate and averaged over the
ensemble, matching the proposed rule. Place next to networks.py / utils.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


CRITERIA = ['random', 'pixel_l2', 'feat_l2', 'feat_cos', 'grad_cos',
            'gradnorm', 'el2n', 'margin', 'ours', 'anti']

TARGET_AWARE = {'pixel_l2', 'feat_l2', 'feat_cos', 'grad_cos', 'ours', 'anti'}
TARGET_AGNOSTIC = {'gradnorm', 'el2n', 'margin'}


def embed_of(net):
    return net.module.embed if isinstance(net, nn.DataParallel) else net.embed


def _standardize(v, eps=1e-8):
    return (v - v.mean()) / (v.std() + eps)


def _margin_toward(z, y_adv):
    """z_adv - max_{c != y_adv} z_c. LOW (or negative) = not confidently y_adv."""
    z_adv = z[:, y_adv].clone()
    z_o = z.clone()
    z_o[:, y_adv] = float('-inf')
    return z_adv - z_o.max(dim=1).values


@torch.no_grad()
def select_bases(criterion, surrogates, images_norm, labels, x_t_norm, y_adv,
                 N_p, lam, device, denorm=None, bs=512, generator=None):
    """Return global indices of N_p selected bases of class y_adv (smaller score
    = better). `denorm` (callable normalized -> [0,1]) is required for pixel_l2."""
    if criterion not in CRITERIA:
        raise ValueError('unknown criterion %r; choose from %s' % (criterion, CRITERIA))

    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    cand = images_norm[cls_idx]
    n = len(cand)

    # ---- controls / input space (no surrogate) ----------------------------
    if criterion == 'random':
        perm = torch.randperm(n, device=device, generator=generator)
        return cls_idx[perm[:N_p]]

    if criterion == 'pixel_l2':
        if denorm is None:
            raise ValueError('pixel_l2 needs denorm')
        xt01 = denorm(x_t_norm.unsqueeze(0)).clamp(0.0, 1.0)
        score = torch.empty(n, device=device)
        for i in range(0, n, bs):
            b01 = denorm(cand[i:i + bs]).clamp(0.0, 1.0)
            score[i:i + bs] = ((b01 - xt01).flatten(1) ** 2).sum(1)   # smaller = closer
        sel = torch.topk(score, N_p, largest=False).indices
        return cls_idx[sel]

    # ---- surrogate-based: standardize per net, average over ensemble ------
    score = torch.zeros(n, device=device)
    needs_logits = criterion in ('grad_cos', 'gradnorm', 'el2n', 'margin', 'ours', 'anti')
    needs_feat_dist = criterion in ('feat_l2', 'ours', 'anti')

    for net in surrogates:
        emb = embed_of(net)
        f_t = emb(x_t_norm.unsqueeze(0))                      # (1, D)
        z_t = net(x_t_norm.unsqueeze(0))
        C = z_t.shape[1]
        e = F.one_hot(torch.tensor(y_adv, device=device), C).float()   # (C,)
        # target last-layer residual + norms (for grad_cos)
        a_t = (z_t.softmax(1) - e).squeeze(0)                 # (C,)
        f_tv = f_t.squeeze(0)                                 # (D,)
        a_t_norm = a_t.norm()
        f_t_term = torch.sqrt(f_tv.pow(2).sum() + 1.0)

        prim = torch.empty(n, device=device)                 # single-term score
        dist = torch.empty(n, device=device)                 # ours/anti: distance
        marg = torch.empty(n, device=device)                 # ours/anti: margin

        for i in range(0, n, bs):
            b = cand[i:i + bs]
            f = emb(b)
            z = net(b) if needs_logits else None

            if needs_feat_dist:
                d = ((f - f_t) ** 2).sum(1)

            if criterion == 'feat_l2':
                prim[i:i + bs] = d
            elif criterion == 'feat_cos':
                prim[i:i + bs] = 1.0 - F.cosine_similarity(f, f_t.expand_as(f), dim=1)
            elif criterion == 'grad_cos':
                a = z.softmax(1) - e
                num = (a @ a_t) * ((f @ f_tv) + 1.0)
                den = a.norm(dim=1) * torch.sqrt(f.pow(2).sum(1) + 1.0) * a_t_norm * f_t_term
                prim[i:i + bs] = 1.0 - num / (den + 1e-8)     # smaller = more aligned
            elif criterion == 'gradnorm':
                a = z.softmax(1) - e
                prim[i:i + bs] = -(a.norm(dim=1) * torch.sqrt(f.pow(2).sum(1) + 1.0))
            elif criterion == 'el2n':
                prim[i:i + bs] = -(z.softmax(1) - e).norm(dim=1)   # larger = harder = better
            elif criterion == 'margin':
                prim[i:i + bs] = _margin_toward(z, y_adv)     # smaller margin = better
            elif criterion in ('ours', 'anti'):
                dist[i:i + bs] = d
                marg[i:i + bs] = _margin_toward(z, y_adv)

        if criterion in ('ours', 'anti'):
            score += _standardize(dist) + lam * _standardize(marg)
        else:
            score += _standardize(prim)

    largest = (criterion == 'anti')                           # 'anti' picks the WORST
    sel = torch.topk(score, N_p, largest=largest).indices
    return cls_idx[sel]

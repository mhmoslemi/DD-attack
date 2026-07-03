"""
eval_standard_nodistill.py

Targeted clean-label poison evaluated in the STANDARD train-from-scratch setting
(NO victim distillation), under MetaPoison's victim protocol.

Two crafting objectives, selected by --attack:
  fc        : feature collision (Eq.2) -- match the target's penultimate feature.
  gradmatch : gradient matching, Witches'-Brew style (Geiping et al. 2020) --
              align the ensemble-averaged poison-gradient with the target's
              adversarial gradient via (1 - cosine), signed-Adam, DiffAugment per
              step, R restarts, keep the best delta. Standard literature recipe.

Surrogates (the frozen feature extractors used by selection and crafting) are
trained on the DISTILLED set S loaded from --syn_data_path. The victim is trained
from scratch on the full poisoned 50k set; no condensation at victim time.

Pipeline:
  S0. load distilled S; train K surrogates on S (evaluate_synset, DSA).
  S1. (optional) clean victim pool -> baseline CTA + per-target clean ASR.
  per (class_pair, target):
    1. select N_p base images of class y_adv (Eq.1, random ablation via --random_select).
    2. craft L_inf<=eps poisons (fc or gradmatch).
    3. inject (clean-label) into a fresh clone of the full normalized train set.
    4. train M victims FROM SCRATCH (MetaPoison: 200 ep, lr 0.1, bs 125, x0.1 @100/150).
    5. record CTA and whether target -> y_adv.

Place next to utils.py / networks.py.

Example (gradient matching, standard):
  python eval_standard_nodistill.py \
      --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
      --surrogate_model ConvNet --model ConvNetBN --class_pairs dog-bird \
      --attack gradmatch --epsilon 0.0313725 --pgd_steps 250 --pgd_alpha 0.0039216 \
      --restarts 8 --num_surrogates 10 --surrogate_epochs 1000 \
      --num_targets 10 --num_victims 6 \
      --victim_epochs 200 --victim_lr 0.1 --victim_bs 125 --victim_decay 100 150 \
      --clean_baseline --target_select random --seed 0
"""

import argparse
import csv
import json
import os
import queue
import sys
import time
import warnings
from types import SimpleNamespace

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from utils import (get_dataset, get_network, DiffAugment, ParamDiffAug, get_time,
                   evaluate_synset, TensorDataset)


# --------------------------------------------------------------------------- #
# logging: tee stdout/stderr to a file, flushing every line (zero delay)
# --------------------------------------------------------------------------- #
class _Tee:
    """Mirror writes to the console and a log file, flushing immediately so the
    log is updated line-by-line with no buffering delay (good for tail -f)."""
    def __init__(self, path, mode='a'):
        self.file = open(path, mode, buffering=1)        # line-buffered text file
        self.console = sys.__stdout__

    def write(self, data):
        self.console.write(data)
        self.console.flush()
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def embed_of(net):
    return net.module.embed if isinstance(net, nn.DataParallel) else net.embed


def multi_embed_of(net):
    m = net.module if isinstance(net, nn.DataParallel) else net
    return m.intermediate_embeds


def standardize(v, eps=1e-8):
    return (v - v.mean()) / (v.std() + eps)


def parse_pair(pair, class_names):
    """'dog-bird' -> (y_adv=class('dog'), target_class=class('bird'))."""
    a, b = pair.split('-')
    return class_names.index(a), class_names.index(b)


def stack_dataset(dst, device):
    imgs = torch.stack([dst[i][0] for i in range(len(dst))]).to(device)   # normalized
    labs = torch.tensor([dst[i][1] for i in range(len(dst))],
                        dtype=torch.long, device=device)
    return imgs, labs


def _flat_grad(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def _cosine(a, b, eps=1e-8):
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


# --------------------------------------------------------------------------- #
# from-scratch trainer for VICTIMS (MetaPoison schedule: no aug, no weight decay)
# --------------------------------------------------------------------------- #
def train_from_scratch(net, images, labels, epochs, lr, bs, decay_at, device,
                       weight_decay=0.0, aug=False, dsa_strategy=None, dsa_param=None):
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss().to(device)
    N = images.shape[0]
    cur_lr = lr
    decay_at = set(decay_at)
    for ep in range(epochs):
        if ep in decay_at:
            cur_lr *= 0.1
            for g in opt.param_groups:
                g['lr'] = cur_lr
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            img = images[idx]
            lab = labels[idx]
            if aug and dsa_strategy:
                img = DiffAugment(img, dsa_strategy, param=dsa_param)
            opt.zero_grad()
            loss = crit(net(img), lab)
            loss.backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def test_acc(net, images, labels, device, bs=512):
    net.eval()
    c = 0
    for i in range(0, len(images), bs):
        c += (net(images[i:i + bs]).argmax(1) == labels[i:i + bs]).sum().item()
    return c / len(images)


@torch.no_grad()
def predict_target(net, x_t_norm):
    net.eval()
    return int(net(x_t_norm.unsqueeze(0)).argmax(1).item())


# --------------------------------------------------------------------------- #
# surrogate ensemble trained on the DISTILLED set S (reuse evaluate_synset)
# --------------------------------------------------------------------------- #
def train_surrogates_on_full(train_imgs, train_labs, test_imgs, test_labs,
                             channel, num_classes, im_size, args, device):
    """Train surrogates on the full real training set (no DSA, SGD like victims)."""
    import time as _time
    crit = nn.CrossEntropyLoss().to(device)
    nets = []
    requires = (args.attack == 'gradmatch')
    for i in range(args.num_surrogates):
        net = get_network(args.surrogate_model, channel, num_classes, im_size)
        t0 = _time.time()
        net = train_from_scratch(net, train_imgs, train_labs, args.surrogate_epochs,
                                 args.surrogate_lr, args.surrogate_bs, [],
                                 device, weight_decay=0.0)
        t_train = _time.time() - t0
        net.eval()
        loss_sum, acc_sum, n = 0.0, 0, 0
        with torch.no_grad():
            for j in range(0, len(train_imgs), 512):
                imgs = train_imgs[j:j + 512]
                labs = train_labs[j:j + 512]
                out = net(imgs)
                loss_sum += crit(out, labs).item() * len(imgs)
                acc_sum += (out.argmax(1) == labs).sum().item()
                n += len(imgs)
        train_loss = loss_sum / n
        train_acc_val = acc_sum / n
        test_acc_val = test_acc(net, test_imgs, test_labs, device)
        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f '
              'train acc = %.4f, test acc = %.4f'
              % (get_time(), i, args.surrogate_epochs, int(t_train),
                 train_loss, train_acc_val, test_acc_val))
        for p in net.parameters():
            p.requires_grad_(requires)
        nets.append(net)
    return nets


def train_surrogates_on_syn(image_syn, label_syn, test_imgs, test_labs,
                            channel, num_classes, im_size, args, dsa_param, device):
    testloader = DataLoader(TensorDataset(test_imgs, test_labs),
                            batch_size=512, shuffle=False, num_workers=0)
    syn_args = SimpleNamespace(
        device=device, lr_net=args.surrogate_lr,
        epoch_eval_train=args.surrogate_epochs, batch_train=args.surrogate_bs,
        dsa=True, dsa_strategy=args.dsa_strategy, dsa_param=dsa_param)
    nets = []
    for i in range(args.num_surrogates):
        net = get_network(args.surrogate_model, channel, num_classes, im_size)
        net, _, acc = evaluate_synset(i, net, image_syn.clone(), label_syn.clone(),
                                      testloader, syn_args)
        net.eval()
        # NOTE: for fc we freeze params (grad only w.r.t. delta); for gradmatch we
        # need d L / d theta, so leave params trainable when --attack gradmatch.
        requires = (args.attack == 'gradmatch')
        for p in net.parameters():
            p.requires_grad_(requires)
        nets.append(net)
    return nets


# --------------------------------------------------------------------------- #
# curvature-leverage score (theory-based 3rd base-selection signal):
#   curv(x) = || C_x^T (H + lambda I)^{-1} g_t ||_1
# computed matrix-free via Hessian-vector products + conjugate gradient.
# --------------------------------------------------------------------------- #
def _flat_params_grad(loss, params, create_graph=False):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grads])


def _conj_grad(matvec, b, iters, tol=1e-8):
    """Solve A x = b for symmetric (ideally PD) A given only matvec(v) = A v,
    via conjugate gradient. Stops early on convergence or on a non-positive
    curvature direction (indefinite A), returning the best iterate so far."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rs_old = torch.dot(r, r)
    for _ in range(iters):
        Ap = matvec(p)
        pAp = torch.dot(p, Ap)
        if pAp <= 1e-12:                       # indefinite / singular: stop here
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if rs_new.sqrt() <= tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def _inverse_hvp(net, g_t, params, train_imgs, train_labs, damping, cg_iters,
                 hessian_bs, device):
    """v_t = (H + damping I)^{-1} g_t, matrix-free: the training-loss Hessian H is
    accessed only through Hessian-vector products on a FIXED random minibatch
    (so CG sees a constant operator), and the system is solved by conjugate grad.
    Requires grad enabled and net params with requires_grad=True (caller's job)."""
    crit_mean = nn.CrossEntropyLoss().to(device)
    n_tr = train_imgs.shape[0]
    hidx = torch.randperm(n_tr, device=device)[:min(hessian_bs, n_tr)]
    g_train = _flat_params_grad(crit_mean(net(train_imgs[hidx]), train_labs[hidx]),
                                params, create_graph=True)

    def hvp(v):                                      # (H + damping I) v, matrix-free
        Hv = torch.autograd.grad(g_train, params, grad_outputs=v, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in Hv]) + damping * v

    return _conj_grad(hvp, g_t, cg_iters).detach()


def curvature_leverage_scores(net, cand, y_base, x_t_norm, y_adv,
                              train_imgs, train_labs, damping, cg_iters,
                              hessian_bs, cand_bs, device):
    """Per-candidate  || C_x^T (H + lambda I)^{-1} g_t ||_1   (higher = better).

      g_t = grad_theta CE(net(x_target), y_adv)        target adversarial gradient
      H   = Hessian of the training CE loss w.r.t. theta (never formed: HVP + CG)
      v_t = (H + lambda I)^{-1} g_t                     solved by conjugate gradient
      C_x = grad_x grad_theta CE(net(x_base), y_base)   mixed input/param 2nd deriv
      C_x^T v_t = grad_x ( grad_theta CE(net(x_base), y_base) . v_t )

    A batch's SUMMED loss lets one input-gradient pass recover every sample's
    C_x^T v_t at once (cross terms vanish), so curvature for cand_bs candidates
    costs a single double-backward. Note eps*||C_x^T v_t||_1 is exactly the best
    influence achievable by an L_inf-eps perturbation, so this score ranks base
    points by their best-case poison."""
    crit_mean = nn.CrossEntropyLoss().to(device)
    crit_sum = nn.CrossEntropyLoss(reduction='sum').to(device)
    params = [p for p in net.parameters()]
    orig_req = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    try:
        with torch.enable_grad():
            # target adversarial gradient g_t (constant w.r.t. candidates)
            y_t = torch.full((1,), y_adv, dtype=torch.long, device=device)
            g_t = _flat_params_grad(crit_mean(net(x_t_norm.unsqueeze(0)), y_t),
                                    params).detach()

            # inverse-Hessian-vector product  v_t = (H + lambda I)^{-1} g_t
            v_t = _inverse_hvp(net, g_t, params, train_imgs, train_labs,
                               damping, cg_iters, hessian_bs, device)

            # per-candidate  || C_x^T v_t ||_1
            scores = torch.empty(len(cand), device=device)
            for i in range(0, len(cand), cand_bs):
                xb = cand[i:i + cand_bs].detach().clone().requires_grad_(True)
                yb = torch.full((xb.shape[0],), y_base, dtype=torch.long, device=device)
                g_c = _flat_params_grad(crit_sum(net(xb), yb), params, create_graph=True)
                cx_t_v = torch.autograd.grad(torch.dot(g_c, v_t), xb)[0]
                scores[i:i + xb.shape[0]] = cx_t_v.abs().flatten(1).sum(1)
    finally:
        for p, req in zip(params, orig_req):
            p.requires_grad_(req)
        net.zero_grad(set_to_none=True)
    return scores.detach()


# --------------------------------------------------------------------------- #
# selection (Eq.1): ensemble-averaged, standardized  d(x) + lambda * M(x)
#                   (optional 3rd term: inverse-Hessian curvature leverage)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def select_base(surrogates, images_norm, labels, x_t_norm, y_adv, N_p, lam, device,
                base_dist='l2', multilayer=False,
                curv=False, lam_curv=1.0, curv_damping=1.0, curv_cg_iters=10,
                curv_hessian_bs=512, curv_cand_bs=128):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    cand = images_norm[cls_idx]
    score = torch.zeros(len(cls_idx), device=device)
    for net in surrogates:
        emb = embed_of(net)
        f_t = emb(x_t_norm.unsqueeze(0))
        if multilayer:
            m_emb = multi_embed_of(net)
            f_t_list = m_emb(x_t_norm.unsqueeze(0))           # list of (1, D_l)
            dists_l_batches = [[] for _ in f_t_list]
        ds, ms = [], []
        for i in range(0, len(cand), 512):
            b = cand[i:i + 512]
            fb = emb(b)
            if base_dist == 'cosine':
                d = 1.0 - F.cosine_similarity(fb, f_t.expand(len(b), -1), dim=1)
            else:  # l2
                d = ((fb - f_t) ** 2).sum(dim=1)
            z = net(b)
            z_adv = z[:, y_adv].clone()
            z_o = z.clone()
            z_o[:, y_adv] = float('-inf')
            m = z_adv - z_o.max(dim=1).values                     # margin toward y_adv
            if multilayer:
                f_list = m_emb(b)
                for l, (fl, f_tl) in enumerate(zip(f_list, f_t_list)):
                    dists_l_batches[l].append(((fl - f_tl) ** 2).sum(1))
            ds.append(d)
            ms.append(m)
        if multilayer:
            dists_l = [torch.cat(batches) for batches in dists_l_batches]
            d_combined = sum(standardize(dl) for dl in dists_l) / len(dists_l)
            score += d_combined + lam * standardize(torch.cat(ms))
        else:
            score += standardize(torch.cat(ds)) + lam * standardize(torch.cat(ms))
        if curv:
            cs = curvature_leverage_scores(
                net, cand, y_adv, x_t_norm, y_adv, images_norm, labels,
                curv_damping, curv_cg_iters, curv_hessian_bs, curv_cand_bs, device)
            score -= lam_curv * standardize(cs)   # higher leverage -> lower (better) score
    score /= len(surrogates)
    sel = torch.topk(score, k=N_p, largest=False).indices          # least conf + closest
    return cls_idx[sel]


def select_base_random(labels, y_adv, N_p, device):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    perm = torch.randperm(len(cls_idx), device=device)
    return cls_idx[perm[:N_p]]


# --------------------------------------------------------------------------- #
# EXACT influence-function base selection (ported from main_IF_exact.py)
#   score(z) = grad_theta l(z_t)^T (H + damping I)^{-1} grad_theta l(z)
# i.e. the curvature-aware influence of upweighting the CLEAN base point z on the
# target loss -- the same derivation as smart-select but KEEPING the inverse
# Hessian. H is the surrogate's empirical-risk Hessian (never formed: CG over
# HVPs), and <s, grad l(z)> is read off for every candidate at once via a central
# finite difference of the per-sample loss along s (two forward passes, no
# per-sample param gradients). Largest score = best base to poison.
# --------------------------------------------------------------------------- #
def _select_params(net, last_layer):
    m = net.module if isinstance(net, nn.DataParallel) else net
    ps = [p for p in m.parameters()]
    return ps[-2:] if last_layer else ps        # last linear (w,b) heuristic


@torch.no_grad()
def _per_sample_ce(net, imgs, y_const, device, bs=512):
    """Per-sample CE of `imgs` against the constant label y_const (no reduction)."""
    net.eval()
    out = []
    for i in range(0, len(imgs), bs):
        b = imgs[i:i + bs]
        yb = torch.full((len(b),), y_const, dtype=torch.long, device=device)
        out.append(F.cross_entropy(net(b), yb, reduction='none'))
    return torch.cat(out)


def _cg_inverse_hvp(net, params, g, hess_imgs, hess_labs, device,
                    damping, iters, tol, hess_bs):
    """Solve (H + damping*I) s = g with conjugate gradients, where H is the
    Hessian of the mean CE on (hess_imgs, hess_labs) w.r.t. `params`. Returns
    (s ~ (H+damping I)^{-1} g, cg_iters_used). HVPs reuse one retained double-
    backward graph. params/g/s are lists of per-parameter tensors."""
    net.eval()
    n = len(hess_imgs)
    idx = torch.arange(n, device=device)
    if hess_bs and hess_bs < n:
        idx = idx[torch.randperm(n, device=device)[:hess_bs]]
    xb, yb = hess_imgs[idx], hess_labs[idx]
    loss = F.cross_entropy(net(xb), yb)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    def hvp(vec):
        dot = sum((gg * vv).sum() for gg, vv in zip(grads, vec))
        hv = torch.autograd.grad(dot, params, retain_graph=True)
        return [h + damping * vv for h, vv in zip(hv, vec)]

    x = [torch.zeros_like(gi) for gi in g]
    r = [gi.clone() for gi in g]
    p = [gi.clone() for gi in g]
    rs_old = sum((ri * ri).sum() for ri in r)
    g_norm = torch.sqrt(rs_old).clamp_min(1e-12)
    used = iters
    for it in range(iters):
        Ap = hvp(p)
        pAp = sum((pi * Api).sum() for pi, Api in zip(p, Ap))
        alpha = rs_old / (pAp + 1e-12)
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * Api for ri, Api in zip(r, Ap)]
        rs_new = sum((ri * ri).sum() for ri in r)
        if torch.sqrt(rs_new) <= tol * g_norm:
            used = it + 1
            break
        beta = rs_new / (rs_old + 1e-12)
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new
    return [xi.detach() for xi in x], used


def select_base_influence(surrogates, images_norm, labels, x_t_norm, y_adv, N_p,
                          hess_imgs, hess_labs, device,
                          damping=0.01, cg_iters=100, cg_tol=1e-4, fd_h=1e-2,
                          last_layer=False, hess_bs=0, max_surrogates=0, verbose=False):
    """Rank candidates of class y_adv by the EXACT (curvature-aware) influence on
    the target loss, score(z) = grad l(z_t)^T H^{-1} grad l(z), averaged over the
    surrogate ensemble. Returns indices into the full training set (largest score
    = most negative target-loss change = best base to poison)."""
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    cand = images_norm[cls_idx]
    nets = surrogates if not max_surrogates else surrogates[:max_surrogates]
    total = torch.zeros(len(cls_idx), device=device)
    cg_used = []
    for net in nets:
        params = _select_params(net, last_layer)
        orig_req = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(True)
        net.eval()
        # target gradient g = grad_th CE(net(x_t), y_adv)
        y_t = torch.tensor([y_adv], dtype=torch.long, device=device)
        loss_t = F.cross_entropy(net(x_t_norm.unsqueeze(0)), y_t)
        g = [gi.detach() for gi in torch.autograd.grad(loss_t, params)]
        # s = (H + damping I)^{-1} g  via CG over HVPs
        s, used = _cg_inverse_hvp(net, params, g, hess_imgs, hess_labs, device,
                                  damping, cg_iters, cg_tol, hess_bs)
        cg_used.append(used)
        # per-candidate score <s, grad_th l(z)> by central finite difference
        snorm = torch.sqrt(sum((si * si).sum() for si in s)).clamp_min(1e-12)
        shat = [si / snorm for si in s]
        orig = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, o, sh in zip(params, orig, shat):
                p.copy_(o + fd_h * sh)
            lp = _per_sample_ce(net, cand, y_adv, device)
            for p, o, sh in zip(params, orig, shat):
                p.copy_(o - fd_h * sh)
            lm = _per_sample_ce(net, cand, y_adv, device)
            for p, o in zip(params, orig):
                p.copy_(o)
        total += snorm * (lp - lm) / (2.0 * fd_h)
        for p, req in zip(params, orig_req):
            p.requires_grad_(req)
    total /= len(nets)
    if verbose:
        print('      [exact] CG iters used per surrogate: %s' % cg_used)
    sel = torch.topk(total, k=N_p, largest=True).indices
    return cls_idx[sel]


@torch.no_grad()
def filter_correct_targets(surrogates, test_imgs, t_idx_all, target_class, device,
                            conf_trim=0.0):
    """Keep only candidates the surrogate ensemble already classifies as target_class.

    conf_trim: fraction in [0, 1) of the lowest-confidence correct targets to drop,
               so the pool excludes the hardest correctly-classified images.
    Returns a CPU LongTensor of filtered indices, sorted by confidence descending.
    """
    cands = test_imgs[t_idx_all]          # (N, C, H, W), already on device
    sum_logits = None
    for net in surrogates:
        logits = net(cands)
        sum_logits = logits if sum_logits is None else sum_logits + logits
    avg_probs = F.softmax(sum_logits / len(surrogates), dim=1)   # (N, C)
    correct = avg_probs.argmax(1) == target_class                 # bool mask
    conf    = avg_probs[:, target_class]                          # softmax score
    correct_local = correct.nonzero(as_tuple=True)[0]            # indices into cands
    if len(correct_local) == 0:
        return t_idx_all.new_empty(0)
    # sort by confidence descending (easiest first)
    order = conf[correct_local].argsort(descending=True)
    correct_local = correct_local[order]
    # drop the bottom conf_trim fraction (hardest end)
    if conf_trim > 0.0:
        keep = max(1, int(len(correct_local) * (1.0 - conf_trim)))
        correct_local = correct_local[:keep]
    return t_idx_all[correct_local.cpu()]


@torch.no_grad()
def select_easy_targets(surrogates, test_imgs, test_labs, y_adv, n_targets, device,
                        easiness=1.0):
    """Pick test samples along an easy<->hard spectrum for flipping toward y_adv.

    Unlike the random/first selection (which only draws from the pair's
    target_class), the candidate pool here is EVERY test image whose true label
    is not y_adv -- any other class is allowed. Each candidate is scored by the
    surrogate ensemble's softmax probability on y_adv: a high score means the
    model is already close to calling it y_adv, i.e. it is easy to attack.
    Samples the clean ensemble already classifies as y_adv are dropped (there is
    nothing left to flip).

    ``easiness`` in [0, 1] slides an n_targets-wide window over the full ranking
    (easiest first): 1.0 returns the n EASIEST candidates, 0.0 the n HARDEST,
    0.5 the middle band. Returns the chosen test indices, easiest first.
    """
    pool = (test_labs != y_adv).nonzero(as_tuple=True)[0]        # label != y_adv
    if len(pool) == 0:
        return []
    cands = test_imgs[pool]
    sum_logits = None
    for net in surrogates:
        logits = net(cands)
        sum_logits = logits if sum_logits is None else sum_logits + logits
    avg_probs = F.softmax(sum_logits / len(surrogates), dim=1)   # (N, C)
    p_adv = avg_probs[:, y_adv]                                  # easiness score
    keep = (avg_probs.argmax(1) != y_adv).nonzero(as_tuple=True)[0]   # not yet flipped
    if len(keep) == 0:                                          # degenerate: keep all
        keep = torch.arange(len(pool), device=pool.device)
    ranked = keep[p_adv[keep].argsort(descending=True)]         # local idx, easiest first
    M = len(ranked)
    n = min(n_targets, M)
    e = min(max(float(easiness), 0.0), 1.0)
    start = int(round((1.0 - e) * (M - n)))                     # 1->easiest, 0->hardest
    chosen_local = ranked[start:start + n]
    return pool[chosen_local].cpu().tolist()


# --------------------------------------------------------------------------- #
# crafting (fc): per-sample L_inf PGD feature collision over the ensemble
# --------------------------------------------------------------------------- #
def craft_fc(surrogates, base01, x_t_norm, norm, eps, steps, alpha, device,
             single_surrogate=False):
    nets = [surrogates[0]] if single_surrogate else surrogates
    base01 = base01.detach()
    with torch.no_grad():
        f_tgts = [embed_of(n)(x_t_norm.unsqueeze(0)).detach() for n in nets]
    delta = torch.empty_like(base01).uniform_(-eps, eps)
    delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01).detach().requires_grad_(True)
    obj_val = float('nan')
    for t in range(steps):
        x_adv_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
        loss = 0.0
        for n, f_t in zip(nets, f_tgts):
            f = embed_of(n)(x_adv_norm)
            loss = loss + F.mse_loss(f, f_t.expand_as(f))
        loss = loss / len(nets)
        obj_val = loss.item()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - alpha * grad.sign()
            delta = delta.clamp_(-eps, eps)
            delta = torch.clamp(base01 + delta, 0.0, 1.0) - base01
        delta = delta.detach().requires_grad_(True)
    return torch.clamp(base01 + delta.detach(), 0.0, 1.0), obj_val


# --------------------------------------------------------------------------- #
# crafting (gradmatch): Witches'-Brew style gradient matching (Geiping et al. 2020)
#   minimize  1 - cos( grad_theta CE(x_t, y_adv) ,  grad_theta CE(poisons, y_adv) )
#   ensemble-averaged, signed-Adam on delta, DiffAugment per step, R restarts,
#   second-order (create_graph=True), keep the lowest-objective delta.
# --------------------------------------------------------------------------- #
def craft_gradmatch(surrogates, base01, x_t_norm, y_adv, norm, eps, step, iters,
                    restarts, device, dsa_strategy=None, dsa_param=None,
                    single_surrogate=False, fast=False):
    nets = [surrogates[0]] if single_surrogate else surrogates
    for net in nets:                       # need d L / d theta
        for p in net.parameters():
            p.requires_grad_(True)
    crit = nn.CrossEntropyLoss().to(device)
    y_t = torch.full((1,), y_adv, dtype=torch.long, device=device)
    y_p = torch.full((base01.shape[0],), y_adv, dtype=torch.long, device=device)

    # target adversarial gradient per net (constant in delta) -> precompute, detach
    g_targets = []
    for net in nets:
        params = [p for p in net.parameters()]
        loss_t = crit(net(x_t_norm.unsqueeze(0)), y_t)
        g_t = torch.autograd.grad(loss_t, params)
        g_targets.append(_flat_grad([g.detach() for g in g_t]))

    base01 = base01.detach()
    use_dsa = dsa_strategy not in (None, '', 'none', 'None')
    best_delta, best_obj = None, float('inf')

    for r in range(restarts):
        delta = torch.empty_like(base01).uniform_(-eps, eps)
        delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01).detach().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=step)
        for t in range(iters):
            x_adv_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
            if use_dsa:
                seed = int(torch.randint(0, 100000, (1,)).item())
                x_adv_norm = DiffAugment(x_adv_norm, dsa_strategy, seed=seed,
                                         param=dsa_param)
            if fast:
                # First-order approximation: compute param grads and delta grad in
                # one backward pass instead of building a second-order graph.
                # Avoids create_graph=True (~2-3x faster per iteration).
                obj_val = 0.0
                grad_accum = torch.zeros_like(delta)
                for net, g_t in zip(nets, g_targets):
                    params = [p for p in net.parameters() if p.requires_grad]
                    loss_p = crit(net(x_adv_norm), y_p)
                    all_grads = torch.autograd.grad(loss_p, params + [delta])
                    g_p = _flat_grad(list(all_grads[:-1])).detach()
                    grad_accum = grad_accum + all_grads[-1].detach()
                    obj_val += (1.0 - _cosine(g_p, g_t)).item()
                obj_val /= len(nets)
                grad_accum /= len(nets)
                opt.zero_grad()
                delta.grad = grad_accum.sign()             # signed Adam
            else:
                # Exact second-order: differentiate cosine(g_p, g_t) through to delta.
                obj = 0.0
                for net, g_t in zip(nets, g_targets):
                    params = [p for p in net.parameters()]
                    loss_p = crit(net(x_adv_norm), y_p)
                    g_p = _flat_grad(torch.autograd.grad(loss_p, params, create_graph=True))
                    obj = obj + (1.0 - _cosine(g_p, g_t))
                obj = obj / len(nets)
                grad = torch.autograd.grad(obj, delta)[0]
                opt.zero_grad()
                delta.grad = grad.sign()                   # signed Adam
                obj_val = obj.item()
            opt.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
                delta.data = torch.clamp(base01 + delta, 0.0, 1.0) - base01
            if obj_val < best_obj:
                best_obj = obj_val
                best_delta = delta.detach().clone()

    return torch.clamp(base01 + best_delta, 0.0, 1.0), best_obj


# --------------------------------------------------------------------------- #
# influence crafting: directly drive the deployed-poison target-loss influence
#   I(delta) = - < v_t , grad_theta CE(net(base+delta), y_adv) > ,
#   v_t = (H + lambda I)^{-1} grad_theta CE(net(x_target), y_adv)
# as negative as possible (i.e. MAXIMIZE the alignment after the minus sign).
# The L_inf one-step optimum is delta* = eps * sign(C_x^T v_t); we run signed PGD
# so the cross-derivative C_x is re-evaluated at base+delta and the [0,1] box and
# eps-ball are respected. Reuses the matrix-free inverse-HVP (HVP + CG).
# --------------------------------------------------------------------------- #
def craft_influence(surrogates, base01, x_t_norm, y_adv, norm, eps, step, iters,
                    restarts, device, train_imgs, train_labs, damping, cg_iters,
                    hessian_bs, single_surrogate=False):
    nets = [surrogates[0]] if single_surrogate else surrogates
    for net in nets:                       # need d/d theta and d/d delta
        for p in net.parameters():
            p.requires_grad_(True)
    crit = nn.CrossEntropyLoss().to(device)
    crit_sum = nn.CrossEntropyLoss(reduction='sum').to(device)
    y_t = torch.full((1,), y_adv, dtype=torch.long, device=device)
    y_p = torch.full((base01.shape[0],), y_adv, dtype=torch.long, device=device)

    # v_t = (H + lambda I)^{-1} g_t per net (constant in delta) -> precompute, detach
    v_ts = []
    for net in nets:
        params = [p for p in net.parameters()]
        g_t = _flat_params_grad(crit(net(x_t_norm.unsqueeze(0)), y_t), params).detach()
        v_ts.append(_inverse_hvp(net, g_t, params, train_imgs, train_labs,
                                 damping, cg_iters, hessian_bs, device))

    base01 = base01.detach()
    best_delta, best_obj = None, float('inf')
    for r in range(restarts):
        delta = torch.empty_like(base01).uniform_(-eps, eps)
        delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01).detach().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=step)
        for t in range(iters):
            x_adv_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
            # minimize I = -alignment  (== maximize the post-minus influence term);
            # summed loss over poisons so each delta_i gets its own C_{x_i}^T v_t
            obj = 0.0
            for net, v_t in zip(nets, v_ts):
                params = [p for p in net.parameters()]
                g_b = _flat_params_grad(crit_sum(net(x_adv_norm), y_p), params,
                                        create_graph=True)
                obj = obj - torch.dot(g_b, v_t)
            obj = obj / len(nets)
            grad = torch.autograd.grad(obj, delta)[0]
            opt.zero_grad()
            delta.grad = grad.sign()                   # signed Adam (== eps*sign step)
            opt.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
                delta.data = torch.clamp(base01 + delta, 0.0, 1.0) - base01
            if obj.item() < best_obj:
                best_obj = obj.item()
                best_delta = delta.detach().clone()

    return torch.clamp(base01 + best_delta, 0.0, 1.0), best_obj


# --------------------------------------------------------------------------- #
# victim training pipelines (ABLATIONS)
# ---------------------------------------------------------------------------
# The poison (selection + crafting) is produced ONCE per target; these
# pipelines only change how a victim is trained on the already-poisoned set,
# so we can re-use the same poison across every ablation.
#
# A pipeline is given as "name" or "name:k1=v1,k2=v2". Supported names:
#   standard      plain SGD (no aug / no wd) -- the already-done baseline
#   diffaug       DiffAugment (DSA) every batch          [strategy=...]
#   mixup         input mixup                            [alpha=1.0]
#   cutmix        CutMix                                 [alpha=1.0]
#   advtrain      PGD/Madry adversarial training         [eps=,alpha=,steps=7]
#   dpsgd         gradient shaping / approx DP-SGD       [clip=1.0,noise=0.01]
#   labelsmooth   label-smoothing CE                     [smoothing=0.1]
#   weightdecay   SGD with weight decay                  [wd=5e-4]
#   gradclip      global grad-norm clipping              [max_norm=1.0]
# Any pipeline also accepts base-schedule overrides: lr=, bs=, epochs=, decay=.
# --------------------------------------------------------------------------- #
def _num(v):
    """Parse a CLI value to int if integral, else float, else leave as str."""
    try:
        return int(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def build_pipeline_cfg(spec, args):
    """Turn a pipeline spec string into a normalized config dict."""
    if ':' in spec:
        name, rest = spec.split(':', 1)
        overrides = {}
        for kv in rest.split(','):
            if not kv:
                continue
            k, v = kv.split('=')
            overrides[k] = _num(v)
    else:
        name, overrides = spec, {}
    name = name.lower()
    cfg = {'name': name, 'spec': spec}

    if name in ('standard', 'plain', 'simple'):
        cfg['name'] = 'standard'
    elif name in ('aug', 'diffaug', 'dsa'):
        cfg['name'] = 'diffaug'
        cfg['aug'] = True
    elif name == 'mixup':
        cfg['mixup_alpha'] = overrides.pop('alpha', 1.0)
    elif name == 'cutmix':
        cfg['cutmix_alpha'] = overrides.pop('alpha', 1.0)
    elif name in ('advtrain', 'adv', 'madry'):
        cfg['name'] = 'advtrain'
        cfg['adv_eps'] = overrides.pop('eps', args.epsilon)
        cfg['adv_alpha'] = overrides.pop('alpha', max(args.epsilon / 4.0, 1.0 / 255.0))
        cfg['adv_steps'] = int(overrides.pop('steps', 7))
    elif name in ('dpsgd', 'gradshaping', 'gradshape'):
        cfg['name'] = 'dpsgd'
        cfg['max_norm'] = overrides.pop('clip', 1.0)
        cfg['noise'] = overrides.pop('noise', 0.01)
    elif name in ('labelsmooth', 'ls'):
        cfg['name'] = 'labelsmooth'
        cfg['smoothing'] = overrides.pop('smoothing', 0.1)
    elif name in ('weightdecay', 'wd'):
        cfg['name'] = 'weightdecay'
        cfg['wd'] = overrides.pop('wd', 5e-4)
    elif name == 'gradclip':
        cfg['max_norm'] = overrides.pop('max_norm', 1.0)
    else:
        raise ValueError('unknown victim pipeline: %r' % spec)

    cfg.update(overrides)        # lr/bs/epochs/decay/strategy/aug pass-through
    return cfg


def _rand_bbox(H, W, lam):
    r = np.sqrt(max(0.0, 1.0 - lam))
    cut_w, cut_h = int(W * r), int(H * r)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))
    return x1, y1, x2, y2


def _pgd_adv(net, x_norm, y, eps, alpha, steps, m, s):
    """L_inf PGD adversarial examples (Madry) in [0,1] pixel space, returned
    normalized. Generated with BN frozen (net.eval()) for stable statistics."""
    x01 = (x_norm * s + m).detach()
    delta = torch.empty_like(x01).uniform_(-eps, eps)
    delta = (torch.clamp(x01 + delta, 0.0, 1.0) - x01).detach()
    crit = nn.CrossEntropyLoss()
    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv_norm = (torch.clamp(x01 + delta, 0.0, 1.0) - m) / s
        loss = crit(net(x_adv_norm), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta + alpha * grad.sign()).clamp_(-eps, eps)
            delta = torch.clamp(x01 + delta, 0.0, 1.0) - x01
    return ((torch.clamp(x01 + delta, 0.0, 1.0) - m) / s).detach()


def train_victim(net, images, labels, cfg, args, m, s, device, dsa_param=None):
    """From-scratch victim trainer with a configurable defense/training pipeline.
    `images` are normalized and never mutated, so a single poisoned tensor can be
    fed to every pipeline."""
    epochs = int(cfg.get('epochs', args.victim_epochs))
    lr = float(cfg.get('lr', args.victim_lr))
    bs = int(cfg.get('bs', args.victim_bs))
    decay_raw = cfg.get('decay', args.victim_decay)
    decay_at = set(decay_raw if isinstance(decay_raw, (list, tuple, set)) else [int(decay_raw)])
    wd = float(cfg.get('wd', 0.0))
    smoothing = float(cfg.get('smoothing', 0.0))
    grad_clip = float(cfg.get('max_norm', 0.0))     # gradclip / dpsgd
    dp_noise = float(cfg.get('noise', 0.0))         # dpsgd
    mixup_a = float(cfg.get('mixup_alpha', 0.0))
    cutmix_a = float(cfg.get('cutmix_alpha', 0.0))
    use_aug = bool(cfg.get('aug', False))
    aug_strategy = cfg.get('strategy', args.dsa_strategy)
    adv_eps = float(cfg.get('adv_eps', 0.0))
    adv_alpha = float(cfg.get('adv_alpha', 0.0))
    adv_steps = int(cfg.get('adv_steps', 0))

    try:
        crit = nn.CrossEntropyLoss(label_smoothing=smoothing).to(device)
    except TypeError:                                # older torch w/o label_smoothing
        crit = nn.CrossEntropyLoss().to(device)

    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    N = images.shape[0]
    cur_lr = lr
    for ep in range(epochs):
        if ep in decay_at:
            cur_lr *= 0.1
            for gp in opt.param_groups:
                gp['lr'] = cur_lr
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            img = images[idx]                        # advanced index -> fresh copy
            lab = labels[idx]

            if adv_steps > 0 and adv_eps > 0:
                net.eval()
                img = _pgd_adv(net, img, lab, adv_eps, adv_alpha, adv_steps, m, s)
                net.train()

            if use_aug and aug_strategy not in (None, '', 'none', 'None'):
                img = DiffAugment(img, aug_strategy, param=dsa_param)

            lab_b, lam = None, 1.0
            if mixup_a > 0:
                lam = float(np.random.beta(mixup_a, mixup_a))
                ridx = torch.randperm(img.size(0), device=device)
                img = lam * img + (1.0 - lam) * img[ridx]
                lab_b = lab[ridx]
            elif cutmix_a > 0:
                lam = float(np.random.beta(cutmix_a, cutmix_a))
                ridx = torch.randperm(img.size(0), device=device)
                H, W = img.shape[2], img.shape[3]
                x1, y1, x2, y2 = _rand_bbox(H, W, lam)
                img[:, :, y1:y2, x1:x2] = img[ridx, :, y1:y2, x1:x2]
                lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(H * W))
                lab_b = lab[ridx]

            opt.zero_grad()
            out = net(img)
            if lab_b is not None:
                loss = lam * crit(out, lab) + (1.0 - lam) * crit(out, lab_b)
            else:
                loss = crit(out, lab)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                if dp_noise > 0:
                    with torch.no_grad():
                        for p in net.parameters():
                            if p.grad is not None:
                                p.grad.add_(torch.randn_like(p.grad) * dp_noise * grad_clip)
            opt.step()
    net.eval()
    return net


def _poison_key(args, pair, tidx):
    """Filename for caching a crafted poison so ablations can re-use it."""
    sel = 'rand' if args.random_select else ('ml' if args.multilayer else 'sl')
    ss = 'ss' if args.single_surrogate else 'ens'
    return ('poison_%s_%s_%s_b%d_eps%d_st%d_ns%d_%s_%s_%s_t%d_seed%d.pt' % (
        args.attack, args.surrogate_model, pair, round(args.budget * 1e4),
        round(args.epsilon * 255), args.pgd_steps, args.num_surrogates,
        args.base_dist, sel, ss, tidx, args.seed))


def _safe_tag(spec):
    """Filename-safe version of a pipeline spec (e.g. 'advtrain:alpha=0.015,steps=10')."""
    return (spec.replace(':', '_').replace('=', '').replace(',', '_')
                .replace('.', 'p').replace('/', '_'))


# --------------------------------------------------------------------------- #
# parallel training across multiple GPUs (surrogates AND victims)
#
# The independent unit of work (one surrogate, or one full victim) is the unit of
# parallelism. Each worker process is pinned to a SINGLE physical GPU by setting
# CUDA_VISIBLE_DEVICES *before* CUDA initializes in the fresh (spawned) process;
# this both routes the work to that GPU and stops get_network() from auto-wrapping
# the net in nn.DataParallel (it only does so when it sees >1 GPU). Jobs are split
# round-robin over the pool, e.g. 4 jobs on GPUs [6,7] -> GPU6 does {0,2}, GPU7 {1,3}.
#
# Victim workers return a scalar result; surrogate workers return a CPU state_dict
# which the PARENT reloads onto its own GPU (the surrogates must live in the parent
# afterwards for target selection and poison crafting).
# --------------------------------------------------------------------------- #
def _set_proc_name(name):
    """Best-effort rename of this process so nvidia-smi / ps / top show `name`
    instead of the long venv python path. Uses setproctitle if installed (this
    rewrites /proc/<pid>/cmdline, which is what nvidia-smi reads), else falls back
    to prctl(PR_SET_NAME) which sets the <=15 char /proc/<pid>/comm."""
    try:
        import setproctitle
        setproctitle.setproctitle(name)
        return
    except Exception:
        pass
    try:
        import ctypes
        buf = ctypes.create_string_buffer(name.encode()[:15])
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(15, ctypes.byref(buf), 0, 0, 0)
    except Exception:
        pass


def _round_robin(n_items, gpu_pool):
    """Assign item indices 0..n_items-1 round-robin to GPUs; -> {gpu: [idx,...]}."""
    assign = {g: [] for g in gpu_pool}
    for k in range(n_items):
        assign[gpu_pool[k % len(gpu_pool)]].append(k)
    return assign


def _drain_and_join(ret_q, procs, n_expected, what):
    """Collect n_expected results off ret_q, then join. Times out + checks worker
    liveness so a crashed worker raises instead of hanging the run forever."""
    results = []
    while len(results) < n_expected:
        try:
            results.append(ret_q.get(timeout=10))
        except queue.Empty:
            if all(not p.is_alive() for p in procs):
                break
    for p in procs:
        p.join()
    if len(results) < n_expected:
        raise RuntimeError('%s: got %d/%d results; a worker crashed (see stderr above)'
                           % (what, len(results), n_expected))
    results.sort(key=lambda r: r[0])
    return results


def _start_workers(gpu_pool, n_items, worker, common_args):
    """Round-robin n_items over gpu_pool and start one spawn process per used GPU.
    Each worker is called as worker(phys_gpu, gpu_rank, item_idxs, *common_args, ret_q)."""
    ctx = mp.get_context('spawn')
    ret_q = ctx.Queue()
    assign = _round_robin(n_items, gpu_pool)
    procs = []
    for rank, g in enumerate(gpu_pool):
        idxs = assign[g]
        if not idxs:
            continue
        p = ctx.Process(target=worker,
                        args=(g, rank, idxs) + tuple(common_args) + (ret_q,))
        p.start()
        procs.append(p)
    return ret_q, procs


# ---- victims ---------------------------------------------------------------- #
def _victim_worker(phys_gpu, gpu_rank, victim_idxs, model_name, channel,
                   num_classes, im_size, poisoned_cpu, labels_cpu, x_t_cpu,
                   test_imgs_cpu, test_labs_cpu, mean, std, cfg, args,
                   dsa_param, ret_q):
    # pin to one GPU BEFORE any CUDA call in this fresh process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(phys_gpu)
    _set_proc_name('main_IF.vic.g%s' % phys_gpu)
    # stagger the first net build so get_network()'s time-based seed (ms
    # resolution) differs across GPUs -> distinct net initialisations
    time.sleep(0.07 * gpu_rank)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    poisoned  = poisoned_cpu.to(device)
    labels    = labels_cpu.to(device)
    test_imgs = test_imgs_cpu.to(device)
    test_labs = test_labs_cpu.to(device)
    x_t       = x_t_cpu.to(device)
    m = torch.tensor(mean, device=device).view(1, channel, 1, 1)
    s = torch.tensor(std, device=device).view(1, channel, 1, 1)
    for vi in victim_idxs:
        net = get_network(model_name, channel, num_classes, im_size)
        net = train_victim(net, poisoned, labels, cfg, args, m, s, device,
                           dsa_param=dsa_param)
        pred = predict_target(net, x_t)
        cta = test_acc(net, test_imgs, test_labs, device)
        ret_q.put((vi, pred, cta))
        del net
        if device == 'cuda':
            torch.cuda.empty_cache()


def train_victims_parallel(gpu_pool, num_victims, model_name, channel, num_classes,
                           im_size, poisoned_cpu, labels_cpu, x_t_cpu,
                           test_imgs_cpu, test_labs_cpu, mean, std, cfg, args,
                           dsa_param):
    """Train num_victims victims spread across gpu_pool in parallel (one process
    per GPU). Returns (preds, ctas) ordered by victim index."""
    ret_q, procs = _start_workers(
        gpu_pool, num_victims, _victim_worker,
        (model_name, channel, num_classes, im_size, poisoned_cpu, labels_cpu,
         x_t_cpu, test_imgs_cpu, test_labs_cpu, mean, std, cfg, args, dsa_param))
    results = _drain_and_join(ret_q, procs, num_victims, 'parallel victims')
    preds = [r[1] for r in results]
    ctas  = [r[2] for r in results]
    return preds, ctas


# ---- surrogates ------------------------------------------------------------- #
def _surrogate_worker_full(phys_gpu, gpu_rank, sur_idxs, model_name, channel,
                           num_classes, im_size, train_imgs_cpu, train_labs_cpu,
                           test_imgs_cpu, test_labs_cpu, args, ret_q):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(phys_gpu)
    _set_proc_name('main_IF.sur.g%s' % phys_gpu)
    time.sleep(0.07 * gpu_rank)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_imgs = train_imgs_cpu.to(device)
    train_labs = train_labs_cpu.to(device)
    test_imgs  = test_imgs_cpu.to(device)
    test_labs  = test_labs_cpu.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    for i in sur_idxs:
        net = get_network(model_name, channel, num_classes, im_size)
        t0 = time.time()
        net = train_from_scratch(net, train_imgs, train_labs, args.surrogate_epochs,
                                 args.surrogate_lr, args.surrogate_bs, [],
                                 device, weight_decay=0.0)
        t_train = int(time.time() - t0)
        net.eval()
        loss_sum, acc_sum, n = 0.0, 0, 0
        with torch.no_grad():
            for j in range(0, len(train_imgs), 512):
                out = net(train_imgs[j:j + 512])
                loss_sum += crit(out, train_labs[j:j + 512]).item() * out.shape[0]
                acc_sum += (out.argmax(1) == train_labs[j:j + 512]).sum().item()
                n += out.shape[0]
        test_acc_val = test_acc(net, test_imgs, test_labs, device)
        # return weights as numpy (pickled by value through the queue) so the
        # worker can exit without invalidating torch's shared-memory fds
        sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        ret_q.put((i, sd, t_train, loss_sum / n, acc_sum / n, test_acc_val))
        del net
        if device == 'cuda':
            torch.cuda.empty_cache()


def _surrogate_worker_syn(phys_gpu, gpu_rank, sur_idxs, model_name, channel,
                          num_classes, im_size, image_syn_cpu, label_syn_cpu,
                          test_imgs_cpu, test_labs_cpu, args, dsa_param, ret_q):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(phys_gpu)
    _set_proc_name('main_IF.sur.g%s' % phys_gpu)
    time.sleep(0.07 * gpu_rank)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_syn = image_syn_cpu.to(device)
    label_syn = label_syn_cpu.to(device)
    test_imgs = test_imgs_cpu.to(device)
    test_labs = test_labs_cpu.to(device)
    testloader = DataLoader(TensorDataset(test_imgs, test_labs),
                            batch_size=512, shuffle=False, num_workers=0)
    syn_args = SimpleNamespace(
        device=device, lr_net=args.surrogate_lr,
        epoch_eval_train=args.surrogate_epochs, batch_train=args.surrogate_bs,
        dsa=True, dsa_strategy=args.dsa_strategy, dsa_param=dsa_param)
    for i in sur_idxs:
        net = get_network(model_name, channel, num_classes, im_size)
        net, _, acc = evaluate_synset(i, net, image_syn.clone(), label_syn.clone(),
                                      testloader, syn_args)
        net.eval()
        sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        ret_q.put((i, sd, float(acc)))
        del net
        if device == 'cuda':
            torch.cuda.empty_cache()


def _rebuild_surrogates(results, model_name, channel, num_classes, im_size,
                        requires, device):
    """Reload worker-trained state_dicts onto the parent's GPU, ordered by index."""
    nets = []
    for r in results:
        i, sd = r[0], r[1]
        net = get_network(model_name, channel, num_classes, im_size)
        net.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
        net = net.to(device).eval()
        for p in net.parameters():
            p.requires_grad_(requires)
        nets.append(net)
    return nets


def train_surrogates_on_full_parallel(gpu_pool, train_imgs_cpu, train_labs_cpu,
                                       test_imgs_cpu, test_labs_cpu, channel,
                                       num_classes, im_size, args, device):
    """Parallel version of train_surrogates_on_full: trains across gpu_pool, then
    rebuilds every surrogate on the parent's GPU."""
    ret_q, procs = _start_workers(
        gpu_pool, args.num_surrogates, _surrogate_worker_full,
        (args.surrogate_model, channel, num_classes, im_size, train_imgs_cpu,
         train_labs_cpu, test_imgs_cpu, test_labs_cpu, args))
    results = _drain_and_join(ret_q, procs, args.num_surrogates, 'parallel surrogates')
    for (i, _sd, t_train, tl, ta, tea) in results:
        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f '
              'train acc = %.4f, test acc = %.4f'
              % (get_time(), i, args.surrogate_epochs, t_train, tl, ta, tea))
    return _rebuild_surrogates(results, args.surrogate_model, channel, num_classes,
                               im_size, (args.attack == 'gradmatch'), device)


def train_surrogates_on_syn_parallel(gpu_pool, image_syn_cpu, label_syn_cpu,
                                     test_imgs_cpu, test_labs_cpu, channel,
                                     num_classes, im_size, args, dsa_param, device):
    """Parallel version of train_surrogates_on_syn (trains on the distilled S)."""
    ret_q, procs = _start_workers(
        gpu_pool, args.num_surrogates, _surrogate_worker_syn,
        (args.surrogate_model, channel, num_classes, im_size, image_syn_cpu,
         label_syn_cpu, test_imgs_cpu, test_labs_cpu, args, dsa_param))
    results = _drain_and_join(ret_q, procs, args.num_surrogates, 'parallel surrogates')
    for (i, _sd, acc) in results:
        print('%s surrogate %02d trained on S (test acc = %.4f)' % (get_time(), i, acc))
    return _rebuild_surrogates(results, args.surrogate_model, channel, num_classes,
                               im_size, (args.attack == 'gradmatch'), device)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or '.', exist_ok=True)
        tee = _Tee(args.log_file)
        sys.stdout = tee
        sys.stderr = tee
        print('%s logging (line-buffered, no delay) -> %s' % (get_time(), args.log_file))

    _set_proc_name('main_IF')

    # ---- multi-GPU victim parallelism --------------------------------------
    # Decide the GPU pool BEFORE any CUDA call (the device list is frozen once
    # CUDA initializes). The pool is read from CUDA_VISIBLE_DEVICES; the parent
    # is pinned to its first GPU so its own work (surrogates, crafting, clean
    # victims) stays single-GPU and get_network() never wraps in DataParallel.
    gpu_pool, use_parallel = [], False
    if args.parallel_victims:
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
        gpu_pool = [g.strip() for g in vis.split(',') if g.strip() != ''] if vis else []
        if len(gpu_pool) > 1:
            use_parallel = True
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_pool[0]
            print('%s parallel victims ON: GPU pool=%s, parent pinned to GPU %s'
                  % (get_time(), gpu_pool, gpu_pool[0]))
        else:
            print('%s --parallel_victims set but <2 GPUs in CUDA_VISIBLE_DEVICES '
                  '(%r); running victims sequentially' % (get_time(), vis))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('%s device=%s' % (get_time(), device))
    print('%s hyperparams: %s' % (get_time(), vars(args)))

    pipelines = [build_pipeline_cfg(s, args) for s in args.victim_pipelines]
    print('%s victim pipelines (ablations): %s'
          % (get_time(), [c['spec'] for c in pipelines]))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # load pre-selected targets produced by select_targets.py (optional)
    preselected = {}
    if args.target_idx_file:
        with open(args.target_idx_file) as _f:
            preselected = json.load(_f)['pairs']
        print('%s loaded pre-selected targets from %s' % (get_time(), args.target_idx_file))
    dsa_param = ParamDiffAug()

    # ---- data (normalized, full) ------------------------------------------
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(args.dataset, args.data_path)
    train_imgs, train_labs = stack_dataset(dst_train, device)
    test_imgs, test_labs = stack_dataset(dst_test, device)
    N_total = train_imgs.shape[0]

    m = torch.tensor(mean, device=device).view(1, channel, 1, 1)
    s = torch.tensor(std, device=device).view(1, channel, 1, 1)
    norm = lambda x01: (x01 - m) / s
    denorm = lambda xn: xn * s + m

    # CPU copies shared (read-only) with the parallel workers; the train labels,
    # clean train images and test set never change, so build them once here.
    train_imgs_cpu = train_imgs.cpu() if use_parallel else None
    train_labs_cpu = train_labs.cpu() if use_parallel else None
    test_imgs_cpu = test_imgs.cpu() if use_parallel else None
    test_labs_cpu = test_labs.cpu() if use_parallel else None

    N_p = int(round(args.budget * N_total))
    print('%s N_total=%d  budget=%.4f -> N_p=%d poisons (all in y_adv class)'
          % (get_time(), N_total, args.budget, N_p))

    # ---- load distilled S and train surrogates ON IT ----------------------
    ckpt = torch.load(args.syn_data_path, map_location='cpu', weights_only=False)
    image_syn, label_syn = ckpt['data'][-1]
    image_syn = image_syn.to(device)
    label_syn = label_syn.to(device)

    _sur_tag = 'fulldata' if args.surrogate_on_full_data else 'syn'
    sur_cache = os.path.join(args.cache_dir,
        'surrogates_%s_%s_%dx%dep_seed%d' % (
            args.surrogate_model, _sur_tag,
            args.num_surrogates, args.surrogate_epochs, args.seed)
    ) if args.cache_dir else ''

    if sur_cache and all(
            os.path.exists(os.path.join(sur_cache, 'surrogate_%d.pt' % i))
            for i in range(args.num_surrogates)):
        print('\n%s === loading %d surrogates from cache: %s ==='
              % (get_time(), args.num_surrogates, sur_cache))
        surrogates = []
        requires = (args.attack == 'gradmatch')
        for i in range(args.num_surrogates):
            net = get_network(args.surrogate_model, channel, num_classes, im_size)
            net.load_state_dict(torch.load(
                os.path.join(sur_cache, 'surrogate_%d.pt' % i), map_location=device))
            net = net.to(device).eval()
            for p in net.parameters():
                p.requires_grad_(requires)
            surrogates.append(net)
    else:
        if args.surrogate_on_full_data:
            print('\n%s === training %d surrogates (%s) on FULL real data (%d ep each)%s ==='
                  % (get_time(), args.num_surrogates, args.surrogate_model,
                     args.surrogate_epochs,
                     (' across GPUs %s' % gpu_pool) if use_parallel else ''))
            if use_parallel:
                surrogates = train_surrogates_on_full_parallel(
                    gpu_pool, train_imgs_cpu, train_labs_cpu, test_imgs_cpu,
                    test_labs_cpu, channel, num_classes, im_size, args, device)
            else:
                surrogates = train_surrogates_on_full(train_imgs, train_labs,
                                                      test_imgs, test_labs,
                                                      channel, num_classes, im_size, args, device)
        else:
            print('\n%s === training %d surrogates (%s) on distilled S (%d ep each)%s ==='
                  % (get_time(), args.num_surrogates, args.surrogate_model,
                     args.surrogate_epochs,
                     (' across GPUs %s' % gpu_pool) if use_parallel else ''))
            if use_parallel:
                surrogates = train_surrogates_on_syn_parallel(
                    gpu_pool, image_syn.cpu(), label_syn.cpu(), test_imgs_cpu,
                    test_labs_cpu, channel, num_classes, im_size, args, dsa_param, device)
            else:
                surrogates = train_surrogates_on_syn(image_syn, label_syn, test_imgs, test_labs,
                                                     channel, num_classes, im_size, args,
                                                     dsa_param, device)
        if sur_cache:
            os.makedirs(sur_cache, exist_ok=True)
            for i, net in enumerate(surrogates):
                torch.save(net.state_dict(), os.path.join(sur_cache, 'surrogate_%d.pt' % i))
            print('%s  saved surrogates to %s' % (get_time(), sur_cache))

    # ---- clean victim pool PER PIPELINE -----------------------------------
    # Each training recipe (advtrain, dpsgd, diffaug, ...) has its own clean
    # accuracy ceiling, so we train clean victims with the SAME recipe and use
    # that as the method-specific baseline CTA (and per-target clean ASR).
    clean_victims_by_pipe = {}
    clean_cta_by_pipe = {}
    n_clean = args.num_victims if args.num_clean_victims is None else args.num_clean_victims
    if args.clean_baseline:
        for cfg in pipelines:
            spec = cfg['spec']
            vic_cache = os.path.join(args.cache_dir,
                'clean_victims_%s_%s_%dx%dep_seed%d' % (
                    args.model, _safe_tag(spec), n_clean, args.victim_epochs, args.seed)
            ) if args.cache_dir else ''
            cvs = []
            if vic_cache and all(
                    os.path.exists(os.path.join(vic_cache, 'victim_%d.pt' % i))
                    for i in range(n_clean)):
                print('\n%s === loading %d clean victim(s) [%s] (%s) from cache: %s ==='
                      % (get_time(), n_clean, spec, args.model, vic_cache))
                for i in range(n_clean):
                    net = get_network(args.model, channel, num_classes, im_size)
                    net.load_state_dict(torch.load(
                        os.path.join(vic_cache, 'victim_%d.pt' % i), map_location=device))
                    cvs.append(net.to(device).eval())
            else:
                print('\n%s === training %d clean victim(s) [%s] (%s) from scratch on clean data ==='
                      % (get_time(), n_clean, spec, args.model))
                for i in range(n_clean):
                    net = get_network(args.model, channel, num_classes, im_size)
                    net = train_victim(net, train_imgs, train_labs, cfg, args,
                                       m, s, device, dsa_param=dsa_param)
                    cvs.append(net)
                if vic_cache:
                    os.makedirs(vic_cache, exist_ok=True)
                    for i, net in enumerate(cvs):
                        torch.save(net.state_dict(), os.path.join(vic_cache, 'victim_%d.pt' % i))
                    print('%s  saved clean victims to %s' % (get_time(), vic_cache))
            clean_victims_by_pipe[spec] = cvs
            clean_cta_by_pipe[spec] = float(np.mean(
                [test_acc(n, test_imgs, test_labs, device) for n in cvs]))
            print('  clean baseline CTA [%s] = %.4f' % (spec, clean_cta_by_pipe[spec]))

    if args.precompute_only:
        print('%s precompute_only: done, exiting.' % get_time())
        return

    # ---- Hessian set for EXACT influence-function selection ----------------
    # = the dataset the surrogate's empirical risk is defined on (real data when
    # surrogates were trained on full data, else the distilled S). Built once.
    hess_imgs = hess_labs = None
    hess_bs = 0
    if args.exact_select:
        if args.if_hess_source == 'full' or args.surrogate_on_full_data:
            hsize = args.if_hess_size if args.if_hess_size > 0 else 2000
            perm = torch.randperm(N_total, device=device)[:hsize]
            hess_imgs, hess_labs = train_imgs[perm], train_labs[perm]
            print('%s exact-IF Hessian set = %d real train images' % (get_time(), len(hess_imgs)))
        else:
            hess_imgs, hess_labs = image_syn, label_syn
            print('%s exact-IF Hessian set = distilled S (%d images)' % (get_time(), len(hess_imgs)))
        hess_bs = args.if_hess_size if (args.if_hess_size > 0 and args.if_hess_source != 'full') else 0

    # ---- per class pair / target ------------------------------------------
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    all_rows = []
    for pair in args.class_pairs:
        y_adv, target_class = parse_pair(pair, class_names)
        print('\n%s ################ pair %s : y_adv=%d(%s)  target_class=%d(%s) ################'
              % (get_time(), pair, y_adv, class_names[y_adv],
                 target_class, class_names[target_class]))

        t_idx_all = (test_labs == target_class).nonzero(as_tuple=True)[0].cpu()
        if pair in preselected:
            chosen = preselected[pair]['indices'][:args.num_targets]
            print('  targets (preselected): %s' % chosen)
        elif args.easy_targets is not None:
            # easy-target ranking depends on the (re-trained) surrogates, so it
            # drifts run-to-run. Cache the chosen indices per (surrogate_model,
            # pair, num_targets, seed, easiness) so repeat runs attack the SAME targets.
            tcache = (os.path.join(args.target_cache_dir,
                      'easy_targets_%s_%s_n%d_seed%d_e%.2f.json'
                      % (args.surrogate_model, pair, args.num_targets, args.seed,
                         args.easy_targets))
                      if args.target_cache_dir else '')
            if tcache and os.path.exists(tcache):
                with open(tcache) as _f:
                    chosen = json.load(_f)['indices'][:args.num_targets]
                print('  targets (easy, loaded cache %s): %s'
                      % (tcache, [(int(i), class_names[int(test_labs[i])]) for i in chosen]))
            else:
                chosen = select_easy_targets(surrogates, test_imgs, test_labs, y_adv,
                                             args.num_targets, device,
                                             easiness=args.easy_targets)
                if tcache:
                    os.makedirs(args.target_cache_dir, exist_ok=True)
                    with open(tcache, 'w') as _f:
                        json.dump({'surrogate_model': args.surrogate_model, 'pair': pair,
                                   'y_adv': y_adv, 'num_targets': args.num_targets,
                                   'seed': args.seed, 'easiness': args.easy_targets,
                                   'indices': [int(i) for i in chosen],
                                   'true_labels': [int(test_labs[i]) for i in chosen]},
                                  _f, indent=2)
                    print('  targets (easy, label!=%s, saved cache %s): %s'
                          % (class_names[y_adv], tcache,
                             [(int(i), class_names[int(test_labs[i])]) for i in chosen]))
                else:
                    print('  targets (easy, label!=%s): %s'
                          % (class_names[y_adv], [(int(i), class_names[int(test_labs[i])])
                                                  for i in chosen]))
        elif args.target_select == 'random':
            perm = torch.randperm(len(t_idx_all), generator=g)[:args.num_targets]
            chosen = t_idx_all[perm].tolist()
            print('  targets (random): %s' % chosen)
        else:  # 'first'
            chosen = t_idx_all[:args.num_targets].tolist()
            print('  targets (first): %s' % chosen)

        # per-pipeline accumulators (each ablation reuses the same poisons);
        # keyed by full spec so repeated names with different knobs stay distinct
        # asr_each / cta_each hold ONE entry per (target, victim) vote so the
        # summary mean/std span all target x victim outcomes, not per-target means
        pstat = {c['spec']: {'asr_each': [], 'cta_each': [], 'clean': [],
                             'tally': np.zeros(num_classes, dtype=np.int64)}
                 for c in pipelines}

        for ti, tidx in enumerate(chosen):
            x_t_norm = test_imgs[tidx]

            # ---- selection + crafting: done ONCE per target (cache to disk) ----
            pkey = (os.path.join(args.poison_cache_dir, _poison_key(args, pair, tidx))
                    if args.poison_cache_dir else '')
            if pkey and os.path.exists(pkey):
                pc = torch.load(pkey, map_location=device)
                base_idx = pc['base_idx'].to(device)
                x_adv01 = pc['x_adv01'].to(device)
                obj, linf = pc['obj'], pc['linf']
                print('  [%s t%d/%d idx=%d] loaded cached poison (obj=%.4f linf=%.4f) %s'
                      % (pair, ti + 1, len(chosen), tidx, obj, linf, pkey))
            else:
                # 1) selection on the S-trained surrogates (or random ablation)
                if args.random_select:
                    base_idx = select_base_random(train_labs, y_adv, N_p, device)
                elif args.exact_select:
                    base_idx = select_base_influence(
                        surrogates, train_imgs, train_labs, x_t_norm, y_adv, N_p,
                        hess_imgs, hess_labs, device,
                        damping=args.if_damping, cg_iters=args.if_cg_iters,
                        cg_tol=args.if_cg_tol, fd_h=args.if_fd_h,
                        last_layer=args.if_last_layer, hess_bs=hess_bs,
                        max_surrogates=args.if_max_surrogates, verbose=args.verbose)
                else:
                    base_idx = select_base(surrogates, train_imgs, train_labs, x_t_norm,
                                           y_adv, N_p, args.lambda_margin, device,
                                           base_dist=args.base_dist,
                                           multilayer=args.multilayer,
                                           curv=args.curv_select,
                                           lam_curv=args.lambda_curv,
                                           curv_damping=args.curv_damping,
                                           curv_cg_iters=args.curv_cg_iters,
                                           curv_hessian_bs=args.curv_hessian_bs,
                                           curv_cand_bs=args.curv_cand_bs)
                # 2) craft on the same surrogates
                base01 = denorm(train_imgs[base_idx]).clamp(0.0, 1.0).detach()
                if args.attack == 'gradmatch':
                    x_adv01, obj = craft_gradmatch(
                        surrogates, base01, x_t_norm, y_adv, norm, args.epsilon,
                        args.pgd_alpha, args.pgd_steps, args.restarts, device,
                        dsa_strategy=args.dsa_strategy, dsa_param=dsa_param,
                        single_surrogate=args.single_surrogate,
                        fast=args.fast_gradmatch)
                elif args.attack == 'influence':
                    x_adv01, obj = craft_influence(
                        surrogates, base01, x_t_norm, y_adv, norm, args.epsilon,
                        args.pgd_alpha, args.pgd_steps, args.restarts, device,
                        train_imgs, train_labs, args.curv_damping, args.curv_cg_iters,
                        args.curv_hessian_bs, single_surrogate=args.single_surrogate)
                else:  # 'fc'
                    x_adv01, obj = craft_fc(
                        surrogates, base01, x_t_norm, norm, args.epsilon, args.pgd_steps,
                        args.pgd_alpha, device, single_surrogate=args.single_surrogate)
                linf = (x_adv01 - base01).abs().max().item()
                if pkey:
                    os.makedirs(args.poison_cache_dir, exist_ok=True)
                    torch.save({'base_idx': base_idx.cpu(), 'x_adv01': x_adv01.detach().cpu(),
                                'obj': obj, 'linf': linf}, pkey)
                print('  [%s t%d/%d idx=%d] %s craft_obj=%.4f linf=%.4f'
                      % (pair, ti + 1, len(chosen), tidx, args.attack, obj, linf))

            # 3) inject (clean-label) into a fresh clone of the full train set;
            #    one poisoned tensor is shared by every pipeline (none mutate it).
            poisoned = train_imgs.clone()
            poisoned[base_idx] = norm(x_adv01)
            # CPU copies handed to the parallel victim workers (per target)
            poisoned_cpu = poisoned.cpu() if use_parallel else None
            x_t_cpu = x_t_norm.cpu() if use_parallel else None

            # 4) for each ABLATION pipeline: train victims, measure ASR/CTA
            for cfg in pipelines:
                pkey_stat = cfg['spec']
                # method-specific clean ASR for this target (clean victims of same recipe)
                if args.clean_baseline:
                    cvs = clean_victims_by_pipe[pkey_stat]
                    clean_asr = 100.0 * sum(predict_target(n, x_t_norm) == y_adv
                                            for n in cvs) / len(cvs)
                    clean_cta_m = clean_cta_by_pipe[pkey_stat]
                else:
                    clean_asr, clean_cta_m = float('nan'), float('nan')
                victim_preds, victim_ctas = [], []
                if use_parallel:
                    print('    [%s] victims: %d across GPUs %s ... '
                          % (cfg['spec'], args.num_victims, gpu_pool),
                          end='', flush=True)
                    victim_preds, victim_ctas = train_victims_parallel(
                        gpu_pool, args.num_victims, args.model, channel, num_classes,
                        im_size, poisoned_cpu, train_labs_cpu, x_t_cpu,
                        test_imgs_cpu, test_labs_cpu, mean, std, cfg, args, dsa_param)
                    for pred in victim_preds:
                        pstat[pkey_stat]['tally'][pred] += 1
                    print('done')
                else:
                    print('    [%s] victims: ' % cfg['spec'], end='', flush=True)
                    for vi in range(args.num_victims):
                        net = get_network(args.model, channel, num_classes, im_size)
                        net = train_victim(net, poisoned, train_labs, cfg, args,
                                           m, s, device, dsa_param=dsa_param)
                        pred = predict_target(net, x_t_norm)
                        cta = test_acc(net, test_imgs, test_labs, device)
                        victim_preds.append(pred)
                        victim_ctas.append(cta)
                        pstat[pkey_stat]['tally'][pred] += 1
                        del net
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        sep = ', ' if vi < args.num_victims - 1 else '\n'
                        print(f'v{vi+1} done', end=sep, flush=True)

                poison_asr = 100.0 * sum(p == y_adv for p in victim_preds) / args.num_victims
                poison_cta = float(np.mean(victim_ctas))
                # record every victim individually (0/100 success, per-victim CTA)
                # so std is taken over target x victims, not over targets alone
                pstat[pkey_stat]['asr_each'].extend(
                    100.0 if p == y_adv else 0.0 for p in victim_preds)
                pstat[pkey_stat]['cta_each'].extend(float(c) for c in victim_ctas)
                pstat[pkey_stat]['clean'].append(clean_asr)

                print('    [%s | %s t%d/%d idx=%d] poison_CTA=%.4f poison_ASR=%.0f%%%s'
                      % (cfg['spec'], pair, ti + 1, len(chosen), tidx,
                         poison_cta, poison_asr,
                         ('  clean_CTA=%.4f clean_ASR=%.0f%%' % (clean_cta_m, clean_asr))
                         if args.clean_baseline else ''))

                all_rows.append({
                    'pair': pair, 'attack': args.attack, 'pipeline': cfg['spec'],
                    'y_adv': y_adv, 'target_class': target_class, 'target_idx': tidx,
                    'target_true_label': int(test_labs[tidx]),
                    'clean_cta': clean_cta_m, 'clean_asr': clean_asr,
                    'poison_cta': poison_cta, 'poison_asr': poison_asr,
                    'craft_obj': obj, 'realized_linf': linf, 'N_p': N_p,
                })

        # ---- per-pipeline summary for this pair ----
        print('\n  ==== pair %s (%s) summary over %d targets x %d victims = %d votes ===='
              % (pair, args.attack, len(chosen), args.num_victims,
                 len(chosen) * args.num_victims))
        for cfg in pipelines:
            st = pstat[cfg['spec']]
            # mean/std over every target x victim vote (std no longer per-target)
            pa, ct = np.array(st['asr_each']), np.array(st['cta_each'])
            line = ('    [%-28s] poison CTA = %.4f +/- %.4f   poison ASR = %.1f%% +/- %.1f%%'
                    % (cfg['spec'], ct.mean(), ct.std(), pa.mean(), pa.std()))
            if args.clean_baseline:
                line += ('   | clean CTA = %.4f   clean ASR = %.1f%%'
                         % (clean_cta_by_pipe[cfg['spec']], float(np.nanmean(st['clean']))))
            print(line)
            print('      tally (%s): %s' % (class_names, st['tally'].tolist()))

    # ---- persist ----------------------------------------------------------
    tag = 'standard_nodistill_%s_%s_b%d_eps%d' % (
        args.attack, args.model, round(args.budget * 1e4), round(args.epsilon * 255))
    with open(os.path.join(args.out_dir, 'results_%s.json' % tag), 'w') as f:
        json.dump({'clean_cta_by_pipeline': clean_cta_by_pipe,
                   'rows': all_rows, 'args': vars(args)}, f, indent=2)
    if all_rows:
        with open(os.path.join(args.out_dir, 'results_%s.csv' % tag), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print('\n%s wrote results_%s.{json,csv} to %s' % (get_time(), tag, args.out_dir))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Standard from-scratch (no victim distillation) eval; surrogates '
                    'trained on the distilled S; --attack fc | gradmatch.')
    # data / model
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data')
    p.add_argument('--model', type=str, default='ConvNetBN',
                   help="VICTIM arch (ConvNetBN to match MetaPoison; this repo's "
                        "ConvNetBN is depth-3, not the 6-layer Finn net)")
    p.add_argument('--out_dir', type=str, default='result/standard_nodistill')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--target_idx_file', type=str, default=None,
                   help='JSON produced by select_targets.py; overrides random/first selection')
    p.add_argument('--dsa_strategy', type=str,
                   default='color_crop_cutout_flip_scale_rotate')
    # distilled S + surrogates (selection + crafting feature extractors)
    p.add_argument('--syn_data_path', type=str,
                   default='result/res_DM_CIFAR10_ConvNet_50ipc.pt',
                   help="distilled S .pt from the step-1 DM run")
    p.add_argument('--surrogate_model', type=str, default='ConvNet',
                   help="arch trained ON the distilled S (matches how S was made)")
    p.add_argument('--num_surrogates', type=int, default=5)
    p.add_argument('--surrogate_epochs', type=int, default=1000)
    p.add_argument('--surrogate_lr', type=float, default=0.01)
    p.add_argument('--surrogate_bs', type=int, default=256)
    # attack
    p.add_argument('--attack', type=str, default='fc',
                   choices=['fc', 'gradmatch', 'influence'],
                   help="fc = feature collision (Eq.2); gradmatch = Witches'-Brew "
                        "gradient matching (Geiping et al. 2020); influence = directly "
                        "drive the inverse-Hessian target-loss influence as negative as "
                        "possible, delta -> eps*sign(C_x^T (H+lambda I)^-1 g_t) (uses the "
                        "--curv_damping/--curv_cg_iters/--curv_hessian_bs knobs)")
    p.add_argument('--class_pairs', nargs='+', default=['dog-bird', 'frog-airplane'],
                   help="MetaPoison naming 'poison-target', e.g. dog-bird frog-airplane")
    p.add_argument('--budget', type=float, default=0.01,
                   help="fraction of the FULL training set; 1%% = 500 poisons in y_adv")
    p.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    p.add_argument('--pgd_steps', type=int, default=250,
                   help="iterations per restart (both attacks)")
    p.add_argument('--pgd_alpha', type=float, default=1.0 / 255.0,
                   help="fc: PGD sign step; gradmatch: signed-Adam lr (step_size)")
    p.add_argument('--restarts', type=int, default=8,
                   help="gradmatch only: random restarts, keep best (Witches'-Brew=8)")
    p.add_argument('--lambda_margin', type=float, default=1.0)
    # protocol (MetaPoison victim side)
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--num_victims', type=int, default=6)
    p.add_argument('--target_select', type=str, default='random',
                   choices=['random', 'first'])
    p.add_argument('--easy_targets', type=float, nargs='?', const=1.0, default=None,
                   metavar='EASINESS',
                   help='instead of random/first selection from the pair target_class, '
                        'pick targets along an easy<->hard spectrum: candidates are all '
                        'test images whose label != y_adv (any other class), ranked by the '
                        'surrogate ensemble probability on y_adv (closest to flipping '
                        'first). Takes a float in [0,1]: 1=easiest targets (default when '
                        'the flag is given with no value), 0=hardest, 0.5=middle band.')
    p.add_argument('--target_cache_dir', type=str, default='',
                   help='dir to save/load the --easy_targets selection (keyed by '
                        'surrogate_model/pair/num_targets/seed). First run saves the '
                        'chosen target indices; later runs reuse them verbatim so the '
                        'attack hits the SAME targets despite surrogate re-training.')
    p.add_argument('--parallel_victims', action='store_true', default=False,
                   help='train the surrogates AND the per-target victims IN PARALLEL '
                        'across the GPUs listed in CUDA_VISIBLE_DEVICES (e.g. =6,7), '
                        'split round-robin (4 nets on 2 GPUs -> 2 per GPU). Needs '
                        '>=2 visible GPUs; the parent is pinned to the first GPU and '
                        'reloads the trained surrogates onto it for selection/crafting.')
    p.add_argument('--victim_epochs', type=int, default=200)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=125)
    p.add_argument('--victim_decay', nargs='+', type=int, default=[100, 150])
    p.add_argument('--victim_aug', action='store_true', default=False,
                   help="MetaPoison default is NO augmentation; leave off to match")
    p.add_argument('--clean_baseline', action='store_true', default=False)
    p.add_argument('--num_clean_victims', type=int, default=None,
                   help='# clean victims per pipeline for the baseline (default: num_victims). '
                        '1 is enough to read each method clean-accuracy ceiling.')
    p.add_argument('--cache_dir', type=str, default='',
                   help='directory to save/load surrogate and clean-victim checkpoints')
    p.add_argument('--precompute_only', action='store_true', default=False,
                   help='train+save surrogates/victims to --cache_dir then exit')
    p.add_argument('--base_dist', type=str, default='l2', choices=['l2', 'cosine'],
                   help='feature distance for base selection: l2 (default) or cosine')
    p.add_argument('--random_select', action='store_true', default=False,
                   help='ablation: replace scored base selection with uniform random')
    p.add_argument('--multilayer', action='store_true', default=False,
                   help='use features from all intermediate stages (not just the last layer) '
                        'for base selection distance; recommended for deep nets like VGG/ResNet')
    # third base-selection signal: inverse-Hessian curvature leverage
    #   curv(x) = || C_x^T (H + lambda I)^{-1} g_t ||_1   (matrix-free: HVP + CG)
    p.add_argument('--curv_select', action='store_true', default=False,
                   help='add inverse-Hessian curvature-leverage as a 3rd base-selection '
                        'score on top of feature distance + margin; ranks base points by '
                        '|| C_x^T (H+lambda I)^-1 g_t ||_1 (matrix-free via HVP + conj. grad)')
    p.add_argument('--lambda_curv', type=float, default=1.0,
                   help='weight of the standardized curvature-leverage term (higher leverage '
                        'is preferred); only used with --curv_select')
    p.add_argument('--curv_damping', type=float, default=1.0,
                   help='Tikhonov damping lambda in (H + lambda I) for the inverse-HVP; larger '
                        'is better-conditioned / more PD for conjugate gradient')
    p.add_argument('--curv_cg_iters', type=int, default=10,
                   help='conjugate-gradient iterations for solving (H + lambda I) v_t = g_t')
    p.add_argument('--curv_hessian_bs', type=int, default=512,
                   help='# training samples used to estimate the Hessian-vector products')
    p.add_argument('--curv_cand_bs', type=int, default=128,
                   help='candidate batch size for the double-backward curvature pass '
                        '(lower if it OOMs; the per-sample scores are batch-size invariant)')
    # EXACT influence-function base selection (curvature-aware, ported from
    #   main_IF_exact.py):  score(z) = grad l(z_t)^T (H + damping I)^-1 grad l(z)
    p.add_argument('--exact_select', action='store_true', default=False,
                   help='replace scored/curv selection with EXACT influence-function '
                        'base selection: rank class-y_adv candidates by grad l(z_t)^T '
                        'H^-1 grad l(z) (matrix-free: CG over HVPs + finite-diff directional '
                        'derivative). Mutually exclusive with --random_select.')
    p.add_argument('--if_damping', type=float, default=0.01,
                   help='Hessian damping (H + damping*I) for CG stability / non-PD nets')
    p.add_argument('--if_cg_iters', type=int, default=100,
                   help='max conjugate-gradient iterations for the inverse-HVP')
    p.add_argument('--if_cg_tol', type=float, default=1e-4,
                   help='CG relative residual tolerance (stop when ||r|| <= tol*||g||)')
    p.add_argument('--if_fd_h', type=float, default=1e-2,
                   help='central finite-difference step for the directional derivative '
                        '<s, grad l(z)> over the surrogate parameters')
    p.add_argument('--if_last_layer', action='store_true', default=False,
                   help='restrict the influence Hessian/gradients to the final linear '
                        'layer (much cheaper; the classic last-layer influence function)')
    p.add_argument('--if_hess_source', type=str, default='syn', choices=['syn', 'full'],
                   help="dataset for the Hessian: 'syn' = distilled S (matches syn "
                        "surrogates), 'full' = a random real-data subsample (auto-forced "
                        "to real data when --surrogate_on_full_data)")
    p.add_argument('--if_hess_size', type=int, default=0,
                   help='subsample size for the Hessian set (0 = all of S, or 2000 if full)')
    p.add_argument('--if_max_surrogates', type=int, default=0,
                   help='cap how many surrogates EXACT-IF averages over (0 = all)')
    p.add_argument('--verbose', action='store_true', default=False,
                   help='extra logging (e.g. exact-IF conjugate-gradient iters used)')
    p.add_argument('--single_surrogate', action='store_true', default=False,
                   help='use only the first surrogate for crafting instead of the ensemble')
    p.add_argument('--fast_gradmatch', action='store_true', default=False,
                   help='first-order approximation for gradmatch: avoids create_graph=True '
                        '(~2-3x faster per iteration; approximates the exact second-order gradient)')
    p.add_argument('--surrogate_on_full_data', action='store_true', default=False,
                   help='train surrogates on the full real training set instead of the distilled S')
    # ablations: alternate victim training pipelines (poison reused across all)
    p.add_argument('--victim_pipelines', nargs='+',
                   default=['diffaug', 'mixup', 'cutmix', 'advtrain', 'dpsgd'],
                   help='victim training ablations applied to the SAME crafted poison. '
                        'Each is "name" or "name:k=v,k=v". Names: standard, diffaug, '
                        'mixup, cutmix, advtrain, dpsgd, labelsmooth, weightdecay, gradclip. '
                        'e.g. advtrain:eps=0.0314,steps=7  dpsgd:clip=1.0,noise=0.01  '
                        'mixup:alpha=1.0  labelsmooth:smoothing=0.1')
    p.add_argument('--poison_cache_dir', type=str, default='',
                   help='dir to save/load crafted poisons (base_idx + perturbed image) '
                        'so ablations can be re-run without re-selecting/re-crafting')
    p.add_argument('--log_file', type=str, default='',
                   help='tee all stdout/stderr to this file, flushed line-by-line (no delay)')
    main(p.parse_args())


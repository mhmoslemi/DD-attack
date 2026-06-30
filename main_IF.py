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
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# selection (Eq.1): ensemble-averaged, standardized  d(x) + lambda * M(x)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def select_base(surrogates, images_norm, labels, x_t_norm, y_adv, N_p, lam, device,
                base_dist='l2', multilayer=False):
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
    score /= len(surrogates)
    sel = torch.topk(score, k=N_p, largest=False).indices          # least conf + closest
    return cls_idx[sel]


def select_base_random(labels, y_adv, N_p, device):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    perm = torch.randperm(len(cls_idx), device=device)
    return cls_idx[perm[:N_p]]


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
def select_easy_targets(surrogates, test_imgs, test_labs, y_adv, n_targets, device):
    """Pick the test samples that are *easiest* to flip toward y_adv.

    Unlike the random/first selection (which only draws from the pair's
    target_class), the candidate pool here is EVERY test image whose true label
    is not y_adv -- any other class is allowed. Each candidate is scored by the
    surrogate ensemble's softmax probability on y_adv: a high score means the
    model is already close to calling it y_adv, i.e. it is easy to attack.
    Samples the clean ensemble already classifies as y_adv are dropped (there is
    nothing left to flip). Returns a list of test indices, easiest first.
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
    order = p_adv[keep].argsort(descending=True)                # easiest first
    chosen_local = keep[order][:n_targets]
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
# main
# --------------------------------------------------------------------------- #
def main(args):
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or '.', exist_ok=True)
        tee = _Tee(args.log_file)
        sys.stdout = tee
        sys.stderr = tee
        print('%s logging (line-buffered, no delay) -> %s' % (get_time(), args.log_file))

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
            print('\n%s === training %d surrogates (%s) on FULL real data (%d ep each) ==='
                  % (get_time(), args.num_surrogates, args.surrogate_model, args.surrogate_epochs))
            surrogates = train_surrogates_on_full(train_imgs, train_labs,
                                                  test_imgs, test_labs,
                                                  channel, num_classes, im_size, args, device)
        else:
            print('\n%s === training %d surrogates (%s) on distilled S (%d ep each) ==='
                  % (get_time(), args.num_surrogates, args.surrogate_model, args.surrogate_epochs))
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
        elif args.easy_targets:
            chosen = select_easy_targets(surrogates, test_imgs, test_labs, y_adv,
                                         args.num_targets, device)
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
        pstat = {c['spec']: {'asr': [], 'cta': [], 'clean': [],
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
                else:
                    base_idx = select_base(surrogates, train_imgs, train_labs, x_t_norm,
                                           y_adv, N_p, args.lambda_margin, device,
                                           base_dist=args.base_dist,
                                           multilayer=args.multilayer)
                # 2) craft on the same surrogates
                base01 = denorm(train_imgs[base_idx]).clamp(0.0, 1.0).detach()
                if args.attack == 'gradmatch':
                    x_adv01, obj = craft_gradmatch(
                        surrogates, base01, x_t_norm, y_adv, norm, args.epsilon,
                        args.pgd_alpha, args.pgd_steps, args.restarts, device,
                        dsa_strategy=args.dsa_strategy, dsa_param=dsa_param,
                        single_surrogate=args.single_surrogate,
                        fast=args.fast_gradmatch)
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
                pstat[pkey_stat]['asr'].append(poison_asr)
                pstat[pkey_stat]['cta'].append(poison_cta)
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
            pa, ct = np.array(st['asr']), np.array(st['cta'])
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
    p.add_argument('--data_path', type=str, default='data')
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
    p.add_argument('--attack', type=str, default='fc', choices=['fc', 'gradmatch'],
                   help="fc = feature collision (Eq.2); gradmatch = Witches'-Brew "
                        "gradient matching (Geiping et al. 2020)")
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
    p.add_argument('--easy_targets', action='store_true', default=False,
                   help='instead of random/first selection from the pair target_class, '
                        'pick the targets EASIEST to attack: candidates are all test '
                        'images whose label != y_adv (any other class), ranked by the '
                        'surrogate ensemble probability on y_adv (closest to flipping first)')
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


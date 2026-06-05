"""
eval_standard_nodistill.py

Targeted clean-label feature-collision poison (selection Eq.1 + PGD feature
collision Eq.2) evaluated in the STANDARD train-from-scratch setting (NO victim
distillation), under MetaPoison's victim protocol, for direct CTA/ASR comparison
to MetaPoison Fig.4 (dog-bird / frog-airplane).

Surrogates (the frozen feature extractors used by BOTH the ensemble-averaged
selection and the PGD crafting) are trained on the DISTILLED set S loaded from
--syn_data_path (the .pt produced by the step-1 DM run, format
{'data': [[image_syn, label_syn], ...], ...}, S already in normalized space).
The victim is trained from scratch on the full poisoned 50k set; no condensation
is applied at victim time.

Pipeline:
  S0. load distilled S; train K surrogates on S (evaluate_synset, DSA), freeze.
  S1. (optional) clean victim pool from scratch on full clean data -> baseline CTA
      and per-target clean ASR.
  per (class_pair, target):
    1. select N_p base images of class y_adv  (Eq.1, ensemble-averaged over the
       S-trained surrogates, standardized).
    2. craft L_inf<=eps feature-collision poisons via PGD (Eq.2, same surrogates).
    3. inject (clean-label) into a fresh clone of the full normalized train set.
    4. train M victims FROM SCRATCH on the poisoned full set (MetaPoison schedule:
       200 ep, lr 0.1, batch 125, lr x0.1 @100/150, no aug, no weight decay).
    5. record CTA and whether target -> y_adv (one ASR vote per victim).

Aggregates over --num_targets x --num_victims votes per pair, with a MetaPoison
Fig.4-style class histogram of the target's predictions. Targets are drawn at
random from the target class of the TEST set (--target_select random; use first
for MetaPoison's exact from-scratch IDs).

Place next to utils.py / networks.py.

Example (surrogates on the distilled S you already have; MetaPoison-matched victim):
  CUDA_VISIBLE_DEVICES=0 python eval_standard_nodistill.py \
      --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
      --surrogate_model ConvNet --model ConvNetBN \
      --class_pairs dog-bird frog-airplane \
      --budget 0.01 --epsilon 0.0313725 --pgd_steps 250 --pgd_alpha 0.0039216 \
      --lambda_margin 1.0 \
      --num_surrogates 5 --surrogate_epochs 1000 \
      --num_targets 10 --num_victims 6 \
      --victim_epochs 200 --victim_lr 0.1 --victim_bs 125 --victim_decay 100 150 \
      --clean_baseline --target_select random --seed 0

  python eval_standard_nodistill.py \
      --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
      --surrogate_model ConvNet --model ConvNetBN \
      --class_pairs dog-bird \
      --budget 0.01 --epsilon 0.0313725 --pgd_steps 250 --pgd_alpha 0.0039216 \
      --lambda_margin 1.0 \
      --num_surrogates 5 --surrogate_epochs 1000 \
      --num_targets 10 --num_victims 6 \
      --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 \
      --target_select random --seed 0
      random_select


"""

import argparse
import csv
import json
import os
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
# helpers
# --------------------------------------------------------------------------- #
def embed_of(net):
    return net.module.embed if isinstance(net, nn.DataParallel) else net.embed


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
        # print('  surrogate %d/%d on S  test acc = %.4f'
        #       % (i + 1, args.num_surrogates, acc))
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        nets.append(net)
    return nets


# --------------------------------------------------------------------------- #
# selection (Eq.1): ensemble-averaged, standardized  d(x) + lambda * M(x)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def select_base(surrogates, images_norm, labels, x_t_norm, y_adv, N_p, lam, device):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError('class %d has %d images < N_p=%d' % (y_adv, len(cls_idx), N_p))
    cand = images_norm[cls_idx]
    score = torch.zeros(len(cls_idx), device=device)
    for net in surrogates:
        emb = embed_of(net)
        f_t = emb(x_t_norm.unsqueeze(0))
        ds, ms = [], []
        for i in range(0, len(cand), 512):
            b = cand[i:i + 512]
            d = ((emb(b) - f_t) ** 2).sum(dim=1)                  # ||f(x)-f(x_t)||^2
            z = net(b)
            z_adv = z[:, y_adv].clone()
            z_o = z.clone()
            z_o[:, y_adv] = float('-inf')
            m = z_adv - z_o.max(dim=1).values                     # margin toward y_adv
            ds.append(d)
            ms.append(m)
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


# --------------------------------------------------------------------------- #
# crafting (Eq.2): per-sample L_inf PGD feature collision over the ensemble
# --------------------------------------------------------------------------- #
def craft_pgd(surrogates, base01, x_t_norm, norm, eps, steps, alpha, device):
    
    surrogates = [surrogates[0]]

    base01 = base01.detach()
    with torch.no_grad():
        f_tgts = [embed_of(n)(x_t_norm.unsqueeze(0)).detach() for n in surrogates]
    delta = torch.empty_like(base01).uniform_(-eps, eps)
    delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01).detach().requires_grad_(True)
    loss_val = float('nan')
    for t in range(steps):
        x_adv_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
        loss = 0.0
        for n, f_t in zip(surrogates, f_tgts):
            f = embed_of(n)(x_adv_norm)
            loss = loss + F.mse_loss(f, f_t.expand_as(f))
        loss = loss / len(surrogates)
        loss_val = float(loss)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - alpha * grad.sign()
            delta = delta.clamp_(-eps, eps)
            delta = torch.clamp(base01 + delta, 0.0, 1.0) - base01
        delta = delta.detach().requires_grad_(True)
    return torch.clamp(base01 + delta.detach(), 0.0, 1.0), loss_val


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('%s device=%s' % (get_time(), device))
    print('%s hyperparams: %s' % (get_time(), vars(args)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
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
    # print('\n%s loading distilled S from %s' % (get_time(), args.syn_data_path))
    ckpt = torch.load(args.syn_data_path, map_location='cpu', weights_only=False)
    image_syn, label_syn = ckpt['data'][-1]
    image_syn = image_syn.to(device)
    label_syn = label_syn.to(device)
    # print('  S: %s  min=%.3f max=%.3f (expected ~[-2.5, 2.7] for normalized CIFAR)'
    #       % (tuple(image_syn.shape), image_syn.min().item(), image_syn.max().item()))

    print('\n%s === training %d surrogates (%s) on distilled S (%d ep each) ==='
          % (get_time(), args.num_surrogates, args.surrogate_model, args.surrogate_epochs))
    surrogates = train_surrogates_on_syn(image_syn, label_syn, test_imgs, test_labs,
                                         channel, num_classes, im_size, args,
                                         dsa_param, device)

    # ---- clean victim pool (baseline CTA + per-target clean ASR) ----------
    clean_victims = []
    clean_cta = None
    if args.clean_baseline:
        print('\n%s === training %d clean victims (%s) from scratch on full clean data ==='
              % (get_time(), args.num_victims, args.model))
        for i in range(args.num_victims):
            net = get_network(args.model, channel, num_classes, im_size)
            net = train_from_scratch(net, train_imgs, train_labs, args.victim_epochs,
                                     args.victim_lr, args.victim_bs, args.victim_decay,
                                     device, weight_decay=0.0, aug=args.victim_aug,
                                     dsa_strategy=args.dsa_strategy, dsa_param=dsa_param)
            clean_victims.append(net)
        clean_cta = float(np.mean([test_acc(n, test_imgs, test_labs, device)
                                   for n in clean_victims]))
        print('  clean baseline CTA = %.4f' % clean_cta)

    # ---- per class pair / target ------------------------------------------
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    all_rows = []
    for pair in args.class_pairs:
        y_adv, target_class = parse_pair(pair, class_names)
        print('\n%s ################ pair %s : y_adv=%d(%s)  target_class=%d(%s) ################'
              % (get_time(), pair, y_adv, class_names[y_adv],
                 target_class, class_names[target_class]))

        t_idx_all = (test_labs == target_class).nonzero(as_tuple=True)[0].cpu()
        if args.target_select == 'random':
            perm = torch.randperm(len(t_idx_all), generator=g)[:args.num_targets]
            chosen = t_idx_all[perm].tolist()
        else:  # 'first'
            chosen = t_idx_all[:args.num_targets].tolist()
        print('  targets (%s): %s' % (args.target_select, chosen))

        tally = np.zeros(num_classes, dtype=np.int64)
        pair_poison_asr, pair_clean_asr, pair_poison_cta = [], [], []

        for ti, tidx in enumerate(chosen):
            x_t_norm = test_imgs[tidx]

            if args.clean_baseline:
                cpreds = [predict_target(n, x_t_norm) for n in clean_victims]
                clean_asr = 100.0 * sum(p == y_adv for p in cpreds) / len(clean_victims)
            else:
                clean_asr = float('nan')

            # 1) selection on the S-trained surrogates (or random ablation)
            if args.random_select:
                base_idx = select_base_random(train_labs, y_adv, N_p, device)
            else:
                base_idx = select_base(surrogates, train_imgs, train_labs, x_t_norm,
                                       y_adv, N_p, args.lambda_margin, device)
            # 2) craft on the same surrogates
            base01 = denorm(train_imgs[base_idx]).clamp(0.0, 1.0).detach()
            x_adv01, coll = craft_pgd(surrogates, base01, x_t_norm, norm,
                                      args.epsilon, args.pgd_steps, args.pgd_alpha, device)
            linf = (x_adv01 - base01).abs().max().item()
            # 3) inject (clean-label) into a fresh clone of the full train set
            poisoned = train_imgs.clone()
            poisoned[base_idx] = norm(x_adv01)

            # 4) victims from scratch on the poisoned full set
            victim_preds, victim_ctas = [], []
            for vi in range(args.num_victims):
                net = get_network(args.model, channel, num_classes, im_size)
                net = train_from_scratch(net, poisoned, train_labs, args.victim_epochs,
                                         args.victim_lr, args.victim_bs, args.victim_decay,
                                         device, weight_decay=0.0, aug=args.victim_aug,
                                         dsa_strategy=args.dsa_strategy, dsa_param=dsa_param)
                pred = predict_target(net, x_t_norm)
                cta = test_acc(net, test_imgs, test_labs, device)
                victim_preds.append(pred)
                victim_ctas.append(cta)
                tally[pred] += 1
                del net
                if device == 'cuda':
                    torch.cuda.empty_cache()

            poison_asr = 100.0 * sum(p == y_adv for p in victim_preds) / args.num_victims
            poison_cta = float(np.mean(victim_ctas))
            pair_poison_asr.append(poison_asr)
            pair_clean_asr.append(clean_asr)
            pair_poison_cta.append(poison_cta)

            print('  [%s t%d/%d idx=%d] coll_mse=%.4f linf=%.4f | clean_ASR=%s '
                  'poison_CTA=%.4f poison_ASR=%.0f%%'
                  % (pair, ti + 1, len(chosen), tidx, coll, linf,
                     ('%.0f%%' % clean_asr) if args.clean_baseline else 'n/a',
                     poison_cta, poison_asr))

            all_rows.append({
                'pair': pair, 'y_adv': y_adv, 'target_class': target_class,
                'target_idx': tidx, 'clean_asr': clean_asr,
                'poison_cta': poison_cta, 'poison_asr': poison_asr,
                'collision_mse': coll, 'realized_linf': linf, 'N_p': N_p,
            })

        pa = np.array(pair_poison_asr)
        ct = np.array(pair_poison_cta)
        print('\n  ---- pair %s summary over %d targets x %d victims = %d votes ----'
              % (pair, len(chosen), args.num_victims, len(chosen) * args.num_victims))
        if args.clean_baseline:
            print('    clean baseline CTA = %.4f   mean clean ASR = %.1f%%'
                  % (clean_cta, float(np.nanmean(pair_clean_asr))))
        print('    poison CTA = %.4f +/- %.4f' % (ct.mean(), ct.std()))
        print('    poison ASR = %.1f%% +/- %.1f%%' % (pa.mean(), pa.std()))
        print('    target-prediction tally (%s): %s' % (class_names, tally.tolist()))

    # ---- persist ----------------------------------------------------------
    tag = 'standard_nodistill_%s_b%d_eps%d' % (
        args.model, round(args.budget * 1e4), round(args.epsilon * 255))
    with open(os.path.join(args.out_dir, 'results_%s.json' % tag), 'w') as f:
        json.dump({'clean_cta': clean_cta, 'rows': all_rows, 'args': vars(args)},
                  f, indent=2)
    if all_rows:
        with open(os.path.join(args.out_dir, 'results_%s.csv' % tag), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print('\n%s wrote results_%s.{json,csv} to %s' % (get_time(), tag, args.out_dir))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Standard from-scratch (no victim distillation) eval of the FC '
                    'poison; surrogates trained on the distilled S.')
    # data / model
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='data')
    p.add_argument('--model', type=str, default='ConvNetBN',
                   help="VICTIM arch (ConvNetBN to match MetaPoison; this repo's "
                        "ConvNetBN is depth-3, not the 6-layer Finn net)")
    p.add_argument('--out_dir', type=str, default='result/standard_nodistill')
    p.add_argument('--seed', type=int, default=0)
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
    p.add_argument('--class_pairs', nargs='+', default=['dog-bird', 'frog-airplane'],
                   help="MetaPoison naming 'poison-target', e.g. dog-bird frog-airplane")
    p.add_argument('--budget', type=float, default=0.01,
                   help="fraction of the FULL training set; 1%% = 500 poisons in y_adv")
    p.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    p.add_argument('--pgd_steps', type=int, default=250)
    p.add_argument('--pgd_alpha', type=float, default=1.0 / 255.0)
    p.add_argument('--lambda_margin', type=float, default=1.0)
    # protocol (MetaPoison victim side)
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--num_victims', type=int, default=6)
    p.add_argument('--target_select', type=str, default='random',
                   choices=['random', 'first'])
    p.add_argument('--victim_epochs', type=int, default=200)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=125)
    p.add_argument('--victim_decay', nargs='+', type=int, default=[100, 150])
    p.add_argument('--victim_aug', action='store_true', default=False,
                   help="MetaPoison default is NO augmentation; leave off to match")
    p.add_argument('--clean_baseline', action='store_true', default=False)
    p.add_argument('--random_select', action='store_true', default=False,
                   help='ablation: replace scored base selection with uniform random')
    main(p.parse_args())
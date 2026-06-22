"""
eval_selection.py

Selection ablation: hold the crafting objective, surrogate ensemble, victim
protocol, targets, and budget FIXED, and vary ONLY the rule used to pick the N_p
base images (--select_criterion, see selection_strategies.CRITERIA). This is the
in-place modification setting (bases are real images, overwritten by their
poisoned versions), matching eval_standard_nodistill.

To make the comparison valid the surrogate ensemble must be identical across
criteria, otherwise cross-criterion gaps are confounded by which ensemble was
drawn. Pass --surrogate_cache PATH: the first run trains the ensemble and saves
it; every later run loads the same weights. (Also a large compute saving: the
ensemble is trained once, not once per criterion.)

Example (run the whole menu against one cached ensemble): see selection_ablation.sh

  python eval_selection.py \
      --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
      --surrogate_cache result/sel_abl/surrogates_K10.pt \
      --surrogate_model ConvNet --model ConvNetBN --class_pairs dog-bird \
      --attack fc --budget 0.01 --epsilon 0.0313725 \
      --pgd_steps 250 --pgd_alpha 0.0039216 \
      --num_surrogates 10 --surrogate_epochs 1000 \
      --num_targets 10 --num_victims 6 \
      --victim_epochs 60 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 \
      --select_criterion feat_l2 --lambda_margin 1.0 \
      --target_select random --seed 0 --out_dir result/sel_abl/feat_l2
"""

import argparse
import csv
import json
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn as nn

from utils import get_dataset, get_network, ParamDiffAug, get_time
from eval_standard_nodistill import (
    embed_of, stack_dataset, train_surrogates_on_syn,
    craft_fc, craft_gradmatch, train_from_scratch, test_acc, predict_target,
    parse_pair,
)
from selection_strategies import select_bases, CRITERIA, TARGET_AWARE


def _unwrap(net):
    return net.module if isinstance(net, nn.DataParallel) else net


def get_surrogates(args, image_syn, label_syn, test_imgs, test_labs,
                   channel, num_classes, im_size, dsa_param, device):
    """Train (and cache) or load the surrogate ensemble. Cached weights make the
    ensemble identical across selection criteria."""
    req = (args.attack == 'gradmatch')
    if args.surrogate_cache and os.path.exists(args.surrogate_cache):
        print('%s loading cached surrogates from %s' % (get_time(), args.surrogate_cache))
        blob = torch.load(args.surrogate_cache, map_location=device, weights_only=False)
        nets = []
        for sd in blob['state_dicts']:
            net = get_network(args.surrogate_model, channel, num_classes, im_size)
            _unwrap(net).load_state_dict(sd)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(req)
            nets.append(net)
        return nets

    nets = train_surrogates_on_syn(image_syn, label_syn, test_imgs, test_labs,
                                   channel, num_classes, im_size, args, dsa_param, device)
    if args.surrogate_cache:
        os.makedirs(os.path.dirname(args.surrogate_cache) or '.', exist_ok=True)
        torch.save({'state_dicts': [_unwrap(n).state_dict() for n in nets]},
                   args.surrogate_cache)
        print('%s cached surrogates to %s' % (get_time(), args.surrogate_cache))
    return nets


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('%s device=%s  criterion=%s  (target-%s)'
          % (get_time(), device, args.select_criterion,
             'aware' if args.select_criterion in TARGET_AWARE else 'agnostic/control'))
    print('%s hyperparams: %s' % (get_time(), vars(args)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dsa_param = ParamDiffAug()

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
    print('%s N_total=%d  budget=%.4f -> N_p=%d bases (class y_adv)'
          % (get_time(), N_total, args.budget, N_p))

    ckpt = torch.load(args.syn_data_path, map_location='cpu', weights_only=False)
    image_syn, label_syn = ckpt['data'][-1]
    image_syn = image_syn.to(device)
    label_syn = label_syn.to(device)

    surrogates = get_surrogates(args, image_syn, label_syn, test_imgs, test_labs,
                                channel, num_classes, im_size, dsa_param, device)

    sel_gen = torch.Generator(device=device).manual_seed(args.seed)  # for random criterion
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    all_rows = []
    for pair in args.class_pairs:
        y_adv, target_class = parse_pair(pair, class_names)
        print('\n%s ####### pair %s : y_adv=%d(%s) target_class=%d(%s)  criterion=%s #######'
              % (get_time(), pair, y_adv, class_names[y_adv],
                 target_class, class_names[target_class], args.select_criterion))

        t_idx_all = (test_labs == target_class).nonzero(as_tuple=True)[0].cpu()
        if args.target_select == 'random':
            perm = torch.randperm(len(t_idx_all), generator=g)[:args.num_targets]
            chosen = t_idx_all[perm].tolist()
        else:
            chosen = t_idx_all[:args.num_targets].tolist()
        print('  targets (%s): %s' % (args.target_select, chosen))

        tally = np.zeros(num_classes, dtype=np.int64)
        pair_poison_asr, pair_poison_cta = [], []

        for ti, tidx in enumerate(chosen):
            x_t_norm = test_imgs[tidx]

            # 1) SELECT bases by the chosen criterion (only thing that varies)
            base_idx = select_bases(args.select_criterion, surrogates, train_imgs,
                                    train_labs, x_t_norm, y_adv, N_p,
                                    args.lambda_margin, device,
                                    denorm=denorm, generator=sel_gen,
                                    multilayer=args.multilayer)

            # 2) craft on the fixed ensemble
            base01 = denorm(train_imgs[base_idx]).clamp(0.0, 1.0).detach()
            if args.attack == 'gradmatch':
                x_adv01, obj = craft_gradmatch(
                    surrogates, base01, x_t_norm, y_adv, norm, args.epsilon,
                    args.pgd_alpha, args.pgd_steps, args.restarts, device,
                    dsa_strategy=args.dsa_strategy, dsa_param=dsa_param,
                    single_surrogate=args.single_surrogate)
            else:
                x_adv01, obj = craft_fc(
                    surrogates, base01, x_t_norm, norm, args.epsilon, args.pgd_steps,
                    args.pgd_alpha, device, single_surrogate=args.single_surrogate)
            linf = (x_adv01 - base01).abs().max().item()

            # 3) inject in place (clean-label) into a fresh clone of the full set
            poisoned = train_imgs.clone()
            poisoned[base_idx] = norm(x_adv01)

            # 4) victims from scratch
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
            pair_poison_cta.append(poison_cta)

            print('  [%s t%d/%d idx=%d] %s/%s craft_obj=%.4f linf=%.4f | '
                  'poison_CTA=%.4f poison_ASR=%.0f%%'
                  % (pair, ti + 1, len(chosen), tidx, args.select_criterion,
                     args.attack, obj, linf, poison_cta, poison_asr))

            all_rows.append({
                'pair': pair, 'criterion': args.select_criterion, 'attack': args.attack,
                'y_adv': y_adv, 'target_class': target_class, 'target_idx': tidx,
                'craft_obj': obj, 'realized_linf': linf,
                'poison_cta': poison_cta, 'poison_asr': poison_asr, 'N_p': N_p,
            })

        pa = np.array(pair_poison_asr)
        ct = np.array(pair_poison_cta)
        print('\n  ---- pair %s (%s, %s) over %d targets x %d victims = %d votes ----'
              % (pair, args.select_criterion, args.attack, len(chosen),
                 args.num_victims, len(chosen) * args.num_victims))
        print('    poison CTA = %.4f +/- %.4f' % (ct.mean(), ct.std()))
        print('    poison ASR = %.1f%% +/- %.1f%%' % (pa.mean(), pa.std()))
        print('    target-prediction tally (%s): %s' % (class_names, tally.tolist()))

    tag = 'sel_%s_%s_%s_b%d_eps%d' % (
        args.select_criterion, args.attack, args.model,
        round(args.budget * 1e4), round(args.epsilon * 255))
    with open(os.path.join(args.out_dir, 'results_%s.json' % tag), 'w') as f:
        json.dump({'criterion': args.select_criterion, 'rows': all_rows,
                   'args': vars(args)}, f, indent=2)
    if all_rows:
        with open(os.path.join(args.out_dir, 'results_%s.csv' % tag), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print('\n%s wrote results_%s.{json,csv} to %s' % (get_time(), tag, args.out_dir))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Selection ablation: vary only --select_criterion, '
                    'fixed (cached) surrogates, FC/gradmatch crafting, MetaPoison victims.')
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='data')
    p.add_argument('--model', type=str, default='ConvNetBN')
    p.add_argument('--out_dir', type=str, default='result/sel_abl')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dsa_strategy', type=str,
                   default='color_crop_cutout_flip_scale_rotate')
    # distilled S + surrogates
    p.add_argument('--syn_data_path', type=str,
                   default='result/res_DM_CIFAR10_ConvNet_50ipc.pt')
    p.add_argument('--surrogate_model', type=str, default='ConvNet')
    p.add_argument('--num_surrogates', type=int, default=10)
    p.add_argument('--surrogate_epochs', type=int, default=1000)
    p.add_argument('--surrogate_lr', type=float, default=0.01)
    p.add_argument('--surrogate_bs', type=int, default=256)
    p.add_argument('--surrogate_cache', type=str, default=None,
                   help='path to cache/reuse the SAME surrogate ensemble across '
                        'criteria (strongly recommended for a valid comparison)')
    # selection
    p.add_argument('--select_criterion', type=str, default='ours', choices=CRITERIA,
                   help='base-selection rule to ablate; see selection_strategies.CRITERIA')
    p.add_argument('--lambda_margin', type=float, default=1.0,
                   help="only used by 'ours'/'anti'")
    p.add_argument('--multilayer', action='store_true', default=False,
                   help='use all intermediate layer features for distance (not just last layer); '
                        'applies to feat_l2/ours/anti; recommended for VGG/ResNet')
    # attack
    p.add_argument('--attack', type=str, default='fc', choices=['fc', 'gradmatch'])
    p.add_argument('--class_pairs', nargs='+', default=['dog-bird'])
    p.add_argument('--budget', type=float, default=0.01)
    p.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    p.add_argument('--pgd_steps', type=int, default=250)
    p.add_argument('--pgd_alpha', type=float, default=1.0 / 255.0)
    p.add_argument('--restarts', type=int, default=8)
    p.add_argument('--single_surrogate', action='store_true', default=False)
    # protocol
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--num_victims', type=int, default=6)
    p.add_argument('--target_select', type=str, default='random',
                   choices=['random', 'first'])
    p.add_argument('--victim_epochs', type=int, default=60)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=125)
    p.add_argument('--victim_decay', nargs='+', type=int, default=[40])
    p.add_argument('--victim_aug', action='store_true', default=False)
    main(p.parse_args())

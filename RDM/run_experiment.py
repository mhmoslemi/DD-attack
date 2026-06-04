"""
RDM-DC experiment driver.

For each random seed (0..len(seed_list)-1) the paper picks one target image x_t
with original label `orig` and adversary label `y_adv` (Fig. 2 for CIFAR10).
We then craft P = round(rate * N) poisons among the class-`y_adv` training
images, build the poisoned (normalised) training set, run dataset condensation
`condense_runs_per_seed` times, and for each condensed set train
`eval_models_per_condense` fresh ConvNets, recording test accuracy and the
binary ASR on x_t. Mean/std are reported over the whole grid.

Vary `experiment.attack` in {gradmatch, dmpoison, direct} and
`experiment.defense` in {none, rdmdc, truncated, trimmed, median} to reproduce
the paper's tables (see README).

Usage:
    python run_experiment.py --config configs/cifar10.yaml
    python run_experiment.py --config configs/cifar10.yaml \
        --override experiment.attack=gradmatch experiment.defense=rdmdc
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from attacks import (direct_attack, dm_poisoning_attack,
                     gradient_matching_attack, pretrain_models)
from condense import distribution_matching
from config import config_to_dict, load_config
from data import Normalizer, build_poisoned_normalized, get_raw_dataset
from evaluate import train_and_eval
from utils import ParamDiffAug, TensorDataset, get_time


# Fig. 2 targets for CIFAR10: seed -> (original_label, adversary_label).
CIFAR10_TARGETS = {0: (7, 5), 1: (0, 9), 2: (5, 9), 3: (4, 9), 4: (2, 8)}


def get_targets(dataset, seed, num_classes):
    if dataset == 'CIFAR10' and seed in CIFAR10_TARGETS:
        return CIFAR10_TARGETS[seed]
    orig = seed % num_classes
    y_adv = (orig + 1) % num_classes
    return orig, y_adv


def resolve_device(spec):
    if spec in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return spec


def compute_drop_count(cfg, num_classes):
    eps_pc = cfg.defense.eps_per_class
    rate = cfg.poison.rate
    if eps_pc == 'auto':
        # Poisons concentrate in ONE class, so the per-class poison fraction in
        # a class batch is rate * num_classes (balanced classes).
        eps_pc = rate * num_classes
    return int(np.floor(3.0 * float(eps_pc) * cfg.condensation.batch_real)), eps_pc


def craft_poisons(attack, base_pixel, x_t_pixel, y_adv, pre_nets, normalizer,
                  cfg, meta, device):
    if attack == 'gradmatch':
        return gradient_matching_attack(base_pixel, x_t_pixel, y_adv, pre_nets,
                                        normalizer, cfg, device)
    if attack == 'dmpoison':
        return dm_poisoning_attack(base_pixel, x_t_pixel, y_adv, normalizer,
                                   cfg, meta, device)
    if attack == 'direct':
        return direct_attack(base_pixel, x_t_pixel, cfg)
    raise ValueError('unknown attack: %s' % attack)


def main():
    parser = argparse.ArgumentParser(description='RDM-DC')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--override', nargs='*', default=None,
                        help="dotted overrides, e.g. experiment.attack=dmpoison")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    cfg.dsa_param = ParamDiffAug()
    device = resolve_device(cfg.experiment.device)
    cfg.experiment.device = device

    os.makedirs(cfg.experiment.save_path, exist_ok=True)

    attack = cfg.experiment.attack
    defense = cfg.experiment.defense
    print('%s RDM-DC | dataset=%s attack=%s defense=%s device=%s'
          % (get_time(), cfg.experiment.dataset, attack, defense, device))

    # ----- data (raw [0,1] pixel space) -------------------------------------
    meta, train_pixel, train_labels, test_pixel, test_labels = get_raw_dataset(
        cfg.experiment.dataset, cfg.experiment.data_path)
    train_pixel = train_pixel.to(device)
    train_labels = train_labels.to(device)
    test_pixel = test_pixel.to(device)
    test_labels = test_labels.to(device)
    N = meta.n_train

    normalizer = Normalizer(meta.mean, meta.std, device=device)
    indices_class = [(train_labels == c).nonzero(as_tuple=True)[0]
                     for c in range(meta.num_classes)]

    # normalised test loader for evaluation
    test_norm = normalizer(test_pixel)
    testloader = torch.utils.data.DataLoader(
        TensorDataset(test_norm, test_labels),
        batch_size=256, shuffle=False, num_workers=0)

    drop_count, eps_pc = compute_drop_count(cfg, meta.num_classes)
    P = int(round(cfg.poison.rate * N))
    print('%s N=%d  P(poisons)=%d  per-class eps=%.4f  drop_count=%d'
          % (get_time(), N, P, float(eps_pc), drop_count))

    # ----- pretrain clean ensemble once (gradient matching only) ------------
    pre_nets = None
    if attack == 'gradmatch':
        clean_norm = normalizer(train_pixel)
        pre_nets = pretrain_models(cfg, meta, clean_norm, train_labels, device)

    # ----- main grid --------------------------------------------------------
    test_accs, asrs = [], []
    saved = []
    for seed in cfg.experiment.seed_list:
        orig, y_adv = get_targets(cfg.experiment.dataset, seed, meta.num_classes)
        g = torch.Generator(device='cpu').manual_seed(seed)

        # target image x_t: a test image of the original class (deterministic)
        test_orig_idx = (test_labels == orig).nonzero(as_tuple=True)[0].cpu()
        x_t_index = test_orig_idx[torch.randint(len(test_orig_idx), (1,),
                                                generator=g).item()].item()
        x_t_pixel = test_pixel[x_t_index:x_t_index + 1]
        x_t_norm = normalizer(x_t_pixel)

        # poison-base images: P class-y_adv training images (deterministic)
        cls_idx = indices_class[y_adv].cpu()
        perm = torch.randperm(len(cls_idx), generator=g)[:P]
        poison_idx = cls_idx[perm].to(device)
        base_pixel = train_pixel[poison_idx]

        print('\n%s === seed %d: orig=%d -> y_adv=%d | x_t=test#%d | crafting %d poisons (%s) ==='
              % (get_time(), seed, orig, y_adv, x_t_index, P, attack))
        poisoned_pixel = craft_poisons(attack, base_pixel, x_t_pixel, y_adv,
                                       pre_nets, normalizer, cfg, meta, device)

        images_norm = build_poisoned_normalized(
            train_pixel, train_labels, normalizer, poison_idx, poisoned_pixel)

        for run in range(cfg.experiment.condense_runs_per_seed):
            print('%s --- seed %d condense run %d/%d ---'
                  % (get_time(), seed, run + 1,
                     cfg.experiment.condense_runs_per_seed))
            image_syn, label_syn = distribution_matching(
                images_norm, train_labels, indices_class, cfg, meta, device,
                defense_method=defense, drop_count=drop_count)

            saved.append({
                'seed': seed, 'run': run, 'orig': orig, 'y_adv': y_adv,
                'x_t_index': x_t_index,
                'image_syn': image_syn.cpu().numpy().astype('float32'),
                'label_syn': label_syn.cpu().numpy().astype('int64'),
            })

            for ev in range(cfg.experiment.eval_models_per_condense):
                acc, asr = train_and_eval(
                    image_syn.clone(), label_syn.clone(), x_t_norm, y_adv,
                    testloader, cfg, meta, device, it_eval=ev)
                test_accs.append(acc)
                asrs.append(asr)
                print('%s seed=%d run=%d eval=%d  test_acc=%.4f  ASR=%.1f'
                      % (get_time(), seed, run, ev, acc, asr))

    test_accs = np.array(test_accs, dtype=np.float64)
    asrs = np.array(asrs, dtype=np.float64)
    summary = {
        'dataset': cfg.experiment.dataset,
        'attack': attack, 'defense': defense,
        'eps': float(cfg.poison.eps), 'rate': float(cfg.poison.rate),
        'per_class_eps': float(eps_pc), 'drop_count': drop_count,
        'n_measurements': int(test_accs.size),
        'test_acc_mean_pct': float(test_accs.mean() * 100),
        'test_acc_std_pct': float(test_accs.std() * 100),
        'asr_mean_pct': float(asrs.mean()),
        'asr_std_pct': float(asrs.std()),
        'config': config_to_dict(cfg),
    }
    summary['config'].pop('dsa_param', None)

    print('\n%s ==================== RESULTS ====================' % get_time())
    print('attack=%s defense=%s  TestAcc = %.2f%% +/- %.2f%%   ASR = %.2f%% +/- %.2f%%'
          % (attack, defense, summary['test_acc_mean_pct'],
             summary['test_acc_std_pct'], summary['asr_mean_pct'],
             summary['asr_std_pct']))

    tag = '%s_%s_%s_eps%d' % (cfg.experiment.dataset, attack, defense,
                              round(float(cfg.poison.eps) * 255))
    json_path = os.path.join(cfg.experiment.save_path, 'results_%s.json' % tag)
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    npz_path = os.path.join(cfg.experiment.save_path, 'synthetic_%s.npz' % tag)
    np.savez_compressed(
        npz_path,
        **{'syn_%d_%d' % (d['seed'], d['run']): d['image_syn'] for d in saved},
        **{'lab_%d_%d' % (d['seed'], d['run']): d['label_syn'] for d in saved})
    print('%s saved %s and %s' % (get_time(), json_path, npz_path))


if __name__ == '__main__':
    main()

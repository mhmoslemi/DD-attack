"""
select_targets.py

Run ONCE before the sweep.  Trains a single clean victim on the full CIFAR-10
training set, then for each class pair selects the `num_targets` test images that:
  (a) the clean model already classifies correctly as target_class, and
  (b) are NOT in the hardest `conf_trim` fraction (sorted by softmax confidence).

Saves the chosen indices to a JSON file (default: result/selected_targets.json).
Pass that file to main_IF.py with --target_idx_file to use these fixed targets.

Example:
  CUDA_VISIBLE_DEVICES=0 python select_targets.py \
      --model ConvNetBN \
      --class_pairs dog-bird frog-airplane \
      --num_targets 10 \
      --conf_trim 0.2 \
      --seed 0
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn.functional as F

from utils import get_dataset, get_network, get_time


def stack_dataset(dst, device):
    imgs = torch.stack([dst[i][0] for i in range(len(dst))]).to(device)
    labs = torch.tensor([dst[i][1] for i in range(len(dst))],
                        dtype=torch.long, device=device)
    return imgs, labs


def parse_pair(pair, class_names):
    a, b = pair.split('-')
    return class_names.index(a), class_names.index(b)


def train_clean(net, images, labels, epochs, lr, bs, decay_at, device):
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss().to(device)
    decay_at = set(decay_at)
    cur_lr = lr
    N = images.shape[0]
    for ep in range(epochs):
        if ep in decay_at:
            cur_lr *= 0.1
            for g in opt.param_groups:
                g['lr'] = cur_lr
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            crit(net(images[idx]), labels[idx]).backward()
            opt.step()
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            print('%s  epoch %d/%d' % (get_time(), ep + 1, epochs))
    net.eval()
    return net


@torch.no_grad()
def get_probs(net, imgs, bs=512):
    parts = []
    for i in range(0, len(imgs), bs):
        parts.append(F.softmax(net(imgs[i:i + bs]), dim=1))
    return torch.cat(parts, dim=0)


def select_for_pair(net, test_imgs, test_labs, target_class, num_targets,
                    conf_trim, seed, device):
    t_idx_all = (test_labs == target_class).nonzero(as_tuple=True)[0]   # on device
    cands = test_imgs[t_idx_all]
    probs = get_probs(net, cands)                    # (N, C)
    pred  = probs.argmax(1)                          # predicted class
    conf  = probs[:, target_class]                   # confidence in target_class

    correct_mask = (pred == target_class)
    n_correct = correct_mask.sum().item()
    print('  class %d: %d/%d test images classified correctly'
          % (target_class, n_correct, len(t_idx_all)))

    if n_correct == 0:
        raise RuntimeError('no correctly classified images found for class %d' % target_class)

    correct_local = correct_mask.nonzero(as_tuple=True)[0]  # indices within cands

    # sort by confidence descending (most confident = easiest)
    order = conf[correct_local].argsort(descending=True)
    correct_local = correct_local[order]

    # drop the bottom conf_trim fraction
    if conf_trim > 0.0:
        keep = max(num_targets, int(len(correct_local) * (1.0 - conf_trim)))
        correct_local = correct_local[:keep]
        print('  after trimming bottom %.0f%%: %d candidates remain'
              % (conf_trim * 100, len(correct_local)))

    if len(correct_local) < num_targets:
        print('  WARNING: only %d candidates, need %d — using all'
              % (len(correct_local), num_targets))

    # random draw from the filtered pool (reproducible)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(correct_local), generator=g)[:num_targets]
    chosen_local = correct_local[perm]

    chosen_global = t_idx_all[chosen_local.cpu()].cpu().tolist()
    chosen_conf   = conf[chosen_local].cpu().tolist()
    return chosen_global, chosen_conf


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('%s device=%s' % (get_time(), device))

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(args.dataset, args.data_path)
    train_imgs, train_labs = stack_dataset(dst_train, device)
    test_imgs,  test_labs  = stack_dataset(dst_test,  device)

    print('\n%s === training clean %s for %d epochs ===' % (get_time(), args.model, args.epochs))
    net = get_network(args.model, channel, num_classes, im_size).to(device)
    net = train_clean(net, train_imgs, train_labs,
                      args.epochs, args.lr, args.bs, args.decay_at, device)

    with torch.no_grad():
        acc = (net(test_imgs).argmax(1) == test_labs).float().mean().item()
    print('%s clean test accuracy: %.2f%%' % (get_time(), acc * 100))

    results = {}
    for pair in args.class_pairs:
        y_adv, target_class = parse_pair(pair, class_names)
        print('\n%s === pair %s : y_adv=%s(%d)  target=%s(%d) ==='
              % (get_time(), pair,
                 class_names[y_adv], y_adv,
                 class_names[target_class], target_class))
        chosen, confs = select_for_pair(net, test_imgs, test_labs, target_class,
                                        args.num_targets, args.conf_trim,
                                        args.seed, device)
        print('  selected indices : %s' % chosen)
        print('  their confidence : %s' % ['%.3f' % c for c in confs])
        results[pair] = {'indices': chosen, 'confidence': confs,
                         'target_class': target_class, 'y_adv': y_adv}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'args': vars(args), 'class_names': class_names, 'pairs': results}, f, indent=2)
    print('\n%s saved → %s' % (get_time(), args.out))
    print('\nTo use in main_IF.py add:  --target_idx_file %s' % args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',     type=str, default='CIFAR10')
    p.add_argument('--data_path',   type=str, default='data')
    p.add_argument('--model',       type=str, default='ConvNetBN',
                   help='architecture to train for filtering (should match victim)')
    p.add_argument('--class_pairs', nargs='+', default=['dog-bird', 'frog-airplane'])
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--conf_trim',   type=float, default=0.2,
                   help='fraction of lowest-confidence (hardest) correct targets to drop')
    p.add_argument('--epochs',      type=int, default=80)
    p.add_argument('--lr',          type=float, default=0.1)
    p.add_argument('--bs',          type=int, default=125)
    p.add_argument('--decay_at',    nargs='+', type=int, default=[40, 60])
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--out',         type=str, default='result/selected_targets.json')
    main(p.parse_args())

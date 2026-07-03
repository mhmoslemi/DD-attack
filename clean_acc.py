"""
clean_acc.py

Clean-data baseline: train victim models from scratch on the FULL clean
CIFAR-10 train set (no poisons, no distillation) with the exact same victim
recipe as main_IF.py (train_from_scratch: SGD momentum 0.9, no aug, no weight
decay, x0.1 lr decay at --victim_decay), then report per-run test accuracy
and the mean / variance / std over --runs runs.

Example:
  python clean_acc.py --model VGG13BN --runs 5 \
      --victim_epochs 40 --victim_lr 0.1 --victim_bs 256 --victim_decay 35
"""

import argparse

import numpy as np
import torch

from utils import get_dataset, get_network, get_time
from main_IF import train_from_scratch, test_acc, stack_dataset


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('%s device=%s  model=%s  runs=%d' % (get_time(), device, args.model, args.runs))
    print('%s hyperparams: %s' % (get_time(), vars(args)))

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(args.dataset, args.data_path)
    train_imgs, train_labs = stack_dataset(dst_train, device)
    test_imgs, test_labs = stack_dataset(dst_test, device)

    accs = []
    for run in range(args.runs):
        seed = args.seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = get_network(args.model, channel, num_classes, im_size).to(device)
        net = train_from_scratch(net, train_imgs, train_labs,
                                 args.victim_epochs, args.victim_lr,
                                 args.victim_bs, args.victim_decay, device)
        acc = test_acc(net, test_imgs, test_labs, device)
        accs.append(acc)
        print('%s [%s] run %d/%d (seed %d): ACC = %.4f'
              % (get_time(), args.model, run + 1, args.runs, seed, acc))

    accs = np.array(accs)
    print('%s [%s] ACCs: %s' % (get_time(), args.model,
                                ' '.join('%.4f' % a for a in accs)))
    print('%s [%s] ACC mean = %.4f  var = %.6f  std = %.4f'
          % (get_time(), args.model, accs.mean(), accs.var(), accs.std()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data')
    p.add_argument('--model', type=str, default='ConvNetBN')
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--victim_epochs', type=int, default=40)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=256)
    p.add_argument('--victim_decay', nargs='+', type=int, default=[35])
    main(p.parse_args())

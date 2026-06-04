"""
select_targets.py

Pick non-random attack targets that are CONSISTENT across every distillation
setting for a dataset.

For one dataset we have (methods x ipcs) distilled sets, e.g.
DC/DM x {10, 50, 100} = 6 combinations, each saved by the DM/DC code as
    res_<method>_<dataset>_<model>_<ipc>ipc.pt
with format torch.save({'data': data_save, 'accs_all_exps': ...}, ...) and
data_save[exp_index] = [image_syn, label_syn].

This script, for the given dataset:
  1. for every combination, trains `num_nets` ConvNets from scratch on its
     distilled set (each net with a fixed seed -> reproducible);
  2. evaluates all nets on the test set and, per combination, marks a test point
     "reliably correct" if at least `per_combo_agreement` of that combo's nets
     classify it correctly (default: all of them);
  3. keeps a test point as a candidate only if it is reliably correct in at
     least `min_combos` combinations (default: all of them) -> these are the
     points that ALL methods and ipcs agree on;
  4. for each target (orig, adv) from targets.py, selects the top
     `targets_per_class` candidates of the source class, ranked by mean
     true-class confidence across all nets (deterministic);
  5. writes the result to a JSON file in APPEND manner, keyed by DATASET, so
     running it for CIFAR10, then SVHN, ... accumulates one entry per dataset.

The saved ids are indices into the torchvision test set in default order
(stable across runs) and are shared by all method/ipc combinations, so the
attack uses the same x_t targets regardless of which distilled set it attacks.

Place this file in the DD-attack repo root so it imports that repo's `utils`
(the pipeline that produced the .pt files). targets.py must be alongside it.

Examples
--------
    python select_targets.py --dataset CIFAR10
    python select_targets.py --dataset SVHN --methods DC DM --ipcs 10 50 100
    # relax if a class has fewer than 10 unanimous points:
    python select_targets.py --dataset STL10 --per_combo_agreement 4 --min_combos 5
"""
import argparse
import itertools
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

# From the DD-attack / DatasetCondensation repo this script lives in.
from utils import (get_dataset, get_network, evaluate_synset, TensorDataset,
                   ParamDiffAug, get_time, get_daparam)
from targets import TARGETS, make_targets, TARGET_SEED


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reinit_net_(net, seed):
    """Deterministically re-initialise all parameters.

    The repo's get_network seeds itself from wall-clock time, so we overwrite
    that: seed the RNG, then reset every layer. Gives distinct but reproducible
    nets across runs.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for m in net.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    return net


def build_args(device, epoch_eval_train, lr_net, batch_train, dsa,
               dsa_strategy):
    args = SimpleNamespace()
    args.device = device
    args.lr_net = lr_net
    args.epoch_eval_train = epoch_eval_train
    args.batch_train = batch_train
    args.dsa = dsa
    args.dsa_strategy = dsa_strategy
    args.dsa_param = ParamDiffAug()
    return args


@torch.no_grad()
def predict_test(net, testloader, device):
    """Return (pred [N], true_class_prob [N], labels [N]) over the test set."""
    net.eval()
    preds, true_probs, labels = [], [], []
    for x, y in testloader:
        x = x.to(device)
        out = net(x)
        prob = F.softmax(out, dim=1).cpu()
        preds.append(out.argmax(dim=1).cpu())
        true_probs.append(prob.gather(1, y.view(-1, 1)).squeeze(1))
        labels.append(y)
    return torch.cat(preds), torch.cat(true_probs), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser(description='Select consistent attack targets.')
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--methods', nargs='+', default=['DC', 'DM'])
    ap.add_argument('--ipcs', nargs='+', type=int, default=[10, 50, 100])
    ap.add_argument('--model', type=str, default='ConvNet')
    ap.add_argument('--result_dir', type=str,
                    default='/home/mmoslem3/scratch/DD-attack/result')
    ap.add_argument('--data_path', type=str,
                    default='/home/mmoslem3/scratch/DD-attack/data')
    ap.add_argument('--exp_index', type=int, default=-1)
    ap.add_argument('--num_nets', type=int, default=5)
    ap.add_argument('--per_combo_agreement', type=int, default=-1,
                    help="#nets per combo that must be correct; -1 = all")
    ap.add_argument('--min_combos', type=int, default=-1,
                    help="#combos a point must be reliable in; -1 = all present")
    ap.add_argument('--num_targets', type=int, default=10)
    ap.add_argument('--targets_per_class', type=int, default=10)
    ap.add_argument('--epoch_eval_train', type=int, default=300)
    ap.add_argument('--lr_net', type=float, default=0.01)
    ap.add_argument('--batch_train', type=int, default=256)
    ap.add_argument('--dsa', action='store_true', default=False)
    ap.add_argument('--no_dsa', dest='dsa', action='store_false')
    ap.add_argument('--dsa_strategy', type=str,
                    default='color_crop_cutout_flip_scale_rotate')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip_missing', action='store_true',
                    help="skip combos whose .pt file is absent instead of error")
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out_json', type=str, default=None)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    per_combo_agreement = args.num_nets if args.per_combo_agreement < 0 \
        else args.per_combo_agreement
    out_json = args.out_json or os.path.join(args.result_dir, 'target_ids.json')

    set_all_seeds(args.seed)

    # ----- dataset + ordered test loader (loaded once) ----------------------
    channel, im_size, num_classes, class_names, mean, std, dst_train, \
        dst_test, _ = get_dataset(args.dataset, args.data_path)
    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=256, shuffle=False, num_workers=0)
    n_test = len(dst_test)
    eval_args = build_args(device, args.epoch_eval_train, args.lr_net,
                           args.batch_train, args.dsa, args.dsa_strategy)

    combos = list(itertools.product(args.methods, args.ipcs))
    print('%s dataset=%s  combos=%s  num_nets=%d'
          % (get_time(), args.dataset,
             ['%s_%dipc' % (m, i) for m, i in combos], args.num_nets))

    # ----- train all combos, gather correctness + confidence ----------------
    combo_info = []                       # per present combo: name, accs
    reliable_masks = []                   # per present combo: bool [N]
    total_true_prob = np.zeros(n_test, dtype=np.float64)
    total_nets = 0
    true_labels = None

    for combo_idx, (method, ipc) in enumerate(combos):
        fname = 'res_%s_%s_%s_%dipc.pt' % (method, args.dataset, args.model, ipc)
        pt_path = os.path.join(args.result_dir, fname)
        cname = '%s_%dipc' % (method, ipc)
        if not os.path.isfile(pt_path):
            msg = 'missing distilled file: %s' % pt_path
            if args.skip_missing:
                print('%s SKIP %s (%s)' % (get_time(), cname, msg))
                continue
            raise FileNotFoundError(msg)
        
        print()
        blob = torch.load(pt_path, map_location='cpu', weights_only=False )
        image_syn, label_syn = blob['data'][args.exp_index]
        image_syn = image_syn.detach().float()
        label_syn = label_syn.detach().long()

        eval_args.dsa = (method == 'DSA')
        eval_args.dc_aug_param = None if eval_args.dsa else get_daparam(
            args.dataset, args.model, args.model, ipc)
        aug_active = eval_args.dsa or (
            eval_args.dc_aug_param is not None and
            eval_args.dc_aug_param['strategy'] != 'none')
        eval_args.epoch_eval_train = 1000 if aug_active else args.epoch_eval_train

        correct_combo = torch.zeros(args.num_nets, n_test, dtype=torch.bool)
        accs = []
        for i in range(args.num_nets):
            net_seed = 1000 * args.seed + 100 * combo_idx + i
            net = get_network(args.model, channel, num_classes, im_size).to(device)
            net = reinit_net_(net, net_seed).to(device)
            torch.manual_seed(net_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(net_seed)

            # print('%s [%s] training net %d/%d (seed=%d)'
                #   % (get_time(), cname, i + 1, args.num_nets, net_seed))
            net, _, acc_test = evaluate_synset(
                i, net, image_syn.clone(), label_syn.clone(), testloader,
                eval_args)
            accs.append(float(acc_test))

            pred, tprob, labels = predict_test(net, testloader, device)
            correct_combo[i] = pred.eq(labels)
            total_true_prob += tprob.numpy()
            total_nets += 1
            if true_labels is None:
                true_labels = labels.numpy()
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reliable = (correct_combo.sum(dim=0) >= per_combo_agreement).numpy()
        reliable_masks.append(reliable)
        combo_info.append({'name': cname, 'per_net_test_acc': accs,
                           'num_reliable': int(reliable.sum())})
        print('%s [%s] reliable (>=%d/%d nets): %d / %d'
              % (get_time(), cname, per_combo_agreement, args.num_nets,
                 int(reliable.sum()), n_test))

    if not reliable_masks:
        raise RuntimeError('no distilled files found for dataset %s' % args.dataset)

    n_present = len(reliable_masks)
    min_combos = n_present if args.min_combos < 0 else args.min_combos
    stack = np.stack(reliable_masks, axis=0)          # [n_present, N]
    reliable_combo_count = stack.sum(axis=0)          # [N]
    candidate_mask = reliable_combo_count >= min_combos
    mean_true_prob = total_true_prob / max(total_nets, 1)
    print('%s candidates reliable in >=%d/%d combos: %d / %d'
          % (get_time(), min_combos, n_present, int(candidate_mask.sum()),
             n_test))

    # ----- per-class candidate pool -----------------------------------------
    correct_ids_per_class = {}
    for c in range(num_classes):
        ids = np.where((true_labels == c) & candidate_mask)[0]
        correct_ids_per_class[str(c)] = [int(j) for j in ids]

    # ----- targets (fixed protocol, independent of --seed) ------------------
    if args.dataset in TARGETS and args.num_targets == len(TARGETS[args.dataset]):
        target_map = TARGETS[args.dataset]
    else:
        target_map = make_targets(num_classes, args.num_targets, seed=TARGET_SEED)

    targets_per_seed = {}
    for seed, (orig, adv) in target_map.items():
        cand = np.where((true_labels == orig) & candidate_mask)[0]
        order = cand[np.argsort(-mean_true_prob[cand], kind='stable')]
        chosen = order[:args.targets_per_class]
        if len(chosen) < args.targets_per_class:
            print('%s WARNING target %d (orig=%d): only %d consistent candidates'
                  % (get_time(), seed, orig, len(chosen)))
        targets_per_seed[str(seed)] = {
            'orig': int(orig), 'adv': int(adv),
            'target_ids': [int(j) for j in chosen],
            'target_true_prob': [float(mean_true_prob[j]) for j in chosen],
        }

    # ----- write JSON in append manner (keyed by dataset) -------------------
    entry = {
        'dataset': args.dataset, 'model': args.model,
        'combos': [c['name'] for c in combo_info],
        'num_nets': args.num_nets,
        'per_combo_agreement': per_combo_agreement,
        'min_combos': min_combos, 'exp_index': args.exp_index,
        'num_targets': args.num_targets,
        'targets_per_class': args.targets_per_class,
        'epoch_eval_train': args.epoch_eval_train, 'seed': args.seed,
        'target_seed': TARGET_SEED,
        'combo_test_acc': {c['name']: c['per_net_test_acc'] for c in combo_info},
        'num_candidates_total': int(candidate_mask.sum()),
        'targets_per_seed': targets_per_seed,
        'correct_ids_per_class': correct_ids_per_class,
    }

    db = {}
    if os.path.isfile(out_json):
        with open(out_json, 'r') as f:
            try:
                db = json.load(f)
            except json.JSONDecodeError:
                db = {}
    db[args.dataset] = entry
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(db, f, indent=2)

    print('%s wrote entry "%s" to %s' % (get_time(), args.dataset, out_json))
    for s, t in targets_per_seed.items():
        print('  target %s  orig=%d adv=%d  ids=%s'
              % (s, t['orig'], t['adv'], t['target_ids']))


if __name__ == '__main__':
    main()
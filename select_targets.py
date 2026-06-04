"""
select_targets.py

Pick non-random target points for the RDM-DC attack pipeline.

Idea: instead of choosing each target image x_t at random from the test set,
choose targets that a model trained on the *clean* distilled data classifies
correctly. Those are the meaningful targets to attack (flipping a point the
clean distilled model already gets right is a real success).

Given (dataset, method, ipc) this script:
  1. builds the distilled-set filename  res_<method>_<dataset>_<model>_<ipc>ipc.pt
     and loads it (the format saved by the DM/DC code:
     torch.save({'data': data_save, 'accs_all_exps': ...}, ...), where
     data_save[exp_index] = [image_syn, label_syn]);
  2. trains `num_nets` ConvNets from scratch on that synthetic set, each with a
     fixed per-net seed so the run is reproducible;
  3. evaluates all nets on the test set and marks every test point that is
     correctly classified by at least `agreement` of them (default: all of them);
  4. for each CIFAR10 Fig.2 seed (orig, adv), selects the top
     `targets_per_class` correctly-classified test points of the original class,
     ranked by mean true-class confidence across the nets (deterministic);
  5. writes everything to a JSON file in APPEND manner, keyed by the distilled
     filename, so running ipc 10 / 50 / 100 accumulates three entries in one
     file. Re-running the same config overwrites its own entry with identical
     content (the seeds make it deterministic).

The saved "ids" are indices into the torchvision test set in its default
(unshuffled) order, which is stable across runs, so they can be fed straight
back into the attack code as the chosen x_t indices.

Place this file in the DD-attack repo root so it imports that repo's `utils`
(get_dataset / get_network / evaluate_synset / TensorDataset / ParamDiffAug),
i.e. the exact pipeline the distilled .pt files were produced with.

Examples
--------
    python select_targets.py --dataset CIFAR10 --method DM --ipc 10
    python select_targets.py --dataset CIFAR10 --method DM --ipc 50
    python select_targets.py --dataset CIFAR10 --method DM --ipc 100
    python select_targets.py --dataset CIFAR10 --method DC --ipc 10 \
        --num_nets 5 --agreement 5 --targets_per_class 10 --epoch_eval_train 1000
"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

# These come from the DD-attack / DatasetCondensation repo this script lives in.
from utils import (get_dataset, get_network, evaluate_synset, TensorDataset,
                   ParamDiffAug, get_time)


# Fig. 2 targets for CIFAR10: seed -> (original_label, adversary_label).
CIFAR10_TARGETS = {0: (7, 5), 1: (0, 9), 2: (5, 9), 3: (4, 9), 4: (2, 8)}
TARGETS = {'CIFAR10': CIFAR10_TARGETS}


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort determinism. Exact bitwise determinism on GPU may also need
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment; the agreement filter
    # below makes the selected ids robust to small residual nondeterminism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reinit_net_(net, seed):
    """Re-initialise all parameters deterministically.

    The repo's get_network seeds itself from wall-clock time, so we overwrite
    that here: seed the global RNG, then reset every layer's parameters. This
    yields `num_nets` distinct but reproducible networks (seed -> seed+1 ...).
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
    """Return (pred [N], true_class_prob [N]) over the test set in order."""
    net.eval()
    preds, true_probs, labels = [], [], []
    for x, y in testloader:
        x = x.to(device)
        out = net(x)
        prob = F.softmax(out, dim=1).cpu()
        preds.append(out.argmax(dim=1).cpu())
        true_probs.append(prob.gather(1, y.view(-1, 1)).squeeze(1))
        labels.append(y)
    return (torch.cat(preds), torch.cat(true_probs), torch.cat(labels))


def main():
    ap = argparse.ArgumentParser(description='Select non-random attack targets.')
    ap.add_argument('--dataset', type=str, default='CIFAR10')
    ap.add_argument('--method', type=str, default='DM', choices=['DM', 'DC'])
    ap.add_argument('--ipc', type=int, required=True)
    ap.add_argument('--model', type=str, default='ConvNet')
    ap.add_argument('--result_dir', type=str,
                    default='/home/mmoslem3/scratch/DD-attack/result')
    ap.add_argument('--data_path', type=str,
                    default='/home/mmoslem3/scratch/DD-attack/data')
    ap.add_argument('--exp_index', type=int, default=0,
                    help="which distilled set in data_save to use")
    ap.add_argument('--num_nets', type=int, default=5)
    ap.add_argument('--agreement', type=int, default=-1,
                    help="#nets that must be correct; -1 means all of them")
    ap.add_argument('--targets_per_class', type=int, default=10)
    ap.add_argument('--epoch_eval_train', type=int, default=1000)
    ap.add_argument('--lr_net', type=float, default=0.01)
    ap.add_argument('--batch_train', type=int, default=256)
    ap.add_argument('--dsa', action='store_true', default=True)
    ap.add_argument('--no_dsa', dest='dsa', action='store_false')
    ap.add_argument('--dsa_strategy', type=str,
                    default='color_crop_cutout_flip_scale_rotate')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out_json', type=str, default=None,
                    help="defaults to <result_dir>/target_ids.json")
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    agreement = args.num_nets if args.agreement < 0 else args.agreement
    out_json = args.out_json or os.path.join(args.result_dir, 'target_ids.json')

    set_all_seeds(args.seed)

    # ----- locate and load the distilled set --------------------------------
    fname = 'res_%s_%s_%s_%dipc.pt' % (args.method, args.dataset, args.model,
                                       args.ipc)
    pt_path = os.path.join(args.result_dir, fname)
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(pt_path)
    print('%s loading %s' % (get_time(), pt_path))
    blob = torch.load(pt_path, map_location='cpu')
    data_save = blob['data']
    if args.exp_index >= len(data_save):
        raise IndexError('exp_index %d but data_save has %d entries'
                         % (args.exp_index, len(data_save)))
    image_syn, label_syn = data_save[args.exp_index]
    image_syn = image_syn.detach().float()
    label_syn = label_syn.detach().long()
    print('%s synthetic set: images %s labels %s'
          % (get_time(), tuple(image_syn.shape), tuple(label_syn.shape)))

    # ----- dataset + ordered test loader ------------------------------------
    channel, im_size, num_classes, class_names, mean, std, dst_train, \
        dst_test, _ = get_dataset(args.dataset, args.data_path)
    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=256, shuffle=False, num_workers=0)

    eval_args = build_args(device, args.epoch_eval_train, args.lr_net,
                           args.batch_train, args.dsa, args.dsa_strategy)

    # ----- train num_nets nets, collect predictions -------------------------
    n_test = len(dst_test)
    correct = torch.zeros(args.num_nets, n_test, dtype=torch.bool)
    true_prob = torch.zeros(args.num_nets, n_test, dtype=torch.float)
    true_labels = None
    per_net_acc = []

    for i in range(args.num_nets):
        net_seed = 1000 * args.seed + i
        net = get_network(args.model, channel, num_classes, im_size).to(device)
        net = reinit_net_(net, net_seed).to(device)
        torch.manual_seed(net_seed)  # make training (shuffle+aug) deterministic
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(net_seed)

        print('%s training net %d/%d (seed=%d) on synthetic set'
              % (get_time(), i + 1, args.num_nets, net_seed))
        net, _, acc_test = evaluate_synset(
            i, net, image_syn.clone(), label_syn.clone(), testloader, eval_args)
        per_net_acc.append(float(acc_test))

        pred, tprob, labels = predict_test(net, testloader, device)
        correct[i] = pred.eq(labels)
        true_prob[i] = tprob
        if true_labels is None:
            true_labels = labels
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    true_labels = true_labels.numpy()
    correct_count = correct.sum(dim=0)                  # [N]
    correct_mask = (correct_count >= agreement).numpy()  # [N] bool
    mean_true_prob = true_prob.mean(dim=0).numpy()       # [N]
    print('%s test points correct by >=%d/%d nets: %d / %d'
          % (get_time(), agreement, args.num_nets, int(correct_mask.sum()),
             n_test))

    # ----- per-class candidate pool -----------------------------------------
    correct_ids_per_class = {}
    for c in range(num_classes):
        ids = np.where((true_labels == c) & correct_mask)[0]
        correct_ids_per_class[str(c)] = [int(j) for j in ids]

    # ----- target selection per Fig.2 seed (CIFAR10) ------------------------
    targets_per_seed = {}
    target_map = TARGETS.get(args.dataset)
    if target_map is None:
        print('%s no Fig.2 target map for %s; saving candidate pool only'
              % (get_time(), args.dataset))
    else:
        for seed, (orig, adv) in target_map.items():
            cand = np.where((true_labels == orig) & correct_mask)[0]
            # rank by mean true-class confidence, descending (deterministic).
            order = cand[np.argsort(-mean_true_prob[cand], kind='stable')]
            chosen = order[:args.targets_per_class]
            if len(chosen) < args.targets_per_class:
                print('%s WARNING seed %d (orig=%d): only %d correct candidates'
                      % (get_time(), seed, orig, len(chosen)))
            targets_per_seed[str(seed)] = {
                'orig': int(orig), 'adv': int(adv),
                'target_ids': [int(j) for j in chosen],
                'target_true_prob': [float(mean_true_prob[j]) for j in chosen],
            }

    # ----- write JSON in append manner --------------------------------------
    entry = {
        'pt_file': pt_path,
        'dataset': args.dataset, 'method': args.method, 'model': args.model,
        'ipc': args.ipc, 'exp_index': args.exp_index,
        'num_nets': args.num_nets, 'agreement': agreement,
        'targets_per_class': args.targets_per_class,
        'epoch_eval_train': args.epoch_eval_train, 'seed': args.seed,
        'per_net_test_acc': per_net_acc,
        'num_correct_total': int(correct_mask.sum()),
        'targets_per_seed': targets_per_seed,
        'correct_ids_per_class': correct_ids_per_class,
    }
    key = os.path.splitext(fname)[0]  # e.g. res_DM_CIFAR10_ConvNet_10ipc

    db = {}
    if os.path.isfile(out_json):
        with open(out_json, 'r') as f:
            try:
                db = json.load(f)
            except json.JSONDecodeError:
                db = {}
    db[key] = entry
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(db, f, indent=2)

    print('%s wrote entry "%s" to %s' % (get_time(), key, out_json))
    if targets_per_seed:
        for s, t in targets_per_seed.items():
            print('  seed %s  orig=%d adv=%d  ids=%s'
                  % (s, t['orig'], t['adv'], t['target_ids']))


if __name__ == '__main__':
    main()
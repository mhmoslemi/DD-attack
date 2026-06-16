"""
Multi-target driver for the targeted clean-label feature-collision poisoning of DM condensation.

Wraps poison_condensation_pipeline.py and adds:
  * A SCREENING phase: train a pool of clean models on S, keep only test targets whose true label
    == --target_class (default 'dog') AND that the clean models classify CORRECTLY (>= --screen_agree
    of them) AND that none of them already map to y_adv. This fixes the source class and removes
    contaminated targets like the original idx=42 (clean S mapped it to class 3 in 2/5 victims).
  * A loop over --num_targets such targets, running selection (Eq.1) -> generator (Eq.2)
    -> inject (Eq.3) -> condense -> evaluate, per target.
  * Aggregation across targets (per-target table + means + CSV + summary .pt).

Three model pools (so the clean baseline is honest, not circular):
  A) clean models  : --num_clean_models nets on clean S. Used for SCREENING and as SURROGATES.
  B) clean victims : --num_victims nets on clean S, DIFFERENT seeds, trained with --victim_epochs.
                     Reused across all targets for the CLEAN baseline ASR. Held out from screening,
                     so the baseline is non-circular.
  C) poison victims: --num_victims fresh nets on each poisoned S', for the POISON ASR (per target).

Cost note: surrogates/clean-models/clean-victims train once. Per target you pay one generator run,
one condensation (--Iteration), and --num_victims victim trainings. 10 targets is roughly
10 x (condensation + num_victims * victim_epochs). Tune --Iteration / --num_victims / --victim_epochs
down for a faster sweep.

Place next to utils.py and poison_condensation_pipeline.py. Run step-1 clean distillation first.

USAGE:
  python poison_condensation_multi.py \
      --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
      --out_dir       result/attack_multi \
      --y_adv 3 --target_class dog --num_targets 10 --N_p 500 --epsilon 0.03137 \
      --num_clean_models 5 --num_surrogates 3 --surrogate_epochs 1000 \
      --gen_epochs 2000 --Iteration 5000 \
      --num_victims 5 --victim_epochs 1000
"""

import warnings
warnings.filterwarnings("ignore")

import os
import csv
import copy
import argparse
import numpy as np
import torch

from utils import get_loops, get_dataset, get_network, evaluate_synset, ParamDiffAug
from poison_condensation_pipeline import (
    PerturbationGenerator, feat_fn, select_base, train_generator,
    condense, train_surrogates, evaluate_attack, craft_pgd, collision_mse,
)


# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_all(nets, X, bs=512):
    """Return [K, N] predictions of K nets over N inputs X (already normalized, on device)."""
    P = []
    for net in nets:
        net.eval()
        preds = [net(X[i:i + bs]).argmax(1).cpu() for i in range(0, len(X), bs)]
        P.append(torch.cat(preds))
    return torch.stack(P)


@torch.no_grad()
def test_acc(net, testloader, device):
    net.eval()
    c = t = 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        c += (net(x).argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


def screen_targets(clean_nets, dst_test, target_class, y_adv, n_targets, agree, seed, device):
    """Pick test indices whose true label == target_class, classified correctly by >= `agree`
    clean nets, and mapped to y_adv by NONE of them. Returns list of (idx, true_label)."""
    X = torch.stack([dst_test[i][0] for i in range(len(dst_test))]).to(device)
    Y = torch.tensor([dst_test[i][1] for i in range(len(dst_test))])
    P = predict_all(clean_nets, X)                 # [K, N]
    correct = (P == Y.unsqueeze(0)).sum(0)         # [N]
    to_adv = (P == y_adv).sum(0)                   # [N]
    eligible = (Y == target_class) & (correct >= agree) & (to_adv == 0)
    idxs = eligible.nonzero(as_tuple=True)[0]
    print(f"  {len(idxs)} eligible targets "
          f"(true == {target_class}, correct by >= {agree}/{len(clean_nets)}, never -> {y_adv})")
    if len(idxs) < n_targets:
        print(f"  WARNING: fewer eligible targets than requested; using all {len(idxs)}.")
        n_targets = len(idxs)
    g = torch.Generator().manual_seed(seed)
    pick = idxs[torch.randperm(len(idxs), generator=g)[:n_targets]]
    return [(int(i), int(Y[i])) for i in pick.tolist()]


# ----------------------------------------------------------------------------
def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.method = 'DM'
    args.dsa_param = ParamDiffAug()
    args.dsa = args.dsa_strategy not in ['none', 'None']
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    if args.screen_agree is None:
        args.screen_agree = args.num_clean_models
    assert args.num_clean_models >= args.num_surrogates, "need num_clean_models >= num_surrogates"
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- dataset (mean/std from here; used everywhere) ---
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
        get_dataset(args.dataset, args.data_path)

    # resolve source class for targets: accept a name ('dog') or an index ('5')
    tc = str(args.target_class)
    if tc.lstrip('-').isdigit():
        args.target_class = int(tc)
    elif tc in class_names:
        args.target_class = class_names.index(tc)
    else:
        raise ValueError(f"--target_class '{tc}' is neither an index nor one of {class_names}")
    assert args.target_class != args.y_adv, "target_class must differ from y_adv (else trivial)"

    m = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    s = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    norm = lambda x01: (x01 - m) / s
    denorm = lambda xn: xn * s + m

    # --- clean (normalized) training pool, kept pristine; re-cloned per target ---
    images_all_clean = torch.cat(
        [torch.unsqueeze(dst_train[i][0], 0) for i in range(len(dst_train))], 0).to(args.device)
    labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))],
                              dtype=torch.long, device=args.device)
    indices_class = [[] for _ in range(num_classes)]
    for i, lab in enumerate(labels_all.tolist()):
        indices_class[lab].append(i)

    # --- clean distilled set S (must be normalized, from step 1) ---
    print(f"\nLoading clean distilled S from {args.syn_data_path}")
    ckpt = torch.load(args.syn_data_path, map_location='cpu', weights_only=False)
    image_syn_clean, label_syn_clean = ckpt['data'][-1]
    image_syn_clean = image_syn_clean.to(args.device)
    label_syn_clean = label_syn_clean.to(args.device)
    print(f"  S stats: min={image_syn_clean.min():.3f} max={image_syn_clean.max():.3f} "
          f"(expected ~[-2.5, 2.7] for normalized CIFAR)")

    # ============================ POOL A: clean models (screen + surrogates) ============
    print(f"\n=== Pool A: training {args.num_clean_models} clean models on S "
          f"(screening + surrogates) ===")
    args.epoch_eval_train = args.surrogate_epochs
    clean_models = train_surrogates(image_syn_clean, label_syn_clean, testloader,
                                    channel, num_classes, im_size, args, args.num_clean_models)
    surrogates = clean_models[:args.num_surrogates]

    # ============================ SCREENING =============================================
    print(f"\n=== Screening for {args.num_targets} targets from class "
          f"{args.target_class} ({class_names[args.target_class]}) -> y_adv={args.y_adv} "
          f"({class_names[args.y_adv]}) ===")
    targets = screen_targets(clean_models, dst_test, args.target_class, args.y_adv,
                             args.num_targets, args.screen_agree, args.screen_seed, args.device)
    print(f"  chosen target indices (all class {class_names[args.target_class]}):",
          [i for i, _ in targets])

    # ============================ POOL B: clean baseline victims ========================
    print(f"\n=== Pool B: training {args.num_victims} held-out clean victims on S "
          f"(honest clean baseline) ===")
    args.epoch_eval_train = args.victim_epochs
    args.seed = args.seed + 9999  # decorrelate from pool A / np seeds
    torch.manual_seed(args.seed)
    clean_victims = train_surrogates(image_syn_clean, label_syn_clean, testloader,
                                     channel, num_classes, im_size, args, args.num_victims)
    clean_cta = float(np.mean([test_acc(n, testloader, args.device) for n in clean_victims]))
    print(f"  clean baseline CTA = {clean_cta:.4f} (constant across targets)")

    # ============================ PER-TARGET PIPELINE ===================================
    results = []
    for ti, (tidx, ttrue) in enumerate(targets):
        print(f"\n########## target {ti+1}/{len(targets)}: idx={tidx} "
              f"true={ttrue}({class_names[ttrue]}) -> y_adv={args.y_adv}({class_names[args.y_adv]}) ##########")
        x_target_norm = dst_test[tidx][0].to(args.device)

        # clean baseline ASR for this target (held-out victims B)
        with torch.no_grad():
            cpreds = [n(x_target_norm.unsqueeze(0)).argmax(1).item() for n in clean_victims]
        clean_asr = 100.0 * sum(p == args.y_adv for p in cpreds) / len(clean_victims)

        # 1) selection (Eq.1) on the clean pool, ensemble-averaged
        base_idx = select_base(surrogates, images_all_clean, labels_all, x_target_norm,
                               args.y_adv, args.N_p, args.lambda_margin, args.device)

        # 2) craft perturbations (Eq.2): amortized generator OR per-sample PGD
        base01 = denorm(images_all_clean[base_idx]).clamp(0.0, 1.0).detach()
        mse0 = collision_mse(surrogates, base01, x_target_norm, norm)  # clean cat -> dog target
        if args.attack == 'pgd':
            x_adv01 = craft_pgd(surrogates, base01, x_target_norm, norm,
                                args.epsilon, args.pgd_steps, args.pgd_alpha, args.device)
        else:
            gen = PerturbationGenerator(channel=channel, epsilon=args.epsilon).to(args.device)
            gen = train_generator(gen, surrogates, base01, x_target_norm, norm,
                                  args.gen_epochs, args.gen_lr, args.device)
            with torch.no_grad():
                x_adv01 = gen(base01)
        with torch.no_grad():
            linf = (x_adv01 - base01).abs().max().item()
            mse1 = collision_mse(surrogates, x_adv01, x_target_norm, norm)
        red = 100.0 * (mse0 - mse1) / max(mse0, 1e-8)
        print(f"  collision MSE: clean={mse0:.4f} -> crafted={mse1:.4f} ({red:.1f}% reduction), "
              f"realized Linf={linf:.4f}  [if reduction is small the attack cannot work]")

        # 3) inject (Eq.3) into a FRESH clone of the clean pool (labels kept = y_adv)
        images_all = images_all_clean.clone()
        images_all[base_idx] = norm(x_adv01)

        # 4) condense poisoned pool -> S'
        args.epoch_eval_train = args.victim_epochs  # restore (train_generator/surrogates may have left it)
        img_syn_p, lab_syn_p = condense(images_all, labels_all, indices_class,
                                        channel, num_classes, im_size, args)

        # save S' for this target
        out_path = os.path.join(
            args.out_dir,
            f"res_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_attack_t{tidx}.pt")
        torch.save({
            'data': [[img_syn_p.cpu(), lab_syn_p.cpu()]],
            'poisoned_indices': base_idx.cpu(),
            'target_idx': tidx, 'target_true_label': ttrue, 'y_adv': args.y_adv,
            'epsilon': args.epsilon, 'N_p': args.N_p, 'realized_linf': linf,
        }, out_path)

        # 5) poison victims (pool C, fresh on S') -> ASR
        cta_mean, cta_std, poison_asr, poison_hits = evaluate_attack(
            img_syn_p, lab_syn_p, testloader, x_target_norm, args.y_adv,
            channel, num_classes, im_size, args, args.num_victims, f"POISON t{tidx}")

        results.append({
            'target_idx': tidx, 'true_label': ttrue, 'true_name': class_names[ttrue],
            'clean_asr': clean_asr, 'poison_cta': cta_mean, 'poison_cta_std': cta_std,
            'poison_asr': poison_asr, 'poison_hits': poison_hits, 'realized_linf': linf,
            'collision_clean': mse0, 'collision_crafted': mse1,
            'out_path': out_path,
        })
        print(f"  -> clean_ASR={clean_asr:.0f}%  poison_CTA={cta_mean:.4f}  "
              f"poison_ASR={poison_asr:.0f}% ({poison_hits}/{args.num_victims})")

    # ============================ AGGREGATE =============================================
    print("\n================================ SWEEP RESULTS ================================")
    print(f"  clean baseline CTA (pool B) = {clean_cta:.4f}")
    print(f"  {'idx':>6} {'true':>10} {'clean_ASR':>10} {'pois_CTA':>10} {'pois_ASR':>10} {'Linf':>8}")
    for r in results:
        print(f"  {r['target_idx']:>6} {r['true_name']:>10} {r['clean_asr']:>9.0f}% "
              f"{r['poison_cta']:>10.4f} {r['poison_asr']:>9.0f}% {r['realized_linf']:>8.4f}")

    pa = np.array([r['poison_asr'] for r in results])
    ca = np.array([r['clean_asr'] for r in results])
    ct = np.array([r['poison_cta'] for r in results])
    n_better = int(np.sum(pa > ca))
    print("  " + "-" * 76)
    print(f"  mean poison ASR = {pa.mean():.1f}% +/- {pa.std():.1f}%   "
          f"mean clean ASR = {ca.mean():.1f}%")
    print(f"  mean poison CTA = {ct.mean():.4f} +/- {ct.std():.4f} "
          f"(clean CTA {clean_cta:.4f}, drop {clean_cta - ct.mean():+.4f})")
    print(f"  attack increased ASR on {n_better}/{len(results)} targets")
    print("==============================================================================")

    # save summary
    torch.save({'results': results, 'clean_cta': clean_cta, 'args': vars(args)},
               os.path.join(args.out_dir, 'summary.pt'))
    with open(os.path.join(args.out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved per-target S' files + summary.pt + summary.csv in {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-target clean-label poisoning of DM condensation")
    # data / model
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--model', type=str, default='ConvNet')
    p.add_argument('--ipc', type=int, default=50)
    p.add_argument('--data_path', type=str, default='data')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init', type=str, default='real')
    p.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    p.add_argument('--dis_metric', type=str, default='ours')
    # input / output
    p.add_argument('--syn_data_path', type=str, required=True, help="clean distilled S .pt from step 1")
    p.add_argument('--out_dir', type=str, default='result/attack_multi')
    # screening
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--target_class', type=str, default='dog',
                   help="true class of all selected targets: a name like 'dog' or an index like '5'. "
                        "Must differ from --y_adv.")
    p.add_argument('--num_clean_models', type=int, default=5, help="pool A: screening + surrogates")
    p.add_argument('--screen_agree', type=int, default=None,
                   help="min #clean models that must classify target correctly (default = all)")
    p.add_argument('--screen_seed', type=int, default=0)
    # attack
    p.add_argument('--y_adv', type=int, default=3)
    p.add_argument('--N_p', type=int, default=500)
    p.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    p.add_argument('--lambda_margin', type=float, default=1.0)
    # surrogate / generator / pgd
    p.add_argument('--attack', type=str, default='pgd', choices=['pgd', 'generator'],
                   help="how to craft the perturbation: per-sample PGD (strong) or amortized G_phi")
    p.add_argument('--num_surrogates', type=int, default=3)
    p.add_argument('--surrogate_epochs', type=int, default=1000)
    p.add_argument('--gen_epochs', type=int, default=2000)
    p.add_argument('--gen_lr', type=float, default=1e-3)
    p.add_argument('--pgd_steps', type=int, default=500)
    p.add_argument('--pgd_alpha', type=float, default=1.0 / 255.0)
    # condensation
    p.add_argument('--Iteration', type=int, default=5000)
    p.add_argument('--lr_img', type=float, default=1.0)
    p.add_argument('--lr_net', type=float, default=0.01)
    p.add_argument('--batch_real', type=int, default=256)
    p.add_argument('--batch_train', type=int, default=256)
    # eval
    p.add_argument('--num_victims', type=int, default=5)
    p.add_argument('--victim_epochs', type=int, default=1000)

    main(p.parse_args())
"""
main_IF_exact.py

Selection ablation: SMART-select (ours, ICLR draft) vs EXACT influence-function
selection, in the feature-collision (fc) poisoning setting.

Both methods pick the m base points to poison; EVERYTHING else is identical and
reused verbatim from main_IF.py (same cached surrogates, same fc craft, same
from-scratch victim training). The ONLY difference is how the bases are scored,
so any gap in ASR/CTA -- and the selection wall-clock time -- is attributable to
the selection rule alone.

  smart : first-order gradient alignment that DROPS the inverse Hessian, i.e.
          H^{-1} ~ I  =>  score(z) = < grad_th l(z_t), grad_th l(z) >.
          In main_IF this is realized as feature-distance + margin (select_base).

  exact : the SAME influence derivation but KEEPING the curvature term,
          score(z) = grad_th l(z_t)^T H^{-1} grad_th l(z),
          with H the Hessian of the surrogate's empirical risk on the set it was
          trained on (the distilled S, or full data). H^{-1} g is solved with
          conjugate gradients over Hessian-vector products (no explicit Hessian);
          per-candidate scores < s, grad_th l(z) > are read off by a central
          finite-difference directional derivative (2 forward passes / surrogate).
          This is "exact influence the way the draft derives it", NOT the original
          Koh & Liang up-weighting normalization.

We DO NOT modify main_IF.py.

Example:
  python main_IF_exact.py \
      --syn_data_path result/res_DM_CIFAR10_ConvNet_100ipc.pt \
      --surrogate_model ConvNet --model ConvNetBN --class_pairs dog-bird \
      --budget 0.01 --num_surrogates 10 --surrogate_epochs 1000 --single_surrogate \
      --num_targets 10 --num_victims 6 --victim_epochs 60 --victim_decay 40 \
      --cache_dir result/cache --methods smart exact --seed 0
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import main_IF as M                       # reuse surrogates / craft / victim / smart-select
from utils import (get_dataset, get_network, ParamDiffAug, get_time)


# --------------------------------------------------------------------------- #
# EXACT influence-function selection
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
    Hessian of the mean CE on (hess_imgs, hess_labs) w.r.t. `params`. Returns s
    (= approx H^{-1} g). HVPs use one retained double-backward graph."""
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
    total /= len(nets)
    if verbose:
        print('      [exact] CG iters used per surrogate: %s' % cg_used)
    sel = torch.topk(total, k=N_p, largest=True).indices
    return cls_idx[sel]


# --------------------------------------------------------------------------- #
# surrogate setup (replicated from main_IF.main so we reuse the SAME cache)
# --------------------------------------------------------------------------- #
def build_surrogates(args, image_syn, label_syn, train_imgs, train_labs,
                     test_imgs, test_labs, channel, num_classes, im_size,
                     dsa_param, device):
    _sur_tag = 'fulldata' if args.surrogate_on_full_data else 'syn'
    sur_cache = os.path.join(args.cache_dir,
        'surrogates_%s_%s_%dx%dep_seed%d' % (
            args.surrogate_model, _sur_tag,
            args.num_surrogates, args.surrogate_epochs, args.seed)
    ) if args.cache_dir else ''

    if sur_cache and all(os.path.exists(os.path.join(sur_cache, 'surrogate_%d.pt' % i))
                         for i in range(args.num_surrogates)):
        print('\n%s === loading %d surrogates from cache: %s ==='
              % (get_time(), args.num_surrogates, sur_cache))
        surrogates = []
        for i in range(args.num_surrogates):
            net = get_network(args.surrogate_model, channel, num_classes, im_size)
            net.load_state_dict(torch.load(
                os.path.join(sur_cache, 'surrogate_%d.pt' % i), map_location=device))
            surrogates.append(net.to(device).eval())
    else:
        if args.surrogate_on_full_data:
            print('\n%s === training %d surrogates (%s) on FULL real data ==='
                  % (get_time(), args.num_surrogates, args.surrogate_model))
            surrogates = M.train_surrogates_on_full(train_imgs, train_labs,
                                                    test_imgs, test_labs,
                                                    channel, num_classes, im_size, args, device)
        else:
            print('\n%s === training %d surrogates (%s) on distilled S ==='
                  % (get_time(), args.num_surrogates, args.surrogate_model))
            surrogates = M.train_surrogates_on_syn(image_syn, label_syn, test_imgs, test_labs,
                                                   channel, num_classes, im_size, args,
                                                   dsa_param, device)
        if sur_cache:
            os.makedirs(sur_cache, exist_ok=True)
            for i, net in enumerate(surrogates):
                torch.save(net.state_dict(), os.path.join(sur_cache, 'surrogate_%d.pt' % i))
            print('%s  saved surrogates to %s' % (get_time(), sur_cache))

    # EXACT influence needs gradients w.r.t. surrogate params (fc froze them);
    # re-enable. smart-select / craft are unaffected by params requiring grad.
    for net in surrogates:
        for p in net.parameters():
            p.requires_grad_(True)
    return surrogates


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or '.', exist_ok=True)
        tee = M._Tee(args.log_file)
        sys.stdout = tee
        sys.stderr = tee
        print('%s logging (line-buffered, no delay) -> %s' % (get_time(), args.log_file))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('%s device=%s' % (get_time(), device))
    print('%s hyperparams: %s' % (get_time(), vars(args)))
    print('%s methods: %s' % (get_time(), args.methods))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dsa_param = ParamDiffAug()

    # ---- data ----
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = \
        get_dataset(args.dataset, args.data_path)
    train_imgs, train_labs = M.stack_dataset(dst_train, device)
    test_imgs, test_labs = M.stack_dataset(dst_test, device)
    N_total = train_imgs.shape[0]
    m = torch.tensor(mean, device=device).view(1, channel, 1, 1)
    s = torch.tensor(std, device=device).view(1, channel, 1, 1)
    norm = lambda x01: (x01 - m) / s
    denorm = lambda xn: xn * s + m

    N_p = int(round(args.budget * N_total))
    print('%s N_total=%d budget=%.4f -> N_p=%d poisons' % (get_time(), N_total, args.budget, N_p))

    # ---- distilled S + surrogates (reused from cache) ----
    ckpt = torch.load(args.syn_data_path, map_location='cpu', weights_only=False)
    image_syn, label_syn = ckpt['data'][-1]
    image_syn, label_syn = image_syn.to(device), label_syn.to(device)
    args.attack = 'fc'                          # this comparison is fc-only
    surrogates = build_surrogates(args, image_syn, label_syn, train_imgs, train_labs,
                                  test_imgs, test_labs, channel, num_classes, im_size,
                                  dsa_param, device)

    # Hessian set for EXACT IF = the set the surrogate's risk is defined on.
    if args.if_hess_source == 'full' or args.surrogate_on_full_data:
        hsize = args.if_hess_size if args.if_hess_size > 0 else 2000
        perm = torch.randperm(N_total, device=device)[:hsize]
        hess_imgs, hess_labs = train_imgs[perm], train_labs[perm]
        print('%s exact-IF Hessian set = %d real train images' % (get_time(), len(hess_imgs)))
    else:
        hess_imgs, hess_labs = image_syn, label_syn
        print('%s exact-IF Hessian set = distilled S (%d images)' % (get_time(), len(hess_imgs)))
    hess_bs = args.if_hess_size if (args.if_hess_size > 0 and args.if_hess_source != 'full') else 0

    # ---- per pair / target ----
    g_cpu = torch.Generator(device='cpu').manual_seed(args.seed)
    all_rows = []
    # method -> accumulators
    agg = {mth: {'asr': [], 'cta': [], 'sel_t': []} for mth in args.methods}

    for pair in args.class_pairs:
        y_adv, target_class = M.parse_pair(pair, class_names)
        print('\n%s ########## pair %s : y_adv=%d(%s) target=%d(%s) ##########'
              % (get_time(), pair, y_adv, class_names[y_adv], target_class, class_names[target_class]))
        t_idx_all = (test_labs == target_class).nonzero(as_tuple=True)[0].cpu()
        if args.target_select == 'random':
            perm = torch.randperm(len(t_idx_all), generator=g_cpu)[:args.num_targets]
            chosen = t_idx_all[perm].tolist()
        else:
            chosen = t_idx_all[:args.num_targets].tolist()
        print('  targets: %s' % chosen)

        for ti, tidx in enumerate(chosen):
            x_t_norm = test_imgs[tidx]
            for mth in args.methods:
                # ---------------- SELECTION (the only thing that differs) ------
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                if mth == 'smart':
                    base_idx = M.select_base(surrogates, train_imgs, train_labs, x_t_norm,
                                             y_adv, N_p, args.lambda_margin, device,
                                             base_dist=args.base_dist, multilayer=args.multilayer)
                elif mth == 'exact':
                    base_idx = select_base_influence(
                        surrogates, train_imgs, train_labs, x_t_norm, y_adv, N_p,
                        hess_imgs, hess_labs, device,
                        damping=args.if_damping, cg_iters=args.if_cg_iters,
                        cg_tol=args.if_cg_tol, fd_h=args.if_fd_h,
                        last_layer=args.if_last_layer, hess_bs=hess_bs,
                        max_surrogates=args.if_max_surrogates, verbose=args.verbose)
                elif mth == 'random':
                    base_idx = M.select_base_random(train_labs, y_adv, N_p, device)
                else:
                    raise ValueError('unknown method %r' % mth)
                if device == 'cuda':
                    torch.cuda.synchronize()
                sel_t = time.perf_counter() - t0
                agg[mth]['sel_t'].append(sel_t)

                if args.select_only:
                    # SELECTION-ONLY: skip craft + victims; just time the selection.
                    obj, linf, asr, cta = (float('nan'),) * 4
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    print('  [%-6s | %s t%d/%d idx=%d] select=%.3fs (N_p=%d, select-only)'
                          % (mth, pair, ti + 1, len(chosen), tidx, sel_t, N_p))
                else:
                    # ---------------- craft (identical fc) ---------------------
                    base01 = denorm(train_imgs[base_idx]).clamp(0.0, 1.0).detach()
                    x_adv01, obj = M.craft_fc(surrogates, base01, x_t_norm, norm,
                                              args.epsilon, args.pgd_steps, args.pgd_alpha,
                                              device, single_surrogate=args.single_surrogate)
                    linf = (x_adv01 - base01).abs().max().item()
                    poisoned = train_imgs.clone()
                    poisoned[base_idx] = norm(x_adv01)

                    # ---------------- victims from scratch (identical) ---------
                    preds, ctas = [], []
                    for vi in range(args.num_victims):
                        net = get_network(args.model, channel, num_classes, im_size)
                        net = M.train_from_scratch(net, poisoned, train_labs, args.victim_epochs,
                                                   args.victim_lr, args.victim_bs, args.victim_decay,
                                                   device, weight_decay=0.0, aug=args.victim_aug,
                                                   dsa_strategy=args.dsa_strategy, dsa_param=dsa_param)
                        preds.append(M.predict_target(net, x_t_norm))
                        ctas.append(M.test_acc(net, test_imgs, test_labs, device))
                        del net
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                    asr = 100.0 * sum(p == y_adv for p in preds) / args.num_victims
                    cta = float(np.mean(ctas))
                    agg[mth]['asr'].append(asr)
                    agg[mth]['cta'].append(cta)
                    print('  [%-6s | %s t%d/%d idx=%d] select=%.3fs craft_obj=%.4f linf=%.4f '
                          'ASR=%.0f%% CTA=%.4f'
                          % (mth, pair, ti + 1, len(chosen), tidx, sel_t, obj, linf, asr, cta))

                all_rows.append({
                    'pair': pair, 'method': mth, 'target_idx': tidx, 'y_adv': y_adv,
                    'select_time_s': sel_t, 'asr': asr, 'cta': cta,
                    'craft_obj': obj, 'realized_linf': linf, 'N_p': N_p,
                })

    # ---- summary ----
    ntar = args.num_targets * len(args.class_pairs)
    print('\n%s ============ SELECTION %s (fc, %s) ============'
          % (get_time(), 'TIMING (select-only)' if args.select_only else 'COMPARISON', args.surrogate_model))
    for mth in args.methods:
        t = np.array(agg[mth]['sel_t'])
        if len(t) == 0:
            continue
        line = ('  %-7s | %2d targets | sel-time mean / median / total = %.3fs / %.3fs / %.1fs'
                % (mth, ntar, t.mean(), np.median(t), t.sum()))
        if not args.select_only and agg[mth]['asr']:
            a, c = np.array(agg[mth]['asr']), np.array(agg[mth]['cta'])
            line += '   | ASR=%.1f%% +/- %.1f | CTA=%.4f' % (a.mean(), a.std(), c.mean())
        print(line)
    # head-to-head time speedup
    if 'smart' in agg and 'exact' in agg and agg['smart']['sel_t'] and agg['exact']['sel_t']:
        sm, ex = np.mean(agg['smart']['sel_t']), np.mean(agg['exact']['sel_t'])
        print('  exact / smart selection time ratio = %.1fx' % (ex / max(sm, 1e-9)))

    tag = 'select_cmp_fc_%s_b%d_eps%d' % (
        args.model, round(args.budget * 1e4), round(args.epsilon * 255))
    with open(os.path.join(args.out_dir, 'results_%s.json' % tag), 'w') as f:
        json.dump({'rows': all_rows, 'summary': {mth: {
            'asr_mean': float(np.mean(agg[mth]['asr'])) if agg[mth]['asr'] else None,
            'cta_mean': float(np.mean(agg[mth]['cta'])) if agg[mth]['cta'] else None,
            'sel_time_mean': float(np.mean(agg[mth]['sel_t'])) if agg[mth]['sel_t'] else None,
            'sel_time_total': float(np.sum(agg[mth]['sel_t'])) if agg[mth]['sel_t'] else None,
        } for mth in args.methods}, 'args': vars(args)}, f, indent=2)
    if all_rows:
        with open(os.path.join(args.out_dir, 'results_%s.csv' % tag), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print('%s wrote results_%s.{json,csv} to %s' % (get_time(), tag, args.out_dir))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='smart-select vs exact influence-function '
                                            'selection in the fc poisoning setting (selection-only ablation).')
    # data / model (mirror main_IF)
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='data')
    p.add_argument('--model', type=str, default='ConvNetBN')
    p.add_argument('--out_dir', type=str, default='result/select_cmp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    p.add_argument('--syn_data_path', type=str, default='result/res_DM_CIFAR10_ConvNet_100ipc.pt')
    p.add_argument('--surrogate_model', type=str, default='ConvNet')
    p.add_argument('--num_surrogates', type=int, default=10)
    p.add_argument('--surrogate_epochs', type=int, default=1000)
    p.add_argument('--surrogate_lr', type=float, default=0.01)
    p.add_argument('--surrogate_bs', type=int, default=256)
    p.add_argument('--surrogate_on_full_data', action='store_true', default=False)
    # fc craft (identical across methods)
    p.add_argument('--class_pairs', nargs='+', default=['dog-bird'])
    p.add_argument('--budget', type=float, default=0.01)
    p.add_argument('--epsilon', type=float, default=8.0 / 255.0)
    p.add_argument('--pgd_steps', type=int, default=250)
    p.add_argument('--pgd_alpha', type=float, default=1.0 / 255.0)
    p.add_argument('--single_surrogate', action='store_true', default=False)
    # smart-select knobs (so 'smart' matches your main.sh exactly)
    p.add_argument('--lambda_margin', type=float, default=1.0)
    p.add_argument('--base_dist', type=str, default='l2', choices=['l2', 'cosine'])
    p.add_argument('--multilayer', action='store_true', default=False)
    # victim protocol (identical across methods)
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--num_victims', type=int, default=6)
    p.add_argument('--target_select', type=str, default='random', choices=['random', 'first'])
    p.add_argument('--victim_epochs', type=int, default=60)
    p.add_argument('--victim_lr', type=float, default=0.1)
    p.add_argument('--victim_bs', type=int, default=125)
    p.add_argument('--victim_decay', nargs='+', type=int, default=[40])
    p.add_argument('--victim_aug', action='store_true', default=False)
    p.add_argument('--cache_dir', type=str, default='result/cache')
    # comparison / EXACT-IF knobs
    p.add_argument('--methods', nargs='+', default=['smart', 'exact'],
                   choices=['smart', 'exact', 'random'],
                   help='selection rules to compare (everything else identical)')
    p.add_argument('--select_only', action='store_true', default=False,
                   help='only run + time the SELECTION step (skip craft and victim '
                        'training); use to compare selection cost across architectures')
    p.add_argument('--if_damping', type=float, default=0.01,
                   help='Hessian damping (H + damping*I) for CG stability / non-PD nets')
    p.add_argument('--if_cg_iters', type=int, default=100, help='max conjugate-gradient iterations')
    p.add_argument('--if_cg_tol', type=float, default=1e-4, help='CG relative residual tolerance')
    p.add_argument('--if_fd_h', type=float, default=1e-2,
                   help='central finite-difference step for the directional derivative <s, grad l(z)>')
    p.add_argument('--if_last_layer', action='store_true', default=False,
                   help='restrict influence to the final linear layer (cheaper, classic IF)')
    p.add_argument('--if_hess_source', type=str, default='syn', choices=['syn', 'full'],
                   help="dataset for the Hessian: 'syn' = distilled S (matches syn surrogates), "
                        "'full' = a random real-data subsample")
    p.add_argument('--if_hess_size', type=int, default=0,
                   help='subsample size for the Hessian set (0 = use all of S / 2000 if full)')
    p.add_argument('--if_max_surrogates', type=int, default=0,
                   help='cap how many surrogates EXACT-IF averages over (0 = all; smart always uses all)')
    p.add_argument('--log_file', type=str, default='',
                   help='tee all stdout/stderr to this file, line-buffered (no delay)')
    p.add_argument('--verbose', action='store_true', default=False)
    main(p.parse_args())

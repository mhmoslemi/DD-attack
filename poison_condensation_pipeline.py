"""
Targeted clean-label feature-collision poisoning of the Dataset Condensation (DM) pipeline.
Single-file implementation of Eqs. (1)-(3).

Pipeline (one run):
  0. Load the *already-distilled clean* synthetic set S produced by your correct step-1 DM run.
  1. Train an ENSEMBLE of K surrogate ConvNets on S (these define f_theta_f, the feature extractor).
     -> reuses utils.evaluate_synset, so the surrogate training matches your tested procedure.
  2. SELECTION (Eq. 1): from real training images of class y_adv, pick the N_p instances minimizing
        z(||embed(x) - embed(x_target)||^2) + lambda * z(M(x)),  M(x) = Z_{y_adv}(x) - max_{j != y_adv} Z_j(x)
     averaged over the ensemble. (z(.) = per-pool standardization so lambda is scale-free.)
  3. GENERATOR (Eq. 2): train G_phi so that f(x + G_phi(x)) collides with f(x_target),
     with ||perturbation||_inf <= epsilon (imperceptibility budget), averaged over the ensemble.
  4. INJECT (Eq. 3): replace the selected base images with their perturbed versions, KEEPING label y_adv
     (clean-label). All other images untouched.
  5. CONDENSE: run DM distribution matching over the poisoned training pool -> distilled poisoned set S'.
  6. SAVE S' in the exact format your scripts expect:  {'data': [[image_syn, label_syn]], ...}.
  7. EVAL: train fresh victims on S' (and on clean S as baseline); report CTA + ASR on x_target.

KEY FIXES vs the multi-file version:
  * Everything stays in the SAME normalized space (the one DM uses). No /255 corruption, no [0,1] vs
    normalized mismatch. The collision is computed in ConvNet.embed space, exactly what condensation matches.
  * mean/std come from utils.get_dataset (not hardcoded), so train/test/eval all agree.
  * The perturbation itself is bounded by epsilon, and x_target's true label is checked against y_adv.

NOTE ON STRENGTH (read this):
  DM matches per-class feature MEANS. Poisoning N_p of ~5000 images in class y_adv shifts that mean by
  ~ N_p / 5000. With N_p=200 that is ~4% and likely too weak to move the boundary, so ASR can be ~0 even
  when the code is correct. The main levers are: increase --N_p (e.g. 500-1500), increase --epsilon, and
  give condensation enough --Iteration. This is a property of attacking a mean-matching objective, not a bug.

USAGE:
  Place this file in your DatasetCondensation repo (next to utils.py).
  Run your correct step-1 DM distillation first to produce the clean S .pt, then:

  python poison_condensation_pipeline.py \
      --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
      --out_path      result/res_DM_CIFAR10_ConvNet_50ipc_attack.pt \
      --target_idx 42 --y_adv 3 --N_p 500 --epsilon 0.03137 \
      --num_surrogates 3 --surrogate_epochs 300 \
      --gen_epochs 2000 --Iteration 5000 \
      --num_victims 5 --victim_epochs 1000
"""

import warnings
warnings.filterwarnings("ignore")

import os
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse your repo's tested machinery so we stay consistent with steps 1/4.
from utils import (
    get_loops, get_dataset, get_network, evaluate_synset,
    get_time, DiffAugment, ParamDiffAug,
)


# ============================================================================
# Generator G_phi  (Eq. 2):  produces an L_inf <= epsilon perturbation in [0,1] pixel space.
# ============================================================================
class PerturbationGenerator(nn.Module):
    def __init__(self, channel=3, epsilon=8.0 / 255.0, width=32):
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.Conv2d(channel, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, channel, 3, padding=1), nn.Tanh(),  # output in [-1, 1]
        )

    def delta(self, x01):
        # tanh output in [-1,1], scaled by epsilon -> ||delta||_inf <= epsilon (the visible budget).
        return self.epsilon * self.net(x01)

    def forward(self, x01):
        return torch.clamp(x01 + self.delta(x01), 0.0, 1.0)


# ============================================================================
# Small helpers
# ============================================================================
def feat_fn(net):
    """ConvNet.embed = penultimate features (the space DM matches)."""
    return net.module.embed if torch.cuda.device_count() > 1 else net.embed


def standardize(v, eps=1e-8):
    return (v - v.mean()) / (v.std() + eps)


# ============================================================================
# 1. Surrogate ensemble (reuse evaluate_synset so training matches the pipeline)
# ============================================================================
def train_surrogates(image_syn, label_syn, testloader, channel, num_classes, im_size, args, k):
    nets = []
    for i in range(k):
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        # evaluate_synset trains net on (image_syn, label_syn) using args.epoch_eval_train / args.lr_net / DSA.
        net, _, acc = evaluate_synset(i, net, copy.deepcopy(image_syn.detach()),
                                      copy.deepcopy(label_syn.detach()), testloader, args)
        print(f"[surrogate {i+1}/{k}] clean test acc on S = {acc:.4f}")
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        nets.append(net)
    return nets


# ============================================================================
# 2. Base selection (Eq. 1), ensemble-averaged, scale-free via standardization
# ============================================================================
@torch.no_grad()
def select_base(nets, images_norm, labels, x_target_norm, y_adv, N_p, lam, device):
    cls_idx = (labels == y_adv).nonzero(as_tuple=True)[0]
    if len(cls_idx) < N_p:
        raise ValueError(f"class {y_adv} has only {len(cls_idx)} images < N_p={N_p}")
    cand = images_norm[cls_idx]  # normalized space, same as everything else

    score = torch.zeros(len(cls_idx), device=device)
    for net in nets:
        emb = feat_fn(net)
        f_t = emb(x_target_norm.unsqueeze(0))  # [1, D]
        dists, margins = [], []
        for i in range(0, len(cand), 512):
            b = cand[i:i + 512]
            d = ((emb(b) - f_t) ** 2).sum(dim=1)                # ||f(x) - f(x_target)||^2
            z = net(b)                                          # logits Z(x)
            z_adv = z[:, y_adv].clone()
            z_other = z.clone()
            z_other[:, y_adv] = float("-inf")
            m = z_adv - z_other.max(dim=1).values               # M(x) = Z_yadv - max_{j!=yadv} Z_j
            dists.append(d)
            margins.append(m)
        d = standardize(torch.cat(dists))
        m = standardize(torch.cat(margins))
        score += d + lam * m
    score /= len(nets)

    sel_local = torch.topk(score, k=N_p, largest=False).indices  # minimize the objective
    return cls_idx[sel_local]


# ============================================================================
# 3. Generator training (Eq. 2), ensemble-averaged collision loss
# ============================================================================
def train_generator(gen, nets, base01, x_target_norm, norm, epochs, lr, device):
    gen.train()
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    with torch.no_grad():
        f_tgts = [feat_fn(net)(x_target_norm.unsqueeze(0)).detach() for net in nets]

    for ep in range(epochs):
        opt.zero_grad()
        x_adv01 = gen(base01)
        x_adv_norm = norm(x_adv01)              # back to the space the surrogates live in
        loss = 0.0
        for net, f_t in zip(nets, f_tgts):
            f = feat_fn(net)(x_adv_norm)
            loss = loss + F.mse_loss(f, f_t.expand_as(f))
        loss = loss / len(nets)
        loss.backward()
        opt.step()
        if (ep + 1) % 500 == 0:
            print(f"  [generator] ep {ep+1}/{epochs}  collision MSE = {loss.item():.6f}")
    gen.eval()
    return gen


@torch.no_grad()
def collision_mse(nets, x01, x_target_norm, norm):
    """Ensemble-mean feature-collision MSE between normalized x01 and the target. Diagnostic only."""
    total = 0.0
    for net in nets:
        emb = feat_fn(net)
        f_t = emb(x_target_norm.unsqueeze(0))
        f = emb(norm(x01))
        total += F.mse_loss(f, f_t.expand_as(f)).item()
    return total / len(nets)


def craft_pgd(nets, base01, x_target_norm, norm, epsilon, steps, alpha, device):
    """Per-sample L_inf PGD collision (Eq. 2 with delta parameterized directly, not amortized).
    Each base image gets its own perturbation, which is far more expressive than a shared G_phi.
    Returns perturbed images in [0,1]."""
    base01 = base01.detach()
    with torch.no_grad():
        f_tgts = [feat_fn(net)(x_target_norm.unsqueeze(0)).detach() for net in nets]
    delta = torch.empty_like(base01).uniform_(-epsilon, epsilon)
    delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01).detach().requires_grad_(True)

    for t in range(steps):
        x_adv_norm = norm(torch.clamp(base01 + delta, 0.0, 1.0))
        loss = 0.0
        for net, f_t in zip(nets, f_tgts):
            f = feat_fn(net)(x_adv_norm)
            loss = loss + F.mse_loss(f, f_t.expand_as(f))
        loss = loss / len(nets)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - alpha * grad.sign()
            delta = delta.clamp_(-epsilon, epsilon)
            delta = (torch.clamp(base01 + delta, 0.0, 1.0) - base01)  # keep image valid
        delta = delta.detach().requires_grad_(True)
        if (t + 1) % max(1, steps // 10) == 0:
            print(f"  [pgd] step {t+1}/{steps}  collision MSE = {loss.item():.6f}")
    return torch.clamp(base01 + delta.detach(), 0.0, 1.0)


# ============================================================================
# 4. DM distribution-matching condensation over the (poisoned) training pool.
#    This is your correct step-1 loop, lifted verbatim and operating on images_all.
# ============================================================================
def condense(images_all, labels_all, indices_class, channel, num_classes, im_size, args):
    def get_images(c, n):
        idx = np.random.permutation(indices_class[c])[:n]
        return images_all[idx]

    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
                            dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)],
                             dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    if args.init == 'real':
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data

    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()
    print('%s condensation begins' % get_time())

    for it in range(args.Iteration + 1):
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        net.train()
        for p in net.parameters():
            p.requires_grad = False
        embed = feat_fn(net)

        loss = torch.tensor(0.0).to(args.device)
        for c in range(num_classes):
            img_real = get_images(c, args.batch_real)
            img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                (args.ipc, channel, im_size[0], im_size[1]))
            if args.dsa:
                seed = int(time.time() * 1000) % 100000
                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
            out_real = embed(img_real).detach()
            out_syn = embed(img_syn)
            loss += torch.sum((torch.mean(out_real, dim=0) - torch.mean(out_syn, dim=0)) ** 2)

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()

        if it % 500 == 0:
            print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss.item() / num_classes))

    return image_syn.detach(), label_syn.detach()


# ============================================================================
# 5. Victim training + ASR/CTA evaluation (reuse evaluate_synset for training)
# ============================================================================
def evaluate_attack(image_syn, label_syn, testloader, x_target_norm, y_adv,
                    channel, num_classes, im_size, args, n_victims, tag):
    accs, asr_hits = [], 0
    xt = x_target_norm.unsqueeze(0).to(args.device)
    for i in range(n_victims):
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        net, _, acc = evaluate_synset(i, net, copy.deepcopy(image_syn.detach()),
                                      copy.deepcopy(label_syn.detach()), testloader, args)
        net.eval()
        with torch.no_grad():
            pred = net(xt).argmax(dim=1).item()
        hit = int(pred == y_adv)
        asr_hits += hit
        accs.append(acc)
        print(f"  [{tag} victim {i+1}/{n_victims}] CTA={acc:.4f}  target->{pred}  {'HIT' if hit else 'miss'}")
    return float(np.mean(accs)), float(np.std(accs)), 100.0 * asr_hits / n_victims, asr_hits


# ============================================================================
# Main
# ============================================================================
def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.method = 'DM'
    args.dsa_param = ParamDiffAug()
    args.dsa = args.dsa_strategy not in ['none', 'None']
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    os.makedirs(args.save_path, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- dataset (mean/std come from here; used everywhere) ---
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
        get_dataset(args.dataset, args.data_path)

    m = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    s = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    norm = lambda x01: (x01 - m) / s
    denorm = lambda xn: xn * s + m

    # --- build the (normalized) training pool, identical ordering to torchvision CIFAR ---
    images_all = torch.cat([torch.unsqueeze(dst_train[i][0], 0) for i in range(len(dst_train))], 0).to(args.device)
    labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))],
                              dtype=torch.long, device=args.device)
    indices_class = [[] for _ in range(num_classes)]
    for i, lab in enumerate(labels_all.tolist()):
        indices_class[lab].append(i)

    # --- target instance (from TEST set), in normalized space ---
    x_target_norm, target_true = dst_test[args.target_idx]
    x_target_norm = x_target_norm.to(args.device)
    print(f"\nTarget idx {args.target_idx}: true label = {target_true} ({class_names[target_true]}), "
          f"adversarial class y_adv = {args.y_adv} ({class_names[args.y_adv]})")
    if target_true == args.y_adv:
        print("  !!! WARNING: x_target's true label == y_adv. ASR is trivially ~100% and the attack is "
              "meaningless. Pick a --target_idx whose true label differs from --y_adv.")

    # --- load the clean distilled set S (must already be normalized, from your step-1 run) ---
    print(f"\nLoading clean distilled S from {args.syn_data_path}")
    ckpt = torch.load(args.syn_data_path, map_location='cpu', weights_only=False)
    image_syn_clean, label_syn_clean = ckpt['data'][-1]
    image_syn_clean = image_syn_clean.to(args.device)
    label_syn_clean = label_syn_clean.to(args.device)
    # Sanity check: normalized CIFAR has values well outside [0,1]. If this trips, S was saved in [0,1]
    # and you must align spaces (do NOT divide by 255 here).
    print(f"  S stats: min={image_syn_clean.min():.3f} max={image_syn_clean.max():.3f} "
          f"(expected roughly [-2.5, 2.7] for normalized CIFAR)")

    # ---------------- 1. surrogates ----------------
    print(f"\n=== Training {args.num_surrogates} surrogate(s) on clean S ===")
    args.epoch_eval_train = args.surrogate_epochs
    nets = train_surrogates(image_syn_clean, label_syn_clean, testloader,
                            channel, num_classes, im_size, args, args.num_surrogates)

    # ---------------- 2. selection (Eq. 1) ----------------
    print(f"\n=== Selecting T_base (N_p={args.N_p}) from class {args.y_adv} ===")
    base_idx = select_base(nets, images_all, labels_all, x_target_norm,
                           args.y_adv, args.N_p, args.lambda_margin, args.device)
    print(f"  selected {len(base_idx)} indices, e.g. {base_idx[:10].tolist()}")

    # ---------------- 3. generator (Eq. 2) ----------------
    print(f"\n=== Optimizing generator (epsilon={args.epsilon:.5f}, {args.gen_epochs} ep) ===")
    base01 = denorm(images_all[base_idx]).clamp(0.0, 1.0).detach()   # work in pixel space
    gen = PerturbationGenerator(channel=channel, epsilon=args.epsilon).to(args.device)
    gen = train_generator(gen, nets, base01, x_target_norm, norm,
                          args.gen_epochs, args.gen_lr, args.device)

    with torch.no_grad():
        x_adv01 = gen(base01)
        linf = (x_adv01 - base01).abs().max().item()
        print(f"  realized ||perturbation||_inf = {linf:.5f} (budget {args.epsilon:.5f})")

    # ---------------- 4. inject (Eq. 3): clean-label, labels untouched ----------------
    images_all[base_idx] = norm(x_adv01)
    print(f"  anchored {len(base_idx)} poisoned images into the pool (labels kept = {args.y_adv})")

    # ---------------- 5. condense poisoned pool -> S' ----------------
    print(f"\n=== Condensing poisoned pool ({args.Iteration} iters) ===")
    image_syn_poison, label_syn_poison = condense(images_all, labels_all, indices_class,
                                                  channel, num_classes, im_size, args)

    # ---------------- 6. save S' in the format your other scripts expect ----------------
    save_obj = {
        'data': [[image_syn_poison.cpu(), label_syn_poison.cpu()]],
        'poisoned_indices': base_idx.cpu(),
        'target_idx': args.target_idx,
        'target_true_label': int(target_true),
        'y_adv': args.y_adv,
        'epsilon': args.epsilon,
        'N_p': args.N_p,
    }
    torch.save(save_obj, args.out_path)
    print(f"\nSaved distilled POISONED set S' to: {args.out_path}")

    # ---------------- 7. evaluate ----------------
    if not args.skip_eval:
        args.epoch_eval_train = args.victim_epochs
        print(f"\n=== Evaluating ({args.num_victims} victims each) ===")
        if args.baseline:
            cb_mean, cb_std, cb_asr, cb_hits = evaluate_attack(
                image_syn_clean, label_syn_clean, testloader, x_target_norm, args.y_adv,
                channel, num_classes, im_size, args, args.num_victims, "CLEAN")
        pm_mean, pm_std, pm_asr, pm_hits = evaluate_attack(
            image_syn_poison, label_syn_poison, testloader, x_target_norm, args.y_adv,
            channel, num_classes, im_size, args, args.num_victims, "POISON")

        print("\n==================== RESULTS ====================")
        if args.baseline:
            print(f" CLEAN  S : CTA = {cb_mean:.4f} +/- {cb_std:.4f} | ASR = {cb_asr:.1f}% ({cb_hits}/{args.num_victims})")
        print(f" POISON S': CTA = {pm_mean:.4f} +/- {pm_std:.4f} | ASR = {pm_asr:.1f}% ({pm_hits}/{args.num_victims})")
        print("=================================================")
        print("Attack worked iff CTA stays ~unchanged AND POISON ASR >> CLEAN ASR.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Targeted clean-label poisoning of DM condensation (Eqs 1-3)")
    # data / model
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--model', type=str, default='ConvNet')
    p.add_argument('--ipc', type=int, default=50)
    p.add_argument('--data_path', type=str, default='data')
    p.add_argument('--save_path', type=str, default='result')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init', type=str, default='real')
    p.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    p.add_argument('--dis_metric', type=str, default='ours')
    # required input / output
    p.add_argument('--syn_data_path', type=str, required=True, help="clean distilled S .pt from step 1")
    p.add_argument('--out_path', type=str, default='result/res_DM_CIFAR10_ConvNet_50ipc_attack.pt')
    # attack
    p.add_argument('--target_idx', type=int, default=42, help="target image index in the TEST set")
    p.add_argument('--y_adv', type=int, default=3, help="adversarial class to push x_target into")
    p.add_argument('--N_p', type=int, default=500, help="number of base instances to poison")
    p.add_argument('--epsilon', type=float, default=8.0 / 255.0, help="L_inf perturbation budget")
    p.add_argument('--lambda_margin', type=float, default=1.0, help="lambda in Eq. 1 (margin weight)")
    # surrogate / generator
    p.add_argument('--num_surrogates', type=int, default=3)
    p.add_argument('--surrogate_epochs', type=int, default=300)
    p.add_argument('--gen_epochs', type=int, default=2000)
    p.add_argument('--gen_lr', type=float, default=1e-3)
    # condensation
    p.add_argument('--Iteration', type=int, default=5000)
    p.add_argument('--lr_img', type=float, default=1.0)
    p.add_argument('--lr_net', type=float, default=0.01, help="used by evaluate_synset")
    p.add_argument('--batch_real', type=int, default=256)
    p.add_argument('--batch_train', type=int, default=256, help="used by evaluate_synset")
    # eval
    p.add_argument('--num_victims', type=int, default=5)
    p.add_argument('--victim_epochs', type=int, default=1000)
    p.add_argument('--baseline', action='store_true', default=True, help="also eval clean S for comparison")
    p.add_argument('--skip_eval', action='store_true', default=False)

    main(p.parse_args())
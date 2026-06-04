"""
Targeted data-poisoning attacks.

Three attacks, all clean-label (the poisoned images keep their true label
y_adv; only their pixels change) and all crafted in [0,1] pixel space under an
L-inf budget eps, then normalised (differentiably) before being fed to the
network, with DiffAugment applied after normalisation to mirror condensation:

  gradient_matching_attack : Algorithm 1 / Eq.(4)  -- Witches' Brew. Matches the
      *full network gradient* of CE on the target sample and on the poison set,
      via cosine similarity, over an ensemble of pretrained clean nets. This is
      a second-order objective in delta (the poison gradient is itself
      differentiated w.r.t. delta), so we build the inner grad with
      create_graph=True.

  dm_poisoning_attack : Algorithm 2 / Eq.(5)  -- the paper's proposed attack.
      Matches the *representation* of the target and the mean poison
      representation, in the squared-L2 sense, over freshly sampled random
      (untrained) ConvNets. First-order only; no pretraining.

  direct_attack : the "direct attack" baseline -- just P copies of the target
      image x_t (labelled y_adv). No optimisation.

Optimiser: signed Adam (we feed sign(grad) to Adam), which the paper notes is
equivalent to signed momentum SGD, with step size = `step_size` (1/255 in the
paper). After each step we project delta back onto the eps-ball and then onto
the [0,1] box. `restarts` independent runs are performed and the delta with the
lowest objective seen across all runs/steps is returned (the paper's text says
to keep the best across restarts; see README for the pseudocode discrepancy).
`round_to_255` optionally snaps the final poisons to the 8-bit grid.
"""
import torch
import torch.nn.functional as F

from utils import DiffAugment, get_embed, get_network, TensorDataset, get_time


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _flat_grad(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def _cosine(a, b, eps=1e-8):
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


def _round_to_255(pixel):
    return torch.round(pixel * 255.0) / 255.0


def _project(delta, base_pixel, eps):
    """In-place projection of delta onto the eps L-inf ball then the [0,1] box."""
    delta.data.clamp_(-eps, eps)
    delta.data = torch.clamp(base_pixel + delta.data, 0.0, 1.0) - base_pixel


def _aug_seed(param_present):
    # A fresh seed per call so each iteration samples new augmentation params,
    # but real/synthetic (or target/poison) share it within the call.
    return int(torch.randint(0, 100000, size=(1,)).item())


# --------------------------------------------------------------------------- #
# clean-model pretraining (for gradient matching)
# --------------------------------------------------------------------------- #
def _train_clean_net(net, images_norm, labels, epochs, lr, batch, device,
                     dsa, dsa_strategy, dsa_param):
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)
    crit = torch.nn.CrossEntropyLoss().to(device)
    ds = TensorDataset(images_norm, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                         num_workers=0)
    for _ in range(epochs):
        for img, lab in loader:
            img = img.float().to(device)
            lab = lab.long().to(device)
            if dsa:
                img = DiffAugment(img, dsa_strategy, param=dsa_param)
            opt.zero_grad()
            loss = crit(net(img), lab)
            loss.backward()
            opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def pretrain_models(cfg, meta, images_norm, labels, device):
    """Train `cfg.poison.num_pretrain_models` independent clean ConvNets, each
    to a different number of epochs spread over
    [pretrain_min_epochs, pretrain_epochs] (Geiping et al. craft on models at
    different training stages)."""
    n = cfg.poison.num_pretrain_models
    lo = cfg.poison.pretrain_min_epochs
    hi = cfg.poison.pretrain_epochs
    if n == 1:
        epoch_grid = [hi]
    else:
        epoch_grid = [int(round(lo + (hi - lo) * i / (n - 1))) for i in range(n)]

    nets = []
    for i, ep in enumerate(epoch_grid):
        net = get_network(cfg.model.arch, meta.channel, meta.num_classes,
                          meta.im_size, device=device)
        print('%s pretrain model %d/%d for %d epochs' % (get_time(), i + 1, n, ep))
        net = _train_clean_net(
            net, images_norm, labels, ep, cfg.evaluation.lr_net,
            cfg.evaluation.batch_train, device,
            cfg.condensation.dsa, cfg.condensation.dsa_strategy, cfg.dsa_param)
        nets.append(net)
    return nets


# --------------------------------------------------------------------------- #
# Attack 1: gradient matching  (Eq. 4)
# --------------------------------------------------------------------------- #
def gradient_matching_attack(base_pixel, x_t_pixel, y_adv, nets, normalizer,
                             cfg, device):
    """base_pixel: [P, C, H, W] clean poison-base images in [0,1].
    x_t_pixel:    [1, C, H, W] target image in [0,1]. Returns poisons [P,...]."""
    eps = float(cfg.poison.eps)
    step = float(cfg.poison.step_size)
    crit = torch.nn.CrossEntropyLoss().to(device)
    y_t = torch.full((1,), y_adv, dtype=torch.long, device=device)
    y_p = torch.full((base_pixel.shape[0],), y_adv, dtype=torch.long, device=device)

    # Target gradients are constant w.r.t. delta -> precompute & detach (per net).
    g_targets = []
    for net in nets:
        params = [p for p in net.parameters()]
        out_t = net(normalizer(x_t_pixel))
        loss_t = crit(out_t, y_t)
        g_t = torch.autograd.grad(loss_t, params)
        g_targets.append(_flat_grad([g.detach() for g in g_t]))

    best_delta = None
    best_obj = float('inf')

    for r in range(cfg.poison.restarts):
        delta = (torch.empty_like(base_pixel).uniform_(-eps, eps)
                 ).requires_grad_(True)
        _project(delta, base_pixel, eps)
        delta.requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=step)

        for t in range(cfg.poison.iterations):
            seed = _aug_seed(True)
            poison_pixel = base_pixel + delta
            obj = 0.0
            for net, g_t in zip(nets, g_targets):
                params = [p for p in net.parameters()]
                inp = DiffAugment(normalizer(poison_pixel),
                                  cfg.condensation.dsa_strategy, seed=seed,
                                  param=cfg.dsa_param) if cfg.condensation.dsa \
                    else normalizer(poison_pixel)
                loss_p = crit(net(inp), y_p)
                g_p = torch.autograd.grad(loss_p, params, create_graph=True)
                obj = obj + (1.0 - _cosine(g_t, _flat_grad(g_p)))
            obj = obj / len(nets)

            grad = torch.autograd.grad(obj, delta)[0]
            opt.zero_grad()
            delta.grad = grad.sign()
            opt.step()
            _project(delta, base_pixel, eps)

            if obj.item() < best_obj:
                best_obj = obj.item()
                best_delta = delta.detach().clone()

    poisons = torch.clamp(base_pixel + best_delta, 0.0, 1.0)
    if cfg.poison.round_to_255:
        poisons = _round_to_255(poisons)
    print('%s gradient_matching: best objective (1-cos) = %.4f' % (get_time(), best_obj))
    return poisons.detach()


# --------------------------------------------------------------------------- #
# Attack 2: DM poisoning  (Eq. 5)
# --------------------------------------------------------------------------- #
def dm_poisoning_attack(base_pixel, x_t_pixel, y_adv, normalizer, cfg, meta,
                        device):
    """Match target vs mean-poison representation over fresh random ConvNets."""
    eps = float(cfg.poison.eps)
    step = float(cfg.poison.step_size)

    best_delta = None
    best_obj = float('inf')

    for r in range(cfg.poison.restarts):
        delta = torch.empty_like(base_pixel).uniform_(-eps, eps)
        _project_tensor = delta  # placeholder for clarity
        delta = delta.requires_grad_(True)
        _project(delta, base_pixel, eps)
        delta.requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=step)

        for t in range(cfg.poison.iterations):
            # Sample a fresh random (untrained) feature extractor: this IS the
            # parameter distribution P_theta the objective expects.
            net = get_network(cfg.model.arch, meta.channel, meta.num_classes,
                              meta.im_size, device=device)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            embed = get_embed(net)

            seed = _aug_seed(True)
            poison_pixel = base_pixel + delta
            if cfg.condensation.dsa:
                inp_p = DiffAugment(normalizer(poison_pixel),
                                    cfg.condensation.dsa_strategy, seed=seed,
                                    param=cfg.dsa_param)
                inp_t = DiffAugment(normalizer(x_t_pixel),
                                    cfg.condensation.dsa_strategy, seed=seed,
                                    param=cfg.dsa_param)
            else:
                inp_p = normalizer(poison_pixel)
                inp_t = normalizer(x_t_pixel)

            r_t = embed(inp_t).detach()                 # [1, d]
            r_p_mean = embed(inp_p).mean(dim=0, keepdim=True)  # [1, d]
            obj = ((r_t - r_p_mean) ** 2).sum()

            grad = torch.autograd.grad(obj, delta)[0]
            opt.zero_grad()
            delta.grad = grad.sign()
            opt.step()
            _project(delta, base_pixel, eps)

            if obj.item() < best_obj:
                best_obj = obj.item()
                best_delta = delta.detach().clone()

    poisons = torch.clamp(base_pixel + best_delta, 0.0, 1.0)
    if cfg.poison.round_to_255:
        poisons = _round_to_255(poisons)
    print('%s dm_poisoning: best objective (||.||^2) = %.6f' % (get_time(), best_obj))
    return poisons.detach()


# --------------------------------------------------------------------------- #
# Attack 3: direct attack (baseline)
# --------------------------------------------------------------------------- #
def direct_attack(base_pixel, x_t_pixel, cfg):
    """Replace the P poison-base images with P copies of the target image."""
    poisons = x_t_pixel.detach().repeat(base_pixel.shape[0], 1, 1, 1).clone()
    if cfg.poison.round_to_255:
        poisons = _round_to_255(poisons)
    return poisons

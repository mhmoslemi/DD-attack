"""
Distribution-matching dataset condensation (Algorithm 3, RDM-DC).

This is the original DM condensation loop (one freshly sampled random ConvNet
per iteration; for each class, embed an augmented real batch and the augmented
synthetic set, and push the synthetic-mean representation toward the real-mean
representation) with exactly one change: the real-mean is replaced by a robust
estimate `aggregate(real_reps, defense, drop_count, ...)`. With defense='none'
this reduces to vanilla DM; with defense='rdmdc' it is the paper's mean
calibration.

Faithful defaults (paper Sec. 5.1): synthetic data initialised from Gaussian
noise N(0, I) (real-image init is noted to make every method vulnerable),
SGD(lr=1.0, momentum=0.5) on the synthetic images, batch_real=256, ipc=50,
20000 iterations.
"""
import copy
import time

import torch

from defenses import aggregate
from utils import DiffAugment, get_embed, get_network, get_time


def distribution_matching(images_norm, labels, indices_class, cfg, meta, device,
                          defense_method, drop_count, log_every=2000):
    channel, (h, w) = meta.channel, meta.im_size
    num_classes = meta.num_classes
    ipc = cfg.condensation.ipc

    def get_images(c, n):
        idx = indices_class[c][torch.randperm(len(indices_class[c]))[:n]]
        return images_norm[idx]

    # ----- initialise synthetic data ----------------------------------------
    image_syn = torch.randn(num_classes * ipc, channel, h, w,
                            dtype=torch.float, device=device)
    label_syn = torch.tensor(
        [c for c in range(num_classes) for _ in range(ipc)],
        dtype=torch.long, device=device)
    if cfg.condensation.init == 'real':
        for c in range(num_classes):
            image_syn.data[c * ipc:(c + 1) * ipc] = get_images(c, ipc).detach().data
    image_syn.requires_grad_(True)

    optimizer_img = torch.optim.SGD([image_syn], lr=cfg.condensation.lr_img,
                                    momentum=cfg.condensation.momentum_img)

    print('%s condensation begins (defense=%s, drop_count=%d)'
          % (get_time(), defense_method, drop_count))

    for it in range(cfg.condensation.iterations + 1):
        net = get_network(cfg.model.arch, channel, num_classes, meta.im_size,
                          device=device)
        net.train()
        for p in net.parameters():
            p.requires_grad_(False)
        embed = get_embed(net)

        loss = torch.tensor(0.0, device=device)
        for c in range(num_classes):
            img_real = get_images(c, cfg.condensation.batch_real)
            img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape(ipc, channel, h, w)

            if cfg.condensation.dsa:
                seed = int(time.time() * 1000) % 100000
                img_real = DiffAugment(img_real, cfg.condensation.dsa_strategy,
                                       seed=seed, param=cfg.dsa_param)
                img_syn = DiffAugment(img_syn, cfg.condensation.dsa_strategy,
                                      seed=seed, param=cfg.dsa_param)

            real_reps = embed(img_real).detach()        # [batch_real, d]
            syn_reps = embed(img_syn)                    # [ipc, d]

            mu_real = aggregate(real_reps, defense_method, drop_count=drop_count,
                                power_iters=cfg.defense.power_iters)
            loss = loss + ((mu_real - syn_reps.mean(dim=0)) ** 2).sum()

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()

        if it % log_every == 0:
            print('%s iter = %05d, loss = %.4f'
                  % (get_time(), it, loss.item() / num_classes))

    return image_syn.detach(), label_syn.detach()

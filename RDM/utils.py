"""
Utilities (trimmed from VICO-UoE/DatasetCondensation `utils.py`).

We keep only what RDM-DC needs:
  * the differentiable Siamese augmentation A_w (DiffAugment + ParamDiffAug),
  * get_network / get_default_convnet_setting for the ConvNet feature extractor,
  * get_embed, a small helper returning net.embed (handles DataParallel),
  * the TensorDataset, epoch, and evaluate_synset training/eval loop,
  * get_time.

Two small compatibility shims relative to the original file:
  1. scipy's rotate import path moved across versions, so we guard it and fall
     back to a no-op if scipy is missing (the DSA pipeline used here does not
     call the scipy-based `augment`, only DiffAugment, so this is harmless).
  2. torch.meshgrid now warns/needs an explicit `indexing` argument; we pass
     indexing='ij' to keep the original (row-major) behaviour.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from networks import ConvNet

# scipy.ndimage.rotate is only used by the legacy `augment` path (not DSA).
# Guard the import so the package runs even when scipy is unavailable.
try:  # newer scipy
    from scipy.ndimage import rotate as scipyrotate
except Exception:  # pragma: no cover
    try:  # older scipy
        from scipy.ndimage.interpolation import rotate as scipyrotate
    except Exception:
        scipyrotate = None


# --------------------------------------------------------------------------- #
# Network construction
# --------------------------------------------------------------------------- #
def get_default_convnet_setting():
    # width 128, depth 3, ReLU, InstanceNorm (GroupNorm), AvgPool.
    return 128, 3, 'relu', 'instancenorm', 'avgpooling'


def get_network(model, channel, num_classes, im_size=(32, 32), device='cpu',
                seed=None):
    """Build a freshly (randomly) initialised network and move it to `device`.

    The original seeds the RNG from wall-clock time so that every call yields a
    different random net (this randomness IS the parameter distribution P_theta
    that distribution matching samples from). We keep that behaviour but allow
    an explicit `seed` for reproducible smoke tests.
    """
    if seed is None:
        seed = int(time.time() * 1000) % 100000
    torch.random.manual_seed(seed)

    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    if model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes,
                      net_width=net_width, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    else:
        raise ValueError(
            'This trimmed utils only builds ConvNet; got %r. Drop in the full '
            'original networks.py + get_network to use other architectures.' % model)

    return net.to(device)


def get_embed(net):
    """Return the feature-extractor method Phi_theta, handling DataParallel."""
    return net.module.embed if isinstance(net, nn.DataParallel) else net.embed


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #
class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


# --------------------------------------------------------------------------- #
# Training / evaluation loop (used to train a net on the synthetic set)
# --------------------------------------------------------------------------- #
def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for datum in dataloader:
        img = datum[0].float().to(args.device)
        if aug and args.dsa:
            img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1),
                              lab.cpu().data.numpy()))

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp
    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    """Train `net` from scratch on the synthetic set, return (net, train_acc, test_acc)."""
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch + 1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer,
                                      criterion, args, aug=True)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
    time_train = time.time() - start

    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion,
                                args, aug=False)
    print('%s Evaluate_%02d: epoch=%04d train_time=%ds train_loss=%.6f '
          'train_acc=%.4f test_acc=%.4f'
          % (get_time(), it_eval, Epoch, int(time_train), loss_train,
             acc_train, acc_test))
    return net, acc_train, acc_test


# --------------------------------------------------------------------------- #
# Differentiable Siamese Augmentation  (A_w in the paper)
# --------------------------------------------------------------------------- #
class ParamDiffAug:
    def __init__(self):
        self.aug_mode = 'S'         # single augmentation per call
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.latestseed = -1
        self.Siamese = False


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    torch.random.manual_seed(param.latestseed)
    param.latestseed += 1


def DiffAugment(x, strategy='', seed=-1, param=None):
    if strategy in ('None', 'none', ''):
        return x

    param.Siamese = (seed != -1)
    param.latestseed = seed

    if param.aug_mode == 'M':
        for p in strategy.split('_'):
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
    elif param.aug_mode == 'S':
        pbties = strategy.split('_')
        set_seed_DiffAug(param)
        p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
        for f in AUGMENT_FNS[p]:
            x = f(x, param)
    else:
        raise ValueError('unknown augmentation mode: %s' % param.aug_mode)
    return x.contiguous()


def rand_scale(x, param):
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    return F.grid_sample(x, grid, align_corners=True)


def rand_rotate(x, param):
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
              [torch.sin(theta[i]), torch.cos(theta[i]), 0]]
             for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    return F.grid_sample(x, grid, align_corners=True)


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        randb[:] = randb[0]
    return x + (randb - 0.5) * ratio


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        rands[:] = rands[0]
    return (x - x_mean) * (rands * ratio) + x_mean


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        randc[:] = randc[0]
    return (x - x_mean) * (randc + ratio) + x_mean


def rand_crop(x, param):
    ratio = param.ratio_crop_pad
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1,
                                  size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1,
                                  size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij',
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y]
    return x.permute(0, 3, 1, 2)


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2),
                             size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2),
                             size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij',
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2,
                         min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2,
                         min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3),
                      dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

"""
Data handling.

Key design choice (faithfulness): perturbations are crafted in PIXEL space
[0, 1] under an L-inf budget, exactly as the paper specifies eps in units of
k/255. We therefore keep the raw training/test tensors in [0, 1] and apply
(per-channel) normalisation as a *differentiable* op (`Normalizer`) only when
feeding a network. This lets the attack project delta onto the [0,1] box and
the eps-ball in the same space the budget is defined in, then optionally
quantise to the 8-bit grid, while the condensation/eval pipelines still see
normalised inputs just like the original code.

`get_raw_dataset` returns:
    meta        : SimpleNamespace(channel, im_size, num_classes, mean, std,
                                  class_names, n_train)
    train_pixel : float tensor [N, C, H, W] in [0, 1]
    train_labels: long tensor [N]
    test_pixel  : float tensor [M, C, H, W] in [0, 1]
    test_labels : long tensor [M]

The 'debug' dataset is pure random tensors (4 classes, 3x16x16); it needs no
torchvision/download and exists only for fast CPU smoke tests of the full
pipeline.
"""
from types import SimpleNamespace

import torch


def get_raw_dataset(dataset, data_path):
    if dataset == 'CIFAR10':
        from torchvision import datasets, transforms
        channel, im_size, num_classes = 3, (32, 32), 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        to_tensor = transforms.ToTensor()  # -> [0,1]
        dst_train = datasets.CIFAR10(data_path, train=True, download=True,
                                     transform=to_tensor)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True,
                                    transform=to_tensor)
        class_names = dst_train.classes
        train_pixel = torch.stack([dst_train[i][0] for i in range(len(dst_train))])
        train_labels = torch.tensor([dst_train[i][1] for i in range(len(dst_train))],
                                    dtype=torch.long)
        test_pixel = torch.stack([dst_test[i][0] for i in range(len(dst_test))])
        test_labels = torch.tensor([dst_test[i][1] for i in range(len(dst_test))],
                                   dtype=torch.long)

    elif dataset == 'TinyImageNet':
        import os
        channel, im_size, num_classes = 3, (64, 64), 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'),
                          map_location='cpu')
        class_names = data['classes']
        train_pixel = data['images_train'].detach().float() / 255.0
        train_labels = data['labels_train'].detach().long()
        test_pixel = data['images_val'].detach().float() / 255.0
        test_labels = data['labels_val'].detach().long()

    elif dataset == 'debug':
        # Tiny synthetic dataset; no torchvision/download needed.
        channel, im_size, num_classes = 3, (16, 16), 4
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        class_names = [str(c) for c in range(num_classes)]
        g = torch.Generator().manual_seed(0)
        n_train, n_test = 400, 80
        train_labels = torch.randint(0, num_classes, (n_train,), generator=g)
        test_labels = torch.randint(0, num_classes, (n_test,), generator=g)
        # Give each class a slightly different mean so condensation has signal.
        train_pixel = torch.rand(n_train, channel, *im_size, generator=g)
        test_pixel = torch.rand(n_test, channel, *im_size, generator=g)
        for c in range(num_classes):
            train_pixel[train_labels == c] = (
                train_pixel[train_labels == c] * 0.5 + 0.1 * c).clamp(0, 1)
            test_pixel[test_labels == c] = (
                test_pixel[test_labels == c] * 0.5 + 0.1 * c).clamp(0, 1)

    else:
        raise ValueError('unknown dataset: %s' % dataset)

    meta = SimpleNamespace(
        channel=channel, im_size=im_size, num_classes=num_classes,
        mean=mean, std=std, class_names=class_names,
        n_train=int(train_labels.shape[0]))
    return meta, train_pixel, train_labels, test_pixel, test_labels


class Normalizer:
    """Differentiable per-channel normalisation (x - mean) / std."""

    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)

    def __call__(self, x):
        return (x - self.mean) / self.std


def build_poisoned_normalized(train_pixel, train_labels, normalizer,
                              poison_idx, poisoned_pixel):
    """Return a normalised training-image tensor where the rows in `poison_idx`
    have been replaced (in place, clean-label) by `poisoned_pixel`.

    The label vector is unchanged: a clean-label attack only edits pixels of
    images that already carry the adversary label y_adv, so class membership
    (and therefore indices_class) is unaffected.
    """
    pixel = train_pixel.clone()
    pixel[poison_idx] = poisoned_pixel.detach().to(pixel.device)
    return normalizer(pixel)

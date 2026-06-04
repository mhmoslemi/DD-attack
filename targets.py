"""
targets.py

Targeted-attack settings: for each dataset, a dict   seed -> (orig, adv)
where `orig` is the source (true) class of the target point and `adv` is the
adversary class the attack tries to flip it to.

There is no single cross-dataset standard for these class pairs in the
literature; the five CIFAR10 pairs used by RDM-DC (Fig. 2) are specific to that
paper. We therefore define a reproducible protocol and freeze it here so every
dataset / method / ipc uses the *same* attack settings:

  * all five datasets (CIFAR10, SVHN, FashionMNIST, STL10, MNIST) have 10
    classes;
  * the source classes are a fixed random permutation of the label set under a
    single seed, so each class is a source exactly once (for num_targets=10);
  * each target label is drawn uniformly from the remaining 9 classes;
  * the same seed is used for every dataset, so the (orig, adv) index pairs are
    identical across datasets ("identical attack settings across datasets").

`make_targets` is deterministic (numpy default_rng with a fixed seed is stable
across machines), so `TARGETS` is reproducible without hardcoding literals.
Print them with:  python targets.py

The original RDM-DC CIFAR10 pairs are kept as RDMDC_CIFAR10_TARGETS for exact
baseline reproduction if you want them.
"""
import numpy as np

DATASETS = ['CIFAR10', 'SVHN', 'FashionMNIST', 'STL10', 'MNIST']
NUM_CLASSES = {ds: 10 for ds in DATASETS}

# Fixed protocol parameters.
TARGET_SEED = 0
NUM_TARGETS = 10

# Original RDM-DC CIFAR10 settings (5 pairs), for baseline reproduction.
RDMDC_CIFAR10_TARGETS = {0: (7, 5), 1: (0, 9), 2: (5, 9), 3: (4, 9), 4: (2, 8)}


def make_targets(num_classes, num_targets=NUM_TARGETS, seed=TARGET_SEED):
    """Return {i: (orig, adv)} for i in 0..num_targets-1.

    Source classes cover the label set (a shuffled permutation, repeated if
    num_targets > num_classes); each target is uniform over the other classes.
    """
    rng = np.random.default_rng(seed)
    reps = int(np.ceil(num_targets / num_classes))
    origs = np.concatenate([rng.permutation(num_classes) for _ in range(reps)])
    origs = origs[:num_targets]
    targets = {}
    for i, o in enumerate(origs):
        choices = [c for c in range(num_classes) if c != int(o)]
        adv = int(rng.choice(choices))
        targets[i] = (int(o), adv)
    return targets


# Frozen, reproducible target settings for every dataset.
TARGETS = {ds: make_targets(NUM_CLASSES[ds], NUM_TARGETS, seed=TARGET_SEED)
           for ds in DATASETS}


if __name__ == '__main__':
    for ds in DATASETS:
        print(ds)
        for seed, (orig, adv) in TARGETS[ds].items():
            print('  seed %2d : orig=%d -> adv=%d' % (seed, orig, adv))
        print()
    print('RDM-DC CIFAR10 (reference):', RDMDC_CIFAR10_TARGETS)
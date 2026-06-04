# RDM-DC: Poisoning Resilient Dataset Condensation with Robust Distribution Matching

A faithful PyTorch reimplementation of the UAI 2023 paper *RDM-DC: Poisoning
Resilient Dataset Condensation with Robust Distribution Matching* (Zheng & Li),
built on top of the VICO-UoE `DatasetCondensation` code (the slim ConvNet and
DiffAugment pieces are trimmed copies of that repo).

The package implements the paper's two targeted clean-label attacks, the
proposed RDM-DC defense, the empirical robust-aggregation baselines, and the
5 x 5 evaluation protocol, all driven by YAML configs.

## What is implemented

| Paper object | Here |
| --- | --- |
| Algorithm 1, Eq. (4): gradient-matching attack (Witches' Brew) | `attacks.gradient_matching_attack` |
| Algorithm 2, Eq. (5): DM poisoning attack (proposed) | `attacks.dm_poisoning_attack` |
| Algorithm 3: RDM-DC condensation | `condense.distribution_matching` |
| Algorithm 4: mean calibration | `defenses.mean_calibration` |
| Algorithm 5: power method | `defenses.power_method` (literal) and `defenses.top_eigenvector_via_data` (matrix-free, used by default) |
| Empirical baselines: median / trimmed / truncated mean | `defenses.coordinate_median` / `trimmed_mean` / `truncated_mean` |
| "Direct attack" baseline | `attacks.direct_attack` |

## File map

```
rdmdc/
  networks.py        ConvNet (+ embed) and Swish. Trimmed from the original repo.
  utils.py           DiffAugment (A_w), ParamDiffAug, get_network, get_embed,
                     TensorDataset, evaluate_synset. Trimmed from the original repo.
  config.py          Nested YAML -> SimpleNamespace, with dotted CLI overrides.
  data.py            Raw [0,1] datasets, differentiable Normalizer,
                     poisoned-set builder, and a torchvision-free 'debug' set.
  defenses.py        Robust mean estimators + aggregate() dispatcher.
  attacks.py         Gradient matching, DM poisoning, direct attack, pretraining.
  condense.py        Distribution-matching condensation with a robust aggregator.
  evaluate.py        Train a fresh ConvNet on the synthetic set; test acc + ASR.
  run_experiment.py  Driver: seeds x condense runs x eval models; JSON + npz out.
  configs/
    cifar10.yaml       Faithful paper config (CIFAR-10).
    debug.yaml         Tiny CPU smoke test (synthetic data, no download).
    tinyimagenet.yaml  TinyImageNet config (untested here; see its header).
  requirements.txt
```

Only the default ConvNet is included in `networks.py` because RDM-DC uses it
everywhere (both as the random feature extractor Phi_theta and as the
evaluation net). To use other architectures, drop in the full original
`networks.py`; nothing else changes, since everything goes through
`utils.get_network`.

## Install

```bash
pip install -r requirements.txt
```

`scipy` is optional. It is only imported by the legacy (non-DSA) augmentation
path, which this code does not exercise; the import is guarded.

## Quick smoke test (CPU, no dataset download)

```bash
cd rdmdc
python run_experiment.py --config configs/debug.yaml
python run_experiment.py --config configs/debug.yaml --override experiment.attack=gradmatch
python run_experiment.py --config configs/debug.yaml --override experiment.attack=direct experiment.defense=truncated
```

This runs the entire pipeline (craft poisons, condense, train, score ASR) on a
synthetic 4-class 3x16x16 dataset in seconds. The numbers are meaningless; it
exists only to verify wiring and tensor shapes.

## Reproducing the paper (CIFAR-10)

CIFAR-10 downloads automatically through torchvision the first time. Each cell
is one attack x defense combination; override them on the command line.

```bash
# Vanilla DM under each attack (no defense), eps = 64/255  (Table 1)
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=dmpoison  experiment.defense=none
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=gradmatch experiment.defense=none

# Same at eps = 128/255
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=dmpoison experiment.defense=none poison.eps=0.501960784

# Defenses vs the DM poisoning attack  (Table 2)
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=dmpoison experiment.defense=median
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=dmpoison experiment.defense=trimmed
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=dmpoison experiment.defense=truncated
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=dmpoison experiment.defense=rdmdc

# RDM-DC vs gradient matching  (Table 3)
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=gradmatch experiment.defense=rdmdc

# Direct-attack baseline  (Table 5)
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=direct experiment.defense=none
python run_experiment.py --config configs/cifar10.yaml --override experiment.attack=direct experiment.defense=rdmdc
```

Each run prints `TestAcc = a% +/- b%   ASR = c% +/- d%` over the 5 x 5 grid and
writes `result/results_<tag>.json` plus `result/synthetic_<tag>.npz` (the
condensed image/label tensors for every seed and run).

## Config reference

The config is nested. `run_experiment.py` and the modules read it with dotted
attribute access. Overrides use `section.key=value` with YAML-typed values
(`--override poison.eps=0.501960784 condensation.ipc=10 experiment.seed_list=[0,1]`).

- `experiment`: `dataset`, `data_path`, `save_path`, `device` (`auto`/`cpu`/`cuda`),
  `seed_list`, `condense_runs_per_seed`, `eval_models_per_condense`, `attack`
  (`gradmatch`/`dmpoison`/`direct`), `defense`
  (`none`/`rdmdc`/`truncated`/`trimmed`/`median`).
- `model.arch`: `ConvNet`.
- `condensation`: `ipc`, `iterations`, `lr_img`, `momentum_img`, `batch_real`,
  `init` (`noise`/`real`), `dsa`, `dsa_strategy`.
- `poison`: `rate`, `eps` (pixel-space L-inf, in [0,1]), `step_size`, `restarts`,
  `iterations`, `optimizer`, `momentum`, `num_pretrain_models`, `pretrain_epochs`,
  `pretrain_min_epochs`, `round_to_255`.
- `defense`: `eps_per_class` (`auto` or a float), `power_iters`.
- `evaluation`: `epoch_eval_train`, `lr_net`, `batch_train`.

## Faithfulness notes and decisions

These are the places where the paper underspecifies something or where a
deliberate choice was made. They are collected here so the behavior is
auditable.

1. **Noise initialization of synthetic data.** Algorithm 3 initializes the
   synthetic set from Gaussian noise N(0, I), and the paper is explicit that
   real-image initialization makes every method vulnerable. The original DM
   default is real-image init. We default to `init: noise`; `init: real` is
   available for comparison.

2. **Per-class eps for the drop count.** All P poisons are concentrated in one
   class, so the poison fraction *within that class* is the relevant eps. With
   balanced classes, per-class fraction = global `rate` * `num_classes` (0.01 *
   10 = 0.10 for CIFAR-10). The number dropped per class is
   `drop_count = floor(3 * eps_per_class * batch_real) = floor(3 * 0.1 * 256) =
   76`. `eps_per_class: auto` computes this; an explicit float overrides it.
   The same `drop_count` is applied to every class's batch, matching Algorithm
   3 (calibrate each class). This per-class reading follows the paper's Sec. 4.3
   note that when poisons map to a single class one should use the proportion
   relative to that class. The empirical baselines (median/trimmed/truncated)
   use the same `drop_count` for consistency.

3. **Pixel-space perturbations, normalization, then augmentation.** Poisons are
   crafted in [0,1] pixel space under the L-inf budget (so eps in k/255 units is
   exact), projected onto the eps-ball and then the [0,1] box each step, and
   optionally snapped to the 8-bit grid at the end (`round_to_255`). The
   network always sees normalized inputs, and DiffAugment is applied *after*
   normalization, mirroring the condensation pipeline.

4. **Signed Adam.** The optimizer feeds `sign(grad)` to Adam at `step_size`
   (1/255), which the paper notes is equivalent to signed momentum SGD.
   Gradients w.r.t. delta are taken with `torch.autograd.grad` so network
   parameter `.grad` buffers are never touched. Gradient matching uses
   `create_graph=True` on the inner poison gradient because Eq. (4) is
   second-order in delta.

5. **Mean calibration via matrix-free power iteration.** The top eigenvector of
   the representation covariance is obtained by iterating
   `Sigma v = centered^T (centered v) / (N - 1)`, which avoids forming the
   2048 x 2048 covariance (feature dim is 128 * 4 * 4 = 2048 for the CIFAR
   ConvNet). `defenses.power_method` provides the literal Algorithm 5 on an
   explicit covariance for reference and testing.

6. **Gradient matching is a from-scratch reimplementation.** It implements the
   Witches' Brew objective (full-gradient cosine similarity, ensemble over
   pretrained nets, signed-Adam, restarts) directly rather than wrapping the
   official repo. For exact state-of-the-art attack numbers, use the official
   Witches' Brew code; trends and the relative effect of the defense are what
   this reproduces.

7. **Clean-label means in-place replacement.** Both the optimized attacks and
   the direct attack replace the pixels of P images that already carry the
   adversary label y_adv. Labels are unchanged, so `indices_class` is unchanged.

8. **Target image selection.** The exact target images cannot be recovered from
   the paper. For each seed we deterministically pick one test-set image of the
   original class (seeded RNG). Absolute ASR can therefore differ from the
   paper, but the comparative trends across attack/defense hold.

9. **ConvNet specifics and DataParallel.** Default ConvNet is width 128, depth
   3, ReLU, InstanceNorm (implemented as GroupNorm), AvgPool. `utils.get_embed`
   returns `net.module.embed` under `DataParallel` and `net.embed` otherwise, so
   the feature extractor works on single- or multi-GPU.

Additional unpinned hyperparameters: the attack `restarts` (8) and per-restart
`iterations` (250), and the pretraining schedule (`num_pretrain_models=16`
spread over `[5, 40]` epochs), are not specified by the paper. They are set to
reasonable Witches'-Brew-style values and are documented as such; change them in
the config if you want to match a specific setup.

### Best-across-restarts vs the pseudocode

The paper's text says to keep the perturbation with the lowest objective across
restarts, while the Algorithm 1/2 pseudocode literally overwrites the poisons at
the end of each restart (so only the last restart would survive). We implement
the text: across all restarts and steps, the delta achieving the lowest
objective is returned.

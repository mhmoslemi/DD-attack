"""
Evaluation: train a fresh ConvNet from scratch on the synthetic set, then
report (a) test accuracy on the clean test set and (b) attack success rate
(ASR) on the single targeted sample.

ASR is binary per (seed, run) exactly as in the paper: 100 if the trained model
predicts the target image x_t as the adversary label y_adv, else 0. The mean /
std reported by run_experiment over the 5x5 grid then take values like
60.00% +/- 48.99%.
"""
import torch

from utils import evaluate_synset, get_network


def train_and_eval(image_syn, label_syn, x_t_norm, y_adv, testloader, cfg, meta,
                   device, it_eval=0):
    net = get_network(cfg.model.arch, meta.channel, meta.num_classes,
                      meta.im_size, device=device)

    # evaluate_synset reads these off the args/cfg object.
    cfg.device = device
    cfg.dsa = cfg.condensation.dsa
    cfg.dsa_strategy = cfg.condensation.dsa_strategy
    cfg.dsa_param = cfg.dsa_param
    cfg.lr_net = cfg.evaluation.lr_net
    cfg.epoch_eval_train = cfg.evaluation.epoch_eval_train
    cfg.batch_train = cfg.evaluation.batch_train

    net, acc_train, acc_test = evaluate_synset(
        it_eval, net, image_syn, label_syn, testloader, cfg)

    net.eval()
    with torch.no_grad():
        pred = net(x_t_norm).argmax(dim=1).item()
    asr = 100.0 if pred == y_adv else 0.0
    return acc_test, asr

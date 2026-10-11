"""E5: where do the mislabelled training images go, and how far does their damage spread?

  python experiments/e5_noise.py cifar10

Same setup as the E2 width sweep (PCA-64 ResNet features, n=2000, 20% flipped
labels, squared loss), at widths below, at and above the interpolation threshold.

  contagion     : for test images whose nearest training image is mislabelled,
                  how often the tree predicts that wrong label
  label-routed  : fraction of mislabelled images whose leaf is more typical of the
                  wrong label than of the true class (control: clean images with a
                  random fake label)
  concentration : share of mislabelled images sitting in leaves whose noise rate is
                  at least twice the overall rate
Results go to results/e5_<dataset>.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.e2_double_descent import pca_features, train_one  # noqa: E402
from experiments.e3_receptive_fields import hard_nodes  # noqa: E402
from treenet.data import add_label_noise  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
NOISE = 0.2


def routing_stats(leaf, y_true, y_given, flipped, n_leaves, k, rng):
    clean = ~flipped
    counts = np.ones((k, n_leaves))
    np.add.at(counts, (y_true[clean], leaf[clean]), 1)
    p_leaf = counts / counts.sum(1, keepdims=True)
    label_routed = float((p_leaf[y_given[flipped], leaf[flipped]] > p_leaf[y_true[flipped], leaf[flipped]]).mean())
    fake = (y_true[clean] + rng.integers(1, k, clean.sum())) % k
    control = float((p_leaf[fake, leaf[clean]] > p_leaf[y_true[clean], leaf[clean]]).mean())
    rate = np.array([flipped[leaf == l].mean() if (leaf == l).any() else 0 for l in range(n_leaves)])
    noisy_leaf = rate >= 2 * flipped.mean()
    concentration = float(noisy_leaf[leaf[flipped]].mean())
    return {"label_routed": label_routed, "label_routed_control": control, "concentration": concentration,
            "leaf_noise_rate": rate.tolist(), "leaf_size": np.bincount(leaf, minlength=n_leaves).tolist()}


def contagion(model, xtr, ytr_given, flipped, xte, yte):
    with torch.no_grad():
        mu, logits, _ = model(xte)
        pred = (mu[..., None] * logits).sum(1).argmax(1).numpy()
    nn_idx = torch.cdist(xte, xtr).argmin(1).numpy()
    nn_noisy = flipped[nn_idx]
    return {"p_pred_eq_noisy_nn_label": float((pred[nn_noisy] == ytr_given[nn_idx][nn_noisy]).mean()),
            "p_pred_eq_clean_nn_label": float((pred[~nn_noisy] == ytr_given[nn_idx][~nn_noisy]).mean()),
            "test_err_nn_noisy": float((pred[nn_noisy] != yte[nn_noisy]).mean()),
            "test_err_nn_clean": float((pred[~nn_noisy] != yte[~nn_noisy]).mean()),
            "frac_nn_noisy": float(nn_noisy.mean())}


def main():
    dataset = sys.argv[1]
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    rng = np.random.default_rng(0)
    _, _, y_train, _ = pca_features(dataset)
    k = int(y_train.max()) + 1
    rows = []
    for depth, widths in ((2, (4, 16, 64, 256, 1024)), (4, (1, 4, 16, 64))):
        for h in widths:
            model, hist, data = train_one(dataset, "mse", NOISE, 0, depth, h)
            xtr, ytr, xte, yte, _, flipped = data
            y_true = y_train[:len(ytr)].numpy()
            flipped = flipped.numpy()
            assert (add_label_noise(y_true, NOISE, k, 0)[1] == flipped).all()
            with torch.no_grad():
                p = model.router_probs(xtr)
            _, leaf = hard_nodes(p, depth)
            row = {"depth": depth, "h": h, "params": model.n_params(),
                   "train_err": 1 - hist[-1]["train_acc"], "test_err": 1 - hist[-1]["test_acc"]}
            row.update(contagion(model, xtr, ytr.numpy(), flipped, xte, yte.numpy()))
            row.update(routing_stats(leaf.numpy(), y_true, ytr.numpy(), flipped, 2 ** depth, k, rng))
            rows.append(row)
            print(json.dumps({a: b for a, b in row.items() if not isinstance(b, list)}), flush=True)
    with open(f"{ROOT}/results/e5_{dataset}.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

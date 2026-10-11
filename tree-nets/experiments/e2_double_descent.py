"""E2: double descent of the soft tree.

Setup shared by every sweep: ResNet-18 features projected to d_pca dims by PCA
(fit on all training features, no labels), a training subset of n images with
a fraction of labels flipped, full-batch Adam, no regularisation, no early stop.

  python experiments/e2_double_descent.py width   cifar10 mse
  python experiments/e2_double_descent.py depth   cifar10 mse
  python experiments/e2_double_descent.py surface cifar10 mse
  python experiments/e2_double_descent.py samples cifar10 mse
  python experiments/e2_double_descent.py epochs  cifar10 mse
  python experiments/e2_double_descent.py wd      cifar10 mse

Each finished run appends one JSON line to results/e2_<sweep>_<dataset>_<loss>.jsonl;
runs already in the file are skipped, so a sweep can be resumed.
"""
import functools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenet.data import add_label_noise  # noqa: E402
from treenet.features import load_features  # noqa: E402
from treenet.train import fit, make_tree  # noqa: E402

D_PCA, N_TRAIN, STEPS, H_ROUTER = 64, 2000, 4000, 4
WIDTHS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024]


@functools.lru_cache(maxsize=None)
def pca_features(dataset, d_pca=D_PCA, trunk="resnet"):
    d = load_features(dataset, trunk)
    X = d["train"]
    mean = X.mean(0)
    torch.manual_seed(0)
    _, _, V = torch.pca_lowrank(X - mean, q=d_pca, center=False, niter=6)
    return (X - mean) @ V, (d["test"] - mean) @ V, d["y_train"], d["y_test"]


def pca_data(dataset, n, noise, seed):
    ftr, fte, y_train, y_test = pca_features(dataset)
    xtr, xte = ftr[:n], fte
    sd = xtr.std(0)
    n_classes = int(y_train.max()) + 1
    y, flipped = add_label_noise(y_train[:n].numpy(), noise, n_classes, seed)
    return xtr / sd, torch.from_numpy(y), xte / sd, y_test, n_classes, torch.from_numpy(flipped)


def train_one(dataset, loss, noise, seed, depth, h, n=N_TRAIN, steps=STEPS, wd=0.0, eval_every=None):
    torch.manual_seed(seed)
    data = pca_data(dataset, n, noise, seed)
    xtr, ytr, xte, yte, k, _ = data
    model = make_tree(D_PCA, D_PCA, k, depth, H_ROUTER, h, seed)
    hist = fit(model, (xtr, None, ytr), {"test": (xte, None, yte)}, loss_kind=loss,
               epochs=steps, lr=1e-3, wd=wd, eval_every=eval_every or steps)
    return model, hist, data


def run(dataset, loss, noise, seed, depth, h, n=N_TRAIN, steps=STEPS, wd=0.0, eval_every=None):
    model, hist, _ = train_one(dataset, loss, noise, seed, depth, h, n, steps, wd, eval_every)
    last = hist[-1]
    return {"dataset": dataset, "loss": loss, "noise": noise, "seed": seed, "depth": depth, "h": h,
            "n": n, "wd": wd, "params": model.n_params(), "train_err": 1 - last["train_acc"],
            "train_loss": last["train_loss"],
            "test_err": 1 - last["test_acc"], "test_loss": last["test_loss"], "time": last["time"],
            "curve": [(r["epoch"], 1 - r["train_acc"], 1 - r["test_acc"]) for r in hist] if eval_every else None}


def configs(sweep, dataset, loss):
    if sweep == "width":
        if loss != "mse":
            return [dict(noise=nz, seed=0, depth=2, h=h) for nz in (0.2, 0.0) for h in WIDTHS]
        out = []
        for s in ((0, 1, 2) if dataset == "cifar10" else (0, 1)):
            for nz in (0.0, 0.2, 0.1):
                if nz == 0.1 and s > 0:
                    continue
                out += [dict(noise=nz, seed=s, depth=2, h=h) for h in WIDTHS if s == 0 or h <= 512]
        return out
    if sweep == "width6k":
        return [dict(noise=0.2, seed=0, depth=2, h=h, n=6000) for h in (1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512)]
    if sweep == "depth":
        return [dict(noise=0.2, seed=s, depth=dp, h=16) for s in (0, 1, 2) for dp in range(0, 8)]
    if sweep == "surface":
        return [dict(noise=0.2, seed=0, depth=dp, h=h) for dp in range(0, 5) for h in (2, 4, 8, 16, 32, 64, 128, 256)]
    if sweep == "samples":
        return [dict(noise=0.2, seed=s, depth=2, h=64, n=n) for s in (0, 1)
                for n in (250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000)]
    if sweep == "epochs":
        return [dict(noise=0.2, seed=0, depth=2, h=h, steps=10000, eval_every=50) for h in (64, 512)]
    if sweep == "wd":
        return [dict(noise=0.2, seed=0, depth=2, h=h, wd=wd) for wd in (1e-3, 1e-2, 1e-1) for h in WIDTHS[::2] if h <= 512]
    raise ValueError(sweep)


def main():
    sweep, dataset, loss = sys.argv[1:4]
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"e2_{sweep}_{dataset}_{loss}.jsonl")
    keys = ("noise", "seed", "depth", "h", "n", "wd")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add(tuple(r[k] for k in keys))
    for cfg in configs(sweep, dataset, loss):
        full = {"n": N_TRAIN, "wd": 0.0, **cfg}
        if tuple(full[k] for k in keys) in done:
            continue
        r = run(dataset, loss, **cfg)
        print(json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

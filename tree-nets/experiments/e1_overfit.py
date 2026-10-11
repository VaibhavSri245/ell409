"""E1: train trees until they fit the full training set (target: 100% train accuracy).

  python experiments/e1_overfit.py <dataset> <trunk> <mode>

trunk: random | cnn | resnet
mode : shared  - routers and leaves read the same (full) feature vector
       matched - routers read shallow trunk taps, leaves read deep taps

No augmentation, dropout or weight decay. The model is saved to
results/models/ for the receptive-field (E3) and compression (E4) experiments.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenet.features import CNN_TAPS, RESNET_TAPS, load_features  # noqa: E402
from treenet.train import evaluate, fit, make_tree  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DEPTH, H_ROUTER, H_LEAF = 3, 32, 128


def split_features(dataset, trunk, mode):
    """Returns dict split -> (x_router, x_leaf) and labels."""
    d = load_features(dataset, trunk)
    if mode == "shared":
        xs = {s: (d[s], d[s]) for s in ("train", "val", "test")}
    else:
        taps = {"resnet": (RESNET_TAPS["l1"].start, RESNET_TAPS["l2"].stop, RESNET_TAPS["l3"].start),
                "cnn": (CNN_TAPS["s2"].start, CNN_TAPS["s2"].stop, CNN_TAPS["s4"].start)}[trunk]
        a, b, c = taps
        xs = {s: (d[s][:, a:b], d[s][:, c:]) for s in ("train", "val", "test")}
    return xs, d


def main():
    dataset, trunk, mode = sys.argv[1:4]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    xs, d = split_features(dataset, trunk, mode)
    k = int(d["y_train"].max()) + 1
    model = make_tree(xs["train"][0].shape[1], xs["train"][1].shape[1], k, DEPTH, H_ROUTER, H_LEAF)
    train = (*xs["train"], d["y_train"])
    evals = {"test": (*xs["test"], d["y_test"])}
    tag = f"{dataset}_{trunk}_{mode}"
    hist = fit(model, train, evals, epochs=epochs, bs=256, lr=1e-3, balance=0.01, eval_every=5,
               stop_at_train_acc=1.0, log=lambda r: print(tag, json.dumps(r), flush=True))
    hard_test, _ = evaluate(model, *xs["test"], d["y_test"], hard=True)
    hard_train, _ = evaluate(model, *xs["train"], d["y_train"], hard=True)
    last = hist[-1]
    row = {"dataset": dataset, "trunk": trunk, "mode": mode, "depth": DEPTH, "h_router": H_ROUTER, "h_leaf": H_LEAF,
           "params": model.n_params(), "active_params": model.active_params(), "epochs": last["epoch"],
           "train_acc": last["train_acc"], "test_acc": last["test_acc"], "hard_train_acc": hard_train,
           "hard_test_acc": hard_test, "time": last["time"],
           "curve": [(r["epoch"], r["train_acc"], r["test_acc"]) for r in hist]}
    os.makedirs(f"{RESULTS}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{RESULTS}/models/e1_{tag}.pt")
    with open(f"{RESULTS}/e1_overfit.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "curve"}))


if __name__ == "__main__":
    main()

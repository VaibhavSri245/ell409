"""E4: how small can the tree get?

  python experiments/e4_compress.py <dataset>

Starts from the E1 tree (ResNet features, shared mode) and tries:
  hard routing          only depth routers + 1 leaf run per image
  subtree collapse      greedily force routers to one side (cost-complexity style)
  magnitude pruning     global unstructured, prune -> fine-tune, 50% ... 99%
  hidden-unit pruning   keep the k strongest hidden units of each leaf, fine-tune
  int8 weights          per-tensor symmetric quantisation
  small trees           trained from scratch on labels vs distilled from the big tree
Results go to results/e4_<dataset>.json.
"""
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenet.features import load_features  # noqa: E402
from treenet.train import evaluate, fit, make_tree  # noqa: E402
from experiments.e1_overfit import DEPTH, H_LEAF, H_ROUTER  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def acc(model, x, y, hard=True):
    return evaluate(model, x, None, y, hard=hard)[0]


def weights(model):
    return [p for n, p in model.named_parameters() if n.split(".")[-1].startswith("w")]


def nonzero(model):
    return sum(int((p != 0).sum()) for p in model.parameters())


def train_masked(model, masks, data, epochs, lr=3e-4):
    """Fine-tune while keeping pruned weights at zero."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = data
    for _ in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 256):
            idx = perm[i:i + 256]
            mu, logits, _ = model(x[idx])
            logp = logits.log_softmax(-1)[torch.arange(len(idx)), :, y[idx]]
            loss = -torch.logsumexp(torch.log(mu + 1e-12) + logp, 1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p, m in zip(weights(model), masks):
                    p.mul_(m)


def magnitude_prune(base, data, test, sparsities, ft_epochs=3):
    model = copy.deepcopy(base)
    rows = []
    for s in sparsities:
        flat = torch.cat([p.detach().abs().flatten() for p in weights(model)])
        thr = flat.kthvalue(max(1, int(s * len(flat)))).values
        masks = [(p.detach().abs() > thr).float() for p in weights(model)]
        with torch.no_grad():
            for p, m in zip(weights(model), masks):
                p.mul_(m)
        before = acc(model, *test)
        train_masked(model, masks, data, ft_epochs)
        rows.append({"sparsity": s, "nonzero": nonzero(model), "acc_before_ft": before, "test_acc": acc(model, *test),
                     "active_nonzero": active_nonzero(model)})
        print("magnitude", rows[-1], flush=True)
    return rows


def active_nonzero(model):
    """Average non-zero weights on one root-to-leaf path (hard routing)."""
    per_r = sum(int((t != 0).sum()) for t in model.routers.parameters()) / (2 ** model.depth - 1)
    per_l = sum(int((t != 0).sum()) for t in model.leaves.parameters()) / 2 ** model.depth
    return int(model.depth * per_r + per_l)


def unit_prune(base, data, test, keeps, ft_epochs=3):
    rows = []
    for k in keeps:
        model = copy.deepcopy(base)
        L = model.leaves
        score = L.w1.detach().norm(dim=1) * L.w2.detach().norm(dim=2)  # (n_leaves, h)
        keep = score.argsort(1, descending=True)[:, :k]
        m1 = torch.zeros_like(L.w1)
        m2 = torch.zeros_like(L.w2)
        for l in range(m1.shape[0]):
            m1[l][:, keep[l]] = 1
            m2[l][keep[l]] = 1
        masks = [m1 if p is L.w1 else m2 if p is L.w2 else torch.ones_like(p) for p in weights(model)]
        with torch.no_grad():
            for p, m in zip(weights(model), masks):
                p.mul_(m)
        train_masked(model, masks, data, ft_epochs)
        n_leaf = model.n_leaves * (L.w1.shape[1] * k + k + k * L.w2.shape[2] + L.w2.shape[2])
        n_router = sum(p.numel() for p in model.routers.parameters())
        rows.append({"hidden_kept": k, "params": n_leaf + n_router, "test_acc": acc(model, *test)})
        print("units", rows[-1], flush=True)
    return rows


def collapse_path(base, val, test):
    """Greedy subtree collapse. Forcing router i to one side removes the other subtree."""
    model = copy.deepcopy(base)
    R = model.routers
    n_r = 2 ** model.depth - 1
    forced = {}

    def children(i):
        return [c for c in (2 * i + 1, 2 * i + 2)]

    def reachable():
        nodes, stack = set(), [0]
        while stack:
            i = stack.pop()
            nodes.add(i)
            if i < n_r:
                stack += [forced[i]] if i in forced else children(i)
        return nodes

    def size():
        nodes = reachable()
        per_r = sum(t[0].numel() for t in R.parameters())
        per_l = sum(t[0].numel() for t in model.leaves.parameters())
        n_routers = sum(1 for i in nodes if i < n_r and i not in forced)
        n_leaves = sum(1 for i in nodes if i >= n_r)
        return n_routers * per_r + n_leaves * per_l, n_leaves

    def apply(fmap):
        with torch.no_grad():
            R.load_state_dict(base.routers.state_dict())
            for i, c in fmap.items():
                R.w2[i].zero_()
                R.b2[i].fill_(50.0 if c == 2 * i + 1 else -50.0)

    path = []
    p0, l0 = size()
    path.append({"params": p0, "leaves": l0, "val_acc": acc(model, *val), "test_acc": acc(model, *test)})
    while True:
        cands = [i for i in reachable() if i < n_r and i not in forced]
        if not cands:
            break
        best = None
        for i in cands:
            for c in children(i):
                trial = {**forced, i: c}
                apply(trial)
                a = acc(model, *val)
                if best is None or a > best[0]:
                    best = (a, i, c)
        forced[best[1]] = best[2]
        apply(forced)
        p, l = size()
        path.append({"params": p, "leaves": l, "val_acc": best[0], "test_acc": acc(model, *test)})
        print("collapse", path[-1], flush=True)
    return path


def quantise(base, test):
    model = copy.deepcopy(base)
    with torch.no_grad():
        for p in model.parameters():
            scale = p.abs().max() / 127
            p.copy_((p / scale).round().clamp(-127, 127) * scale)
    return {"bits": 8, "test_acc": acc(model, *test), "bytes": sum(p.numel() for p in model.parameters())}


def small_trees(teacher, data, test, sizes, epochs=60):
    x, y = data
    with torch.no_grad():
        mu, logits, _ = teacher(x)
        soft = (mu[..., None] * (logits / 2.0).softmax(-1)).sum(1)
    rows = []
    for depth, h_r, h_l in sizes:
        for mode in ("labels", "distill"):
            m = make_tree(x.shape[1], x.shape[1], int(y.max()) + 1, depth, h_r, h_l)
            if mode == "labels":
                hist = fit(m, (x, None, y), epochs=epochs, bs=256, lr=1e-3, balance=0.01, eval_every=epochs,
                           stop_at_train_acc=1.0)
                train_acc = hist[-1]["train_acc"]
            else:
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                for _ in range(epochs):
                    perm = torch.randperm(len(y))
                    for i in range(0, len(y), 256):
                        idx = perm[i:i + 256]
                        mu, lg, _ = m(x[idx])
                        q = (mu[..., None] * (lg / 2.0).softmax(-1)).sum(1)
                        loss = F.kl_div(torch.log(q + 1e-9), soft[idx], reduction="batchmean")
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                train_acc = acc(m, x, y, hard=False)
            rows.append({"depth": depth, "h_router": h_r, "h_leaf": h_l, "mode": mode, "params": m.n_params(),
                         "active_params": m.active_params(), "train_acc": train_acc,
                         "test_acc_soft": acc(m, *test, hard=False), "test_acc_hard": acc(m, *test)})
            print("small", rows[-1], flush=True)
    return rows


def main():
    dataset = sys.argv[1]
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    torch.manual_seed(0)
    d = load_features(dataset, "resnet")
    k = int(d["y_train"].max()) + 1
    base = make_tree(960, 960, k, DEPTH, H_ROUTER, H_LEAF)
    base.load_state_dict(torch.load(f"{ROOT}/results/models/e1_{dataset}_resnet_shared.pt"))
    train, val, test = (d["train"], d["y_train"]), (d["val"], d["y_val"]), (d["test"], d["y_test"])
    out = {"base": {"params": base.n_params(), "active_params": base.active_params(),
                    "test_acc_soft": acc(base, *test, hard=False), "test_acc_hard": acc(base, *test)}}
    print(out, flush=True)
    out["int8"] = quantise(base, test)
    out["collapse"] = collapse_path(base, val, test)
    out["magnitude"] = magnitude_prune(base, train, test, [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995])
    out["units"] = unit_prune(base, train, test, [64, 32, 16, 8, 4, 2])
    out["small"] = small_trees(base, train, test, [(0, 0, 0), (1, 4, 8), (2, 4, 8), (2, 8, 16), (3, 8, 16), (3, 16, 32)])
    with open(f"{ROOT}/results/e4_{dataset}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

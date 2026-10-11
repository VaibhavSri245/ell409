"""E3: what does each node of a trained tree look at?

  python experiments/e3_receptive_fields.py <dataset> <trunk>     # uses the E1 shared-feature tree
  python experiments/e3_receptive_fields.py <dataset> pixel       # trains and plots a pixel tree

For every router: the classes it sends left/right, its most confident images on
each side, the input-gradient saliency (effective receptive field) averaged over
the images that reach it, and an image synthesised to push it hardest each way.
For every leaf: the classes it receives and the saliency of its decision.
Numbers go to results/e3_<dataset>_<trunk>.json, figures to figures/.
"""
import json
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenet.data import classes, load  # noqa: E402
from treenet.features import build_trunk, load_features, to_tensor  # noqa: E402
from treenet.train import fit, make_tree  # noqa: E402
from experiments.e1_overfit import DEPTH, H_LEAF, H_ROUTER  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
VEHICLES = {0, 1, 8, 9}


def hard_nodes(p, depth):
    """(N, depth) router indices visited under hard routing, and the (N,) leaf index."""
    node = torch.zeros(p.shape[0], dtype=torch.long)
    visited = []
    for _ in range(depth):
        visited.append(node)
        left = p.gather(1, node[:, None])[:, 0] >= 0.5
        node = 2 * node + 1 + (~left).long()
    return torch.stack(visited, 1), node - (2 ** depth - 1)


def nmi(a, b):
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(a, b))


def radius(sal):
    """Saliency-weighted mean distance from the image centre, as a fraction of the half-width."""
    h, w = sal.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2) / (h / 2)
    return float((sal * r).sum() / sal.sum())


class Pipeline(torch.nn.Module):
    """image in [0,1] -> trunk features -> standardise -> tree outputs."""

    def __init__(self, trunk, mu, sd, tree):
        super().__init__()
        self.trunk, self.mu, self.sd, self.tree = trunk, mu, sd, tree

    def forward(self, x):
        z = (self.trunk(x) - self.mu) / self.sd
        return self.tree.routers(z)[..., 0].T, self.tree.leaves(z).transpose(0, 1)


def saliency(fn, x):
    """Mean over images of the channel-summed |d fn / d x|, normalised to sum 1."""
    x = x.clone().requires_grad_(True)
    fn(x).sum().backward()
    s = x.grad.abs().sum(1).mean(0).numpy()
    return s / s.sum()


def act_max(fn, shape, sign, steps=150):
    """Image maximising sign * fn, with jitter and a total-variation penalty."""
    torch.manual_seed(0)
    v = torch.zeros(1, *shape, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.05)
    for _ in range(steps):
        x = torch.sigmoid(v)
        dx, dy = np.random.randint(-2, 3, 2)
        xj = torch.roll(x, (int(dy), int(dx)), (2, 3))
        tv = (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        loss = -sign * fn(xj).sum() + 2.0 * tv
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.sigmoid(v)[0].detach().permute(1, 2, 0).numpy()


def analyse(dataset, trunk_kind):
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    names = classes(dataset)
    k = len(names)
    imgs = load(dataset)
    feats = load_features(dataset, trunk_kind)
    d_in = feats["train"].shape[1]
    tree = make_tree(d_in, d_in, k, DEPTH, H_ROUTER, H_LEAF)
    tree.load_state_dict(torch.load(f"{ROOT}/results/models/e1_{dataset}_{trunk_kind}_shared.pt"))
    tree.eval()
    pipe = Pipeline(build_trunk(trunk_kind, dataset), feats["mu"], feats["sd"], tree).eval()
    for p in pipe.parameters():
        p.requires_grad_(False)

    x_test = to_tensor(imgs["x_test"])
    y = feats["y_test"].numpy()
    with torch.no_grad():
        p = tree.router_probs(feats["test"])
        leaf_logits = tree.leaves(feats["test"]).transpose(0, 1)
    nodes, leaf = hard_nodes(p, DEPTH)
    rng = np.random.default_rng(0)
    out = {"dataset": dataset, "trunk": trunk_kind, "routers": [], "leaves": [],
           "nmi_leaf_class": nmi(leaf.numpy(), y)}
    if dataset == "cifar10":
        out["nmi_root_vehicle"] = nmi((p[:, 0] >= 0.5).numpy(), np.isin(y, list(VEHICLES)))

    n_routers = 2 ** DEPTH - 1
    fig, axes = plt.subplots(n_routers, 9, figsize=(13, 1.6 * n_routers))
    for i in range(n_routers):
        reach = (nodes == i).any(1).numpy()
        idx = np.where(reach)[0]
        left = (p[idx, i] >= 0.5).numpy()
        hist_l = np.bincount(y[idx[left]], minlength=k)
        hist_r = np.bincount(y[idx[~left]], minlength=k)
        sub = torch.from_numpy(rng.choice(idx, min(64, len(idx)), replace=False))
        mean_sal = saliency(lambda x: pipe(x)[0][:, i], x_test[sub])
        order = idx[np.argsort(-p[idx, i].numpy())]
        am_l = act_max(lambda x: pipe(x)[0][:, i], x_test.shape[1:], +1)
        am_r = act_max(lambda x: pipe(x)[0][:, i], x_test.shape[1:], -1)
        out["routers"].append({
            "node": i, "level": int(np.floor(np.log2(i + 1))), "n_reach": int(reach.sum()),
            "left_frac": float(left.mean()),
            "left_top": [names[c] for c in np.argsort(-hist_l)[:3]],
            "right_top": [names[c] for c in np.argsort(-hist_r)[:3]],
            "hist_left": hist_l.tolist(), "hist_right": hist_r.tolist(),
            "nmi_split_class": nmi(left, y[idx]), "saliency_radius": radius(mean_sal)})
        row = axes[i]
        row[0].imshow(gaussian_filter(mean_sal, 1), cmap="inferno")
        row[0].set_title(f"node {i}: saliency", fontsize=7)
        row[1].imshow(am_l)
        row[1].set_title("synth. left", fontsize=7)
        row[2].imshow(am_r)
        row[2].set_title("synth. right", fontsize=7)
        for j in range(3):
            row[3 + j].imshow(imgs["x_test"][order[j]])
            row[3 + j].set_title(f"L: {names[y[order[j]]]}", fontsize=7)
            row[6 + j].imshow(imgs["x_test"][order[-1 - j]])
            row[6 + j].set_title(f"R: {names[y[order[-1 - j]]]}", fontsize=7)
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{ROOT}/figures/e3_routers_{dataset}_{trunk_kind}.png", dpi=130)
    plt.close(fig)

    n_leaves = 2 ** DEPTH
    fig, axes = plt.subplots(n_leaves, 6, figsize=(9, 1.5 * n_leaves))
    for l in range(n_leaves):
        idx = np.where((leaf == l).numpy())[0]
        hist = np.bincount(y[idx], minlength=k)
        pred = leaf_logits[idx, l].argmax(1).numpy() if len(idx) else np.array([])
        entry = {"leaf": l, "n": int(len(idx)), "top": [names[c] for c in np.argsort(-hist)[:3]],
                 "hist": hist.tolist(), "acc": float((pred == y[idx]).mean()) if len(idx) else None}
        row = axes[l]
        row[0].bar(range(k), hist, color="#4C72B0")
        row[0].set_xticks(range(k))
        row[0].set_xticklabels([c[:3] for c in names], fontsize=5, rotation=90)
        row[0].set_title(f"leaf {l} (n={len(idx)})", fontsize=7)
        if len(idx):
            sub = torch.from_numpy(rng.choice(idx, min(64, len(idx)), replace=False))
            cls = torch.from_numpy(y[sub.numpy()])
            mean_sal = saliency(lambda x: pipe(x)[1][:, l].gather(1, cls[:, None])[:, 0], x_test[sub])
            entry["saliency_radius"] = radius(mean_sal)
            row[1].imshow(gaussian_filter(mean_sal, 1), cmap="inferno")
            row[1].set_title("saliency", fontsize=7)
            for j, s in enumerate(rng.choice(idx, min(4, len(idx)), replace=False)):
                row[2 + j].imshow(imgs["x_test"][s])
                row[2 + j].set_title(names[y[s]], fontsize=7)
        for a in row[1:]:
            a.axis("off")
        out["leaves"].append(entry)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/figures/e3_leaves_{dataset}_{trunk_kind}.png", dpi=130)
    plt.close(fig)
    with open(f"{ROOT}/results/e3_{dataset}_{trunk_kind}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("routers", "leaves")}))
    for r in out["routers"]:
        print(r["node"], r["n_reach"], r["left_top"], "|", r["right_top"], round(r["saliency_radius"], 3))


def pixel_tree(dataset):
    """Tree with linear routers on raw pixels: each router's weight vector is its receptive field."""
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    names = classes(dataset)
    imgs = load(dataset)
    x = {s: to_tensor(imgs[f"x_{s}"]) for s in ("train", "test")}
    if x["train"].shape[-1] > 32:
        x = {s: F.avg_pool2d(v, 2) for s, v in x.items()}
    shape = x["train"].shape[1:]
    mu, sd = x["train"].mean(0), x["train"].std(0) + 1e-6
    z = {s: ((v - mu) / sd).flatten(1) for s, v in x.items()}
    y = {s: torch.from_numpy(imgs[f"y_{s}"]) for s in ("train", "test")}
    tree = make_tree(z["train"].shape[1], z["train"].shape[1], len(names), DEPTH, 0, 64)
    hist = fit(tree, (z["train"], None, y["train"]), {"test": (z["test"], None, y["test"])},
               epochs=30, bs=256, lr=3e-4, balance=0.01, eval_every=10, log=print)
    with torch.no_grad():
        p = tree.router_probs(z["test"])
    nodes, leaf = hard_nodes(p, DEPTH)
    torch.save(tree.state_dict(), f"{ROOT}/results/models/e3_pixeltree_{dataset}.pt")
    w = tree.routers.w2[:, :, 0].detach().view(-1, *shape)
    fig, axes = plt.subplots(2, 2 ** DEPTH - 1, figsize=(2 * (2 ** DEPTH - 1), 4.4))
    info = []
    for i in range(2 ** DEPTH - 1):
        img = w[i].permute(1, 2, 0).numpy()
        img = 0.5 + img / (2 * np.abs(img).max())
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"node {i} weights", fontsize=8)
        smooth = gaussian_filter(img, (1.5, 1.5, 0))
        axes[1, i].imshow(0.5 + (smooth - 0.5) / (2 * np.abs(smooth - 0.5).max()))
        axes[1, i].set_title("low-pass view", fontsize=8)
        reach = (nodes == i).any(1)
        left = p[reach, i] >= 0.5
        yy = y["test"][reach].numpy()
        hl = np.bincount(yy[left.numpy()], minlength=len(names))
        hr = np.bincount(yy[~left.numpy()], minlength=len(names))
        info.append({"node": i, "left_top": [names[c] for c in np.argsort(-hl)[:3]],
                     "right_top": [names[c] for c in np.argsort(-hr)[:3]],
                     "radius": radius(gaussian_filter(np.abs(w[i]).sum(0).numpy(), 1.5))})
    for a in axes.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{ROOT}/figures/e3_pixeltree_{dataset}.png", dpi=130)
    plt.close(fig)
    out = {"train_acc": hist[-1]["train_acc"], "test_acc": hist[-1]["test_acc"], "routers": info}
    with open(f"{ROOT}/results/e3_{dataset}_pixel.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    ds, kind = sys.argv[1:3]
    if kind == "pixel":
        pixel_tree(ds)
    else:
        analyse(ds, kind)

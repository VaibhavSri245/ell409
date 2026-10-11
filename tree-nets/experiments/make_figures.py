"""Turn results/*.json(l) into the report figures in figures/.

  python experiments/make_figures.py
"""
import json
import os
from collections import defaultdict

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [(BLUE, "o"), (ORANGE, "s"), (AQUA, "^")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 4, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight"})


def jsonl(name):
    path = f"{RES}/{name}"
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def mean_band(rows, xkey, ykey, group):
    """Group rows, then average ykey over seeds at each xkey. Returns {g: (x, mean, min, max)}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r[group]][r[xkey]].append(r[ykey])
    out = {}
    for g, d in acc.items():
        xs = sorted(d)
        out[g] = (np.array(xs), np.array([np.mean(d[x]) for x in xs]),
                  np.array([np.min(d[x]) for x in xs]), np.array([np.max(d[x]) for x in xs]))
    return out


def interpolation_point(rows, tol=5e-3):
    """Smallest parameter count whose mean final training loss is below tol (noisiest setting)."""
    noise = max(r["noise"] for r in rows)
    x, m, _, _ = mean_band([r for r in rows if "train_loss" in r and r["noise"] == noise], "params", "train_loss", "noise")[noise]
    return x[np.argmax(m < tol)]


def width_figure(dataset, loss, k, tag):
    rows = jsonl(f"e2_width_{dataset}_{loss}.jsonl")
    if not rows:
        return
    p_star = interpolation_point(rows) if loss == "mse" else None
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    panels = ((axes[0], "test_err", "Test error"), (axes[1], "train_err", "Train error"),
              (axes[2], "train_loss" if loss == "mse" else "test_loss",
               "Train loss (squared)" if loss == "mse" else "Test loss (NLL)"))
    for ax, key, title in panels:
        bands = mean_band([r for r in rows if key in r], "params", key, "noise")
        for (color, marker), noise in zip(SERIES, sorted(bands)):
            x, m, lo, hi = bands[noise]
            if key == "train_loss":
                m, lo, hi = (np.maximum(v, 1e-5) for v in (m, lo, hi))
            ax.plot(x, m, color=color, marker=marker, label=f"{int(noise * 100)}% label noise")
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        if p_star is not None:
            ax.axvline(p_star, color=INK2, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        if key == "train_loss":
            ax.set_yscale("log")
        ax.set_xlabel("parameters in the tree")
        ax.set_title(title, fontsize=9, color=INK)
    extra = jsonl(f"e2_width6k_{dataset}_{loss}.jsonl")
    if extra:
        extra.sort(key=lambda r: r["params"])
        axes[0].plot([r["params"] for r in extra], [r["test_err"] for r in extra], color=INK2, marker="D",
                     linestyle="--", linewidth=1.5, label="20% noise, n = 6000")
        axes[1].plot([r["params"] for r in extra], [r["train_err"] for r in extra], color=INK2, marker="D",
                     linestyle="--", linewidth=1.5, label="20% noise, n = 6000")
    if p_star is not None:
        axes[0].text(p_star * 1.08, axes[0].get_ylim()[1], f"train loss < 0.005\nfrom P = {p_star:,}",
                     color=INK2, fontsize=7, va="top")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.savefig(f"{FIG}/{tag}.png")
    plt.close(fig)


def depth_figure():
    rows = jsonl("e2_depth_cifar10_mse.jsonl")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for key, (color, marker), label in (("test_err", SERIES[0], "test"), ("train_err", SERIES[1], "train")):
        x, m, lo, hi = mean_band(rows, "depth", key, "noise")[0.2]
        ax.plot(x, m, color=color, marker=marker, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    params = {r["depth"]: r["params"] for r in rows}
    ax.set_xticks(sorted(params))
    ax.set_xticklabels([f"{d}\n{params[d] / 1000:.0f}k" for d in sorted(params)], fontsize=7)
    ax.set_xlabel("depth (leaves = 2^depth)  /  parameters")
    ax.set_ylabel("error")
    ax.legend(fontsize=7)
    fig.savefig(f"{FIG}/e2_depth.png")
    plt.close(fig)


def surface_figure():
    rows = jsonl("e2_surface_cifar10_mse.jsonl")
    if not rows:
        return
    depths = sorted({r["depth"] for r in rows})
    widths = sorted({r["h"] for r in rows})
    te = np.full((len(depths), len(widths)), np.nan)
    tr = np.full_like(te, np.nan)
    pp = np.full_like(te, np.nan)
    for r in rows:
        i, j = depths.index(r["depth"]), widths.index(r["h"])
        te[i, j], tr[i, j], pp[i, j] = r["test_err"], r["train_loss"], r["params"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(te, cmap="Blues", origin="lower", aspect="auto")
    for i in range(len(depths)):
        for j in range(len(widths)):
            if not np.isnan(te[i, j]):
                ax.text(j, i, f"{te[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if te[i, j] > np.nanmean(te) else INK)
    ax.contour(np.arange(len(widths)), np.arange(len(depths)), tr, levels=[0.005], colors=[ORANGE], linewidths=2)
    ax.contour(np.arange(len(widths)), np.arange(len(depths)), np.log10(pp), levels=[np.log10(20000)],
               colors=[INK2], linewidths=1, linestyles="--")
    ax.plot([], [], color=ORANGE, linewidth=2, label="train loss = 0.005")
    ax.plot([], [], color=INK2, linewidth=1, linestyle="--", label="P = nK = 20,000")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel("hidden width of each leaf")
    ax.set_ylabel("depth")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="test error")
    fig.savefig(f"{FIG}/e2_surface.png")
    plt.close(fig)


def samples_figure():
    rows = jsonl("e2_samples_cifar10_mse.jsonl")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for key, (color, marker), label in (("test_err", SERIES[0], "test"), ("train_err", SERIES[1], "train")):
        x, m, lo, hi = mean_band(rows, "n", key, "noise")[0.2]
        ax.plot(x, m, color=color, marker=marker, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    p = rows[0]["params"]
    ax.axvline(p / 10, color=INK2, linestyle="--", linewidth=1)
    ax.text(p / 10 * 1.05, 0.2, f"n = P/K = {p // 10:,}", color=INK2, fontsize=7)
    ax.set_xscale("log")
    ticks = [250, 500, 1000, 2000, 4000, 8000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("training images n (model fixed, P = %s)" % f"{p:,}")
    ax.set_ylabel("error")
    ax.legend(fontsize=7)
    fig.savefig(f"{FIG}/e2_samples.png")
    plt.close(fig)


def epochs_figure():
    rows = jsonl("e2_epochs_cifar10_mse.jsonl")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3))
    for (color, marker), r in zip(SERIES, sorted(rows, key=lambda r: r["h"])):
        c = np.array(r["curve"])
        ax.plot(c[:, 0], c[:, 2], color=color, label=f"test, width {r['h']} ({r['params'] / 1000:.0f}k params)")
        ax.plot(c[:, 0], c[:, 1], color=color, linestyle=":", linewidth=1.5, label=f"train, width {r['h']}")
    ax.set_xscale("log")
    ax.set_xlabel("full-batch Adam steps")
    ax.set_ylabel("error")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(f"{FIG}/e2_epochs.png")
    plt.close(fig)


def wd_figure():
    rows = jsonl("e2_wd_cifar10_mse.jsonl") + [r for r in jsonl("e2_width_cifar10_mse.jsonl")
                                               if r["noise"] == 0.2 and r["seed"] == 0]
    if not any(r["wd"] > 0 for r in rows):
        return
    fig, ax = plt.subplots(figsize=(4.6, 3))
    wds = sorted({r["wd"] for r in rows})
    palette = [(INK2, "o")] + SERIES
    for (color, marker), wd in zip(palette, wds):
        sel = sorted((r for r in rows if r["wd"] == wd), key=lambda r: r["params"])
        ax.plot([r["params"] for r in sel], [r["test_err"] for r in sel], color=color, marker=marker,
                label="no weight decay" if wd == 0 else f"weight decay {wd:g}")
    ax.set_xscale("log")
    ax.set_xlabel("parameters in the tree")
    ax.set_ylabel("test error (20% label noise)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(f"{FIG}/e2_wd.png")
    plt.close(fig)


def e1_figure():
    rows = jsonl("e1_overfit.jsonl")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, ds in zip(axes, ("cifar10", "catsdogs")):
        sel = [r for r in rows if r["dataset"] == ds and r["mode"] == "shared"]
        for (color, marker), r in zip(SERIES, sorted(sel, key=lambda r: r["trunk"])):
            c = np.array(r["curve"])
            ax.plot(c[:, 0], c[:, 1], color=color, linestyle=":", linewidth=1.5)
            ax.plot(c[:, 0], c[:, 2], color=color, label=f"{r['trunk']} features")
        ax.set_title(f"{ds}: train (dotted) and test (solid) accuracy", fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(["5", "10", "20", "50", "100"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7)
    fig.savefig(f"{FIG}/e1_curves.png")
    plt.close(fig)


def e4_figure(dataset):
    path = f"{RES}/e4_{dataset}.json"
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    ax = axes[0]
    mag = d["magnitude"]
    ax.plot([r["nonzero"] for r in mag], [r["test_acc"] for r in mag], color=BLUE, marker="o",
            label="magnitude pruning (stored)")
    ax.plot([r["active_nonzero"] for r in mag], [r["test_acc"] for r in mag], color=BLUE, marker="o",
            linestyle=":", linewidth=1.5, label="magnitude pruning (per image)")
    un = d["units"]
    ax.plot([r["params"] for r in un], [r["test_acc"] for r in un], color=ORANGE, marker="s",
            label="leaf hidden-unit pruning")
    co = d["collapse"]
    ax.plot([r["params"] for r in co], [r["test_acc"] for r in co], color=AQUA, marker="^",
            label="subtree collapse")
    ax.scatter([d["base"]["params"]], [d["base"]["test_acc_hard"]], color=INK, zorder=5, s=25)
    ax.annotate("E1 tree", (d["base"]["params"], d["base"]["test_acc_hard"]), fontsize=7,
                xytext=(-12, 6), textcoords="offset points")
    lin = next(r for r in d["small"] if r["depth"] == 0 and r["mode"] == "labels")
    ax.set_xscale("log")
    ax.axhline(lin["test_acc_hard"], color=INK2, linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[0] * 1.15, lin["test_acc_hard"] + 0.001,
            f"linear classifier, {lin['params']:,} params", color=INK2, fontsize=7, va="bottom")
    ax.set_xlabel("non-zero parameters")
    ax.set_ylabel("test accuracy (hard routing)")
    ax.set_title("Compressing the trained tree", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[1]
    for (color, marker), mode in zip(SERIES, ("labels", "distill")):
        sel = sorted((r for r in d["small"] if r["mode"] == mode), key=lambda r: r["params"])
        ax.plot([r["params"] for r in sel], [r["test_acc_hard"] for r in sel], color=color, marker=marker,
                label="trained on labels" if mode == "labels" else "distilled from E1 tree")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_title("Small trees from scratch", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(f"{FIG}/e4_{dataset}.png")
    plt.close(fig)


def e5_figure():
    path = f"{RES}/e5_cifar10.json"
    if not os.path.exists(path):
        return
    rows = [r for r in json.load(open(path)) if r["depth"] == 2]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = [r["params"] for r in rows]
    series = (("test error, nearest training image clean", [r["test_err_nn_clean"] for r in rows]),
              ("test error, nearest training image mislabelled", [r["test_err_nn_noisy"] for r in rows]),
              ("excess label-routing of mislabelled images", [r["label_routed"] - r["label_routed_control"] for r in rows]))
    for (color, marker), (label, y) in zip(SERIES, series):
        ax.plot(x, y, color=color, marker=marker, label=label)
    ax.axvline(20000, color=INK2, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("parameters in the tree (depth 2)")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(f"{FIG}/e5_noise.png")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    width_figure("cifar10", "mse", 10, "e2_width_cifar10")
    width_figure("catsdogs", "mse", 2, "e2_width_catsdogs")
    width_figure("cifar10", "mixture", 10, "e2_width_cifar10_ce")
    depth_figure()
    surface_figure()
    samples_figure()
    epochs_figure()
    wd_figure()
    e1_figure()
    e4_figure("cifar10")
    e4_figure("catsdogs")
    e5_figure()

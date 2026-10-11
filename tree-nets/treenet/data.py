"""Dataset preparation and loading.

CIFAR-10 and Cats vs Dogs are read from the Hugging Face parquet exports and
stored as uint8 arrays (N, H, W, 3) in data/<name>.npz with a fixed split.
"""
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
CVD_CLASSES = ["cat", "dog"]
CVD_SIZE = 64


def _decode(blob, size=None):
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    if size is not None:
        w, h = img.size
        s = size / min(w, h)
        img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.uint8)


def _read(path, img_col, label_col, size=None):
    t = pq.read_table(path).to_pydict()
    x = np.stack([_decode(d["bytes"], size) for d in t[img_col]])
    return x, np.array(t[label_col], dtype=np.int64)


def prepare_cifar(raw_dir, n_val=5000, seed=0):
    xtr, ytr = _read(f"{raw_dir}/cifar_train.parquet", "img", "label")
    xte, yte = _read(f"{raw_dir}/cifar_test.parquet", "img", "label")
    perm = np.random.default_rng(seed).permutation(len(xtr))
    val, tr = perm[:n_val], perm[n_val:]
    np.savez(f"{DATA}/cifar10.npz", x_train=xtr[tr], y_train=ytr[tr],
             x_val=xtr[val], y_val=ytr[val], x_test=xte, y_test=yte)


def prepare_catsdogs(raw_dir, n_val=1000, n_test=3000, seed=0):
    parts = [_read(f"{raw_dir}/cvd_{i}.parquet", "image", "labels", CVD_SIZE) for i in (0, 1)]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    perm = np.random.default_rng(seed).permutation(len(x))
    te, val, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    np.savez(f"{DATA}/catsdogs.npz", x_train=x[tr], y_train=y[tr],
             x_val=x[val], y_val=y[val], x_test=x[te], y_test=y[te])


def load(name):
    d = np.load(f"{DATA}/{name}.npz")
    return {k: d[k] for k in d.files}


def classes(name):
    return CIFAR_CLASSES if name == "cifar10" else CVD_CLASSES


def add_label_noise(y, frac, n_classes, seed=0):
    """Replace a fraction of labels with a different, uniformly random class.

    Returns the noisy labels and a boolean mask of the corrupted positions.
    """
    rng = np.random.default_rng(seed)
    y = y.copy()
    flip = rng.random(len(y)) < frac
    shift = rng.integers(1, n_classes, size=flip.sum())
    y[flip] = (y[flip] + shift) % n_classes
    return y, flip


if __name__ == "__main__":
    import sys
    raw = sys.argv[1]
    os.makedirs(DATA, exist_ok=True)
    prepare_cifar(raw)
    prepare_catsdogs(raw)
    for n in ("cifar10", "catsdogs"):
        d = load(n)
        print(n, {k: v.shape for k, v in d.items()}, np.bincount(d["y_train"]))

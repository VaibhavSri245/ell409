"""Feature trunks and feature caching.

Three trunks give the tree its input features:
  random  - untrained random CNN (no labels used)
  cnn     - small CNN trained separately on the training split
  resnet  - ImageNet ResNet-18 (no CIFAR / Cats-vs-Dogs labels used)

Each trunk maps a float image batch (N, 3, H, W) in [0, 1] to a feature
vector. Features for every split are cached in data/feat_<dataset>_<trunk>.npz.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .data import DATA, load

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(x_uint8):
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 255.0


class ResNetTaps(nn.Module):
    """ResNet-18 returning global-average-pooled outputs of layer1..layer4 (64+128+256+512)."""

    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()

    def forward(self, x):
        if x.shape[-1] < 64:
            x = F.interpolate(x, size=64, mode="bilinear", align_corners=False)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        taps = []
        for layer in (n.layer1, n.layer2, n.layer3, n.layer4):
            x = layer(x)
            taps.append(x.mean((2, 3)))
        return torch.cat(taps, 1)


class RandomConv(nn.Module):
    """Frozen random CNN: 3 conv layers, 2x2 average pool at the end -> 1024 features."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.convs = nn.ModuleList([nn.Conv2d(3, 64, 3, padding=1),
                                    nn.Conv2d(64, 128, 3, padding=1),
                                    nn.Conv2d(128, 256, 3, padding=1)])
        for c in self.convs:
            fan_in = c.in_channels * 9
            c.weight.data = torch.randn(c.weight.shape, generator=g) * (2 / fan_in) ** 0.5
            c.bias.data.zero_()

    def forward(self, x):
        x = (x - 0.5) / 0.25
        x = F.max_pool2d(F.relu(self.convs[0](x)), 2)
        x = F.max_pool2d(F.relu(self.convs[1](x)), 2)
        x = F.relu(self.convs[2](x))
        return F.adaptive_avg_pool2d(x, 2).flatten(1)


class SmallCNN(nn.Module):
    """4-stage CNN (32-64-128-256). features() returns pooled stage-2 (64) and stage-4 (256) outputs."""

    def __init__(self, n_classes):
        super().__init__()
        chans = [3, 32, 64, 128, 256]
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(),
                          nn.MaxPool2d(2))
            for a, b in zip(chans[:-1], chans[1:])])
        self.head = nn.Linear(256, n_classes)

    def features(self, x):
        x = (x - 0.5) / 0.25
        taps = []
        for i, s in enumerate(self.stages):
            x = s(x)
            if i in (1, 3):
                taps.append(x.mean((2, 3)))
        return torch.cat(taps, 1)

    def forward(self, x):
        return self.head(self.features(x)[:, 64:])


def train_small_cnn(dataset, epochs, seed=0):
    torch.manual_seed(seed)
    d = load(dataset)
    x, y = to_tensor(d["x_train"]), torch.from_numpy(d["y_train"])
    xv, yv = to_tensor(d["x_val"]), torch.from_numpy(d["y_val"])
    model = SmallCNN(int(y.max()) + 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * (len(x) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=steps)
    pad = x.shape[-1] // 8
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        t0 = time.time()
        for i in range(len(x) // 128):
            idx = perm[i * 128:(i + 1) * 128]
            xb = x[idx]
            flip = torch.rand(len(xb)) < 0.5
            xb[flip] = xb[flip].flip(-1)
            xb = F.pad(xb, (pad,) * 4, mode="reflect")
            dx, dy = np.random.randint(0, 2 * pad + 1, 2)
            xb = xb[..., dy:dy + x.shape[-2], dx:dx + x.shape[-1]]
            loss = F.cross_entropy(model(xb), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            acc = (batched(model, xv).argmax(1) == yv).float().mean().item()
        print(f"[{dataset} cnn] epoch {ep} loss {loss.item():.3f} val acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), f"{DATA}/trunk_cnn_{dataset}.pt")
    return model


@torch.no_grad()
def batched(fn, x, bs=500):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def build_trunk(kind, dataset):
    if kind == "resnet":
        return ResNetTaps().eval()
    if kind == "random":
        return RandomConv().eval()
    n_classes = 10 if dataset == "cifar10" else 2
    m = SmallCNN(n_classes)
    m.load_state_dict(torch.load(f"{DATA}/trunk_cnn_{dataset}.pt"))
    m.eval()
    m.forward = m.features
    return m


def cache_features(kind, dataset):
    d = load(dataset)
    trunk = build_trunk(kind, dataset)
    out = {}
    for split in ("train", "val", "test"):
        t0 = time.time()
        out[f"f_{split}"] = batched(trunk, to_tensor(d[f"x_{split}"])).numpy().astype(np.float32)
        out[f"y_{split}"] = d[f"y_{split}"]
        print(f"[{dataset} {kind}] {split} {out[f'f_{split}'].shape} {time.time() - t0:.0f}s", flush=True)
    np.savez(f"{DATA}/feat_{dataset}_{kind}.npz", **out)


def load_features(dataset, kind):
    """Load cached features, standardised with training-set mean and std."""
    d = np.load(f"{DATA}/feat_{dataset}_{kind}.npz")
    f = {s: d[f"f_{s}"] for s in ("train", "val", "test")}
    mu, sd = f["train"].mean(0), f["train"].std(0) + 1e-6
    out = {s: torch.from_numpy((f[s] - mu) / sd).float() for s in f}
    out.update({f"y_{s}": torch.from_numpy(d[f"y_{s}"]) for s in ("train", "val", "test")})
    out["mu"], out["sd"] = torch.from_numpy(mu).float(), torch.from_numpy(sd).float()
    return out


RESNET_TAPS = {"l1": slice(0, 64), "l2": slice(64, 192), "l3": slice(192, 448), "l4": slice(448, 960)}
CNN_TAPS = {"s2": slice(0, 64), "s4": slice(64, 320)}

if __name__ == "__main__":
    import sys
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    kind, dataset = sys.argv[1], sys.argv[2]
    if kind == "cnn" and not os.path.exists(f"{DATA}/trunk_cnn_{dataset}.pt"):
        train_small_cnn(dataset, epochs=int(sys.argv[3]))
    cache_features(kind, dataset)

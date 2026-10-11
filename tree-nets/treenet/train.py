"""Training and evaluation for SoftTree."""
import time

import torch

from .tree import SoftTree, balance_penalty, tree_loss


@torch.no_grad()
def evaluate(model, xr, xl, y, hard=False, loss_kind="mixture", bs=5000):
    correct, loss = 0, 0.0
    for i in range(0, len(y), bs):
        sl = slice(i, i + bs)
        mu, logits, _ = model(xr[sl], None if xl is None else xl[sl])
        if loss_kind == "mse":
            score = (mu[..., None] * logits).sum(1)
        else:
            score = model.predict(xr[sl], None if xl is None else xl[sl], hard=hard)
        correct += (score.argmax(1) == y[sl]).sum().item()
        loss += tree_loss(mu, logits, y[sl], loss_kind).item() * len(y[sl])
    return correct / len(y), loss / len(y)


def fit(model, train, evals=(), loss_kind="mixture", epochs=100, bs=None, lr=1e-3,
        balance=0.0, wd=0.0, eval_every=10, stop_at_train_acc=None, log=None):
    """Train with Adam. train/evals entries are (xr, xl, y); bs=None means full batch.

    evals: dict name -> (xr, xl, y), evaluated every eval_every epochs.
    Returns a history list of dicts.
    """
    xr, xl, y = train
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = len(y)
    bs = bs or n
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n) if bs < n else torch.arange(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            mu, logits, p = model(xr[idx], None if xl is None else xl[idx])
            loss = tree_loss(mu, logits, y[idx], loss_kind)
            if balance > 0:
                loss = loss + balance * balance_penalty(mu, p, model.depth)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % eval_every == 0 or ep == epochs:
            row = {"epoch": ep, "time": time.time() - t0, "train_loss": loss.item()}
            row["train_acc"], _ = evaluate(model, xr, xl, y, loss_kind=loss_kind)
            for name, (er, el, ey) in dict(evals).items():
                row[f"{name}_acc"], row[f"{name}_loss"] = evaluate(model, er, el, ey, loss_kind=loss_kind)
            hist.append(row)
            if log:
                log(row)
            if stop_at_train_acc is not None and row["train_acc"] >= stop_at_train_acc:
                break
    return hist


def make_tree(d_r, d_l, n_classes, depth, h_router, h_leaf, seed=0):
    torch.manual_seed(seed)
    return SoftTree(d_r, d_l, n_classes, depth, h_router, h_leaf)

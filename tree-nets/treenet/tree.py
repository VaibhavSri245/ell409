"""Soft binary tree whose nodes are small neural networks.

Internal node i (router) : x_r -> MLP -> sigmoid -> probability of going left.
Leaf l (classifier)      : x_l -> MLP -> class scores.

A sample reaches leaf l with probability mu_l(x) = product of router
probabilities on the root-to-leaf path. All nodes of one kind share a shape,
so their weights are stored as stacked tensors and evaluated in one einsum.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedMLP(nn.Module):
    """n independent MLPs d_in -> h -> d_out (h = 0 gives linear maps)."""

    def __init__(self, n, d_in, h, d_out):
        super().__init__()
        self.h = h
        if h > 0:
            self.w1 = nn.Parameter(torch.randn(n, d_in, h) * math.sqrt(2 / d_in))
            self.b1 = nn.Parameter(torch.zeros(n, 1, h))
            self.w2 = nn.Parameter(torch.randn(n, h, d_out) * math.sqrt(1 / h))
        else:
            self.w2 = nn.Parameter(torch.randn(n, d_in, d_out) * math.sqrt(1 / d_in))
        self.b2 = nn.Parameter(torch.zeros(n, 1, d_out))

    def forward(self, x):
        """x: (N, d_in) -> (n, N, d_out)"""
        if self.h > 0:
            x = F.relu(torch.einsum("nd,mdh->mnh", x, self.w1) + self.b1)
            return torch.einsum("mnh,mhk->mnk", x, self.w2) + self.b2
        return torch.einsum("nd,mdk->mnk", x, self.w2) + self.b2


class SoftTree(nn.Module):
    def __init__(self, d_router, d_leaf, n_classes, depth, h_router=16, h_leaf=64):
        super().__init__()
        self.depth = depth
        self.n_classes = n_classes
        self.n_leaves = 2 ** depth
        self.routers = StackedMLP(self.n_leaves - 1, d_router, h_router, 1) if depth > 0 else None
        self.leaves = StackedMLP(self.n_leaves, d_leaf, h_leaf, n_classes)
        self.temperature = 1.0

    def router_probs(self, x_r):
        """(N, n_routers) probability of going left at every internal node (breadth-first order)."""
        return torch.sigmoid(self.routers(x_r)[..., 0].T / self.temperature)

    def path_probs(self, p):
        """Leaf reach probabilities mu (N, n_leaves) from router probabilities p."""
        mu = torch.ones(p.shape[0], 1)
        for level in range(self.depth):
            pl = p[:, 2 ** level - 1: 2 ** (level + 1) - 1]
            mu = torch.stack([mu * pl, mu * (1 - pl)], 2).flatten(1)
        return mu

    def forward(self, x_r, x_l=None):
        """Returns mu (N, L), leaf logits (N, L, K), router probs (N, R)."""
        x_l = x_r if x_l is None else x_l
        logits = self.leaves(x_l).transpose(0, 1)
        if self.depth == 0:
            return torch.ones(x_l.shape[0], 1), logits, None
        p = self.router_probs(x_r)
        return self.path_probs(p), logits, p

    @staticmethod
    def hard_leaf(p, depth):
        """Leaf index reached by following the more likely branch at every router."""
        node = torch.zeros(p.shape[0], dtype=torch.long)
        for _ in range(depth):
            go_left = p.gather(1, node[:, None])[:, 0] >= 0.5
            node = 2 * node + 1 + (~go_left).long()
        return node - (2 ** depth - 1)

    def predict(self, x_r, x_l=None, hard=False):
        """Class probabilities (N, K)."""
        mu, logits, p = self(x_r, x_l)
        probs = logits.softmax(-1)
        if hard and self.depth > 0:
            leaf = self.hard_leaf(p, self.depth)
            return probs[torch.arange(len(leaf)), leaf]
        return (mu[..., None] * probs).sum(1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def active_params(self):
        """Parameters evaluated per sample under hard routing: depth routers + one leaf."""
        per_router = sum(t[0].numel() for t in self.routers.parameters()) if self.depth else 0
        per_leaf = sum(t[0].numel() for t in self.leaves.parameters())
        return self.depth * per_router + per_leaf


def tree_loss(mu, logits, y, kind="mixture"):
    """mixture: -log sum_l mu_l p_l(y)    (hierarchical mixture of experts)
    mse    : || sum_l mu_l out_l - onehot(y) ||^2   (raw leaf outputs, squared loss)
    """
    if kind == "mixture":
        logp = logits.log_softmax(-1)[torch.arange(len(y)), :, y]
        return -torch.logsumexp(torch.log(mu + 1e-12) + logp, 1).mean()
    if kind == "mse":
        pred = (mu[..., None] * logits).sum(1)
        return ((pred - F.one_hot(y, logits.shape[-1]).float()) ** 2).sum(1).mean()
    raise ValueError(kind)


def balance_penalty(mu, p, depth):
    """Frosst & Hinton (2017): push each router to split its reaching mass about 50/50.

    Weighted by 2^-level so that deep nodes, which see few samples, are pushed less.
    """
    if depth == 0:
        return torch.zeros(())
    reach = torch.ones(p.shape[0], 1)
    total = 0.0
    for level in range(depth):
        pl = p[:, 2 ** level - 1: 2 ** (level + 1) - 1]
        alpha = (reach * pl).sum(0) / (reach.sum(0) + 1e-8)
        alpha = alpha.clamp(1e-4, 1 - 1e-4)
        total = total - 2.0 ** -level * 0.5 * (alpha.log() + (1 - alpha).log()).mean()
        reach = torch.stack([reach * pl, reach * (1 - pl)], 2).flatten(1)
    return total

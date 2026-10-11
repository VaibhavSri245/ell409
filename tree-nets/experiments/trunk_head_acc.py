"""Test accuracy of the separately trained CNN trunk's own linear head (reference for E1)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenet.data import DATA, load  # noqa: E402
from treenet.features import SmallCNN, batched, to_tensor  # noqa: E402

out = {}
for ds, k in (("cifar10", 10), ("catsdogs", 2)):
    m = SmallCNN(k)
    m.load_state_dict(torch.load(f"{DATA}/trunk_cnn_{ds}.pt"))
    m.eval()
    d = load(ds)
    pred = batched(m, to_tensor(d["x_test"])).argmax(1).numpy()
    out[ds] = float((pred == d["y_test"]).mean())
print(out)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "trunk_cnn_test.json"), "w"))

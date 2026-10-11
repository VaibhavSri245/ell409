# Trees of small neural networks (ELL409)

Repository: https://github.com/VaibhavSri245/ell409

A binary tree whose internal nodes (routers) and leaves (classifiers) are small MLPs, trained on
CIFAR-10 and Cats vs Dogs. The code covers the five parts of the assignment:

| Part | Script | Output |
|---|---|---|
| Features for the tree (3 trunks) | `treenet/features.py` | `data/feat_<dataset>_<trunk>.npz` |
| E1: fit the training set (100% train accuracy) | `experiments/e1_overfit.py` | `results/e1_overfit.jsonl` |
| E2: double descent (width, depth, 2-D, samples, epochs, weight decay) | `experiments/e2_double_descent.py`, `experiments/run_e2.sh` | `results/e2_*.jsonl` |
| E3: receptive fields of every node | `experiments/e3_receptive_fields.py` | `results/e3_*.json`, `figures/e3_*.png` |
| E4: making the tree small | `experiments/e4_compress.py` | `results/e4_*.json` |
| E5: where mislabelled images go | `experiments/e5_noise.py` | `results/e5_cifar10.json` |
| Figures and tables for the report | `experiments/make_figures.py`, `experiments/make_tables.py` | `figures/*.png`, `report/tables/*.tex` |

The report is `report/report.pdf` (source `report/report.tex`).

## Main results

- E1: all ten trees (depth 3, 0.3M-1.3M parameters) reach 100% training accuracy. Test accuracy is set by the trunk:
  59% / 74% (CIFAR-10 / Cats vs Dogs) with random features, 82% / 91% with ImageNet ResNet-18, 85% / 92% with a CNN
  trained on the task.
- E2: with 2,000 images and squared loss, test error peaks where training loss reaches zero (P = nK = 20k parameters on
  CIFAR-10) and falls again past it. On a depth x width grid the peak follows P = nK. Weight decay 1e-2 removes it.
- E3: every node's receptive field covers the whole image. Cats vs Dogs routers split by background and setting, not
  by class; a task-trained trunk makes routing more semantic and saliency more central.
- E4: a linear classifier on the same features matches the 1.2M-parameter tree; hard routing runs 18% of the tree per
  image; 90% magnitude pruning costs about 1 point.
- E5: mislabelled images are not isolated in a few leaves. Small leaves make the routers bend to wrong labels; large
  leaves memorise them locally.

## Model

`treenet/tree.py`. A tree of depth D has 2^D - 1 routers and 2^D leaves.

- Router j: `z_r -> Linear -> ReLU -> Linear -> sigmoid`, the probability of going left.
- Leaf l: `z_l -> Linear -> ReLU -> Linear`, K class scores.
- Soft routing: leaf l is reached with probability mu_l(x), the product of router probabilities on its path.
  The prediction mixes the leaves: `p(y|x) = sum_l mu_l(x) softmax(f_l(x))_y`.
- Hard routing: follow the larger probability at each router, so only D routers and one leaf run per image.

Losses (`tree_loss`): `mixture` (negative log-likelihood of the mixture) and `mse` (squared loss on one-hot targets). A small balance penalty (Frosst and Hinton, 2017) stops routers
from sending everything one way. All routers share one shape and all leaves share one shape, so each kind is stored as
a stacked weight tensor and evaluated with one `einsum`.

## Features

`treenet/features.py` caches features once; every tree then trains on vectors.

- `random`: an untrained 3-layer CNN (no labels used), 1024 features.
- `cnn`: a 4-layer CNN trained separately on the training split; pooled stage-2 (64) and stage-4 (256) outputs.
- `resnet`: ImageNet ResNet-18; pooled outputs of layer1 to layer4 (64 + 128 + 256 + 512).

`shared` mode feeds all features to every node. `matched` mode feeds shallow taps to routers and deep taps to leaves.

## Reproduce

Python 3.10+, CPU is enough (all results were produced on 2 CPU cores).

```bash
pip install -r requirements.txt
# 1. download the Hugging Face parquet files into raw/
mkdir raw
curl -L -o raw/cifar_train.parquet https://huggingface.co/api/datasets/uoft-cs/cifar10/parquet/plain_text/train/0.parquet
curl -L -o raw/cifar_test.parquet  https://huggingface.co/api/datasets/uoft-cs/cifar10/parquet/plain_text/test/0.parquet
curl -L -o raw/cvd_0.parquet https://huggingface.co/api/datasets/microsoft/cats_vs_dogs/parquet/default/train/0.parquet
curl -L -o raw/cvd_1.parquet https://huggingface.co/api/datasets/microsoft/cats_vs_dogs/parquet/default/train/1.parquet
python -m treenet.data raw                       # data/cifar10.npz, data/catsdogs.npz

# 2. features
for ds in cifar10 catsdogs; do
  python -m treenet.features resnet $ds
  python -m treenet.features random $ds
  python -m treenet.features cnn $ds 12          # trains the CNN trunk first
done

# 3. experiments
for ds in cifar10 catsdogs; do
  for t in resnet random cnn; do python experiments/e1_overfit.py $ds $t shared; done
  for t in resnet cnn; do python experiments/e1_overfit.py $ds $t matched; done
  python experiments/e3_receptive_fields.py $ds resnet
  python experiments/e3_receptive_fields.py $ds cnn
  python experiments/e3_receptive_fields.py $ds pixel
  python experiments/e4_compress.py $ds
done
./experiments/run_e2.sh
python experiments/e5_noise.py cifar10
python experiments/trunk_head_acc.py
python experiments/make_figures.py
python experiments/make_tables.py
cd report && pdflatex report.tex && pdflatex report.tex
```

Splits are fixed by seed 0: CIFAR-10 uses 45,000 train / 5,000 validation (taken from the official training set) /
10,000 test. Cats vs Dogs (23,410 images, resized to 64x64 with a centre crop) uses 19,410 / 1,000 / 3,000.

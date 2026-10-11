#!/bin/sh
# Double-descent sweeps used in the report (about 4 CPU-hours in total).
cd "$(dirname "$0")/.."
python3 experiments/e2_double_descent.py width cifar10 mse
python3 experiments/e2_double_descent.py width catsdogs mse
python3 experiments/e2_double_descent.py width6k catsdogs mse
python3 experiments/e2_double_descent.py surface cifar10 mse
python3 experiments/e2_double_descent.py depth cifar10 mse
python3 experiments/e2_double_descent.py samples cifar10 mse
python3 experiments/e2_double_descent.py epochs cifar10 mse
python3 experiments/e2_double_descent.py width cifar10 mixture
python3 experiments/e2_double_descent.py wd cifar10 mse

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_PATH="${PROJECT_DIR}/main.py"
UV_BIN="${UV_BIN:-uv}"

configs=(
  "cifar10_resnet18_sgd_bs16.yaml"
  "cifar10_resnet18_sgd_bs64.yaml"
  "cifar10_resnet18_sgd_bs256.yaml"
  "cifar10_resnet18_sgd_bs1024.yaml"
  "cifar10_resnet18_adamw_bs16.yaml"
  "cifar10_resnet18_adamw_bs64.yaml"
  "cifar10_resnet18_adamw_bs256.yaml"
  "cifar10_resnet18_adamw_bs1024.yaml"
  "cifar100_resnet34_sgd_bs16.yaml"
  "cifar100_resnet34_sgd_bs64.yaml"
  "cifar100_resnet34_sgd_bs256.yaml"
  "cifar100_resnet34_sgd_bs1024.yaml"
  "cifar100_resnet34_adamw_bs16.yaml"
  "cifar100_resnet34_adamw_bs64.yaml"
  "cifar100_resnet34_adamw_bs256.yaml"
  "cifar100_resnet34_adamw_bs1024.yaml"
  "cifar10_cabs_2conv_3dense_sgd_bs16.yaml"
  "cifar10_cabs_2conv_3dense_sgd_bs64.yaml"
  "cifar10_cabs_2conv_3dense_sgd_bs256.yaml"
  "cifar10_cabs_2conv_3dense_sgd_bs1024.yaml"
)

for rel in "${configs[@]}"; do
  echo "==> ${rel}"
  "${UV_BIN}" run python "${MAIN_PATH}" --config "${SCRIPT_DIR}/${rel}"
done

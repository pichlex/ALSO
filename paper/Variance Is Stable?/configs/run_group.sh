#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <dataset> <optimizer> <model>" >&2
  echo "example: $0 cifar10 sgd resnet18" >&2
  exit 1
fi

dataset="$1"
optimizer="$2"
model="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_PATH="${PROJECT_DIR}/main.py"
UV_BIN="${UV_BIN:-uv}"

batch_sizes=(16 64 256 1024)

for bs in "${batch_sizes[@]}"; do
  config="${SCRIPT_DIR}/${dataset}_${model}_${optimizer}_bs${bs}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "missing config: ${config}" >&2
    exit 1
  fi
  echo "==> $(basename "${config}")"
  "${UV_BIN}" run python "${MAIN_PATH}" --config "${config}"
done

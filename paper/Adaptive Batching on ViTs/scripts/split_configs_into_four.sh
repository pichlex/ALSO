#!/usr/bin/env bash
set -euo pipefail

# Split a config directory into four runnable scripts.
CONFIG_ROOT="${CONFIG_ROOT:-paper/Adaptive Batching on ViTs/configs-uai/sgd/food101}"
WORKDIR="${WORKDIR:-paper/Adaptive Batching on ViTs}"
OUT_PREFIX="${OUT_PREFIX:-${CONFIG_ROOT}/run-part}"
VARIANT="${VARIANT:-}"
UV_BIN="${UV_BIN:-uv}"
MAIN_PATH="${MAIN_PATH:-main.py}"

CONFIG_ROOT_ABS=$(cd "${CONFIG_ROOT}" && pwd)
WORKDIR_ABS=$(cd "${WORKDIR}" && pwd)

configs=()
if [[ -n "${VARIANT}" ]]; then
  while IFS= read -r line; do
    configs+=("$line")
  done < <(find "${CONFIG_ROOT_ABS}" -type f -name '*.json' -path "*/${VARIANT}/*" | sort)
else
  while IFS= read -r line; do
    configs+=("$line")
  done < <(find "${CONFIG_ROOT_ABS}" -type f -name '*.json' | sort)
fi
total=${#configs[@]}

if (( total == 0 )); then
  echo "No configs found under ${CONFIG_ROOT}" >&2
  exit 1
fi

per_chunk=$(((total + 3) / 4))

write_chunk() {
  local idx=$1
  local start=$2
  local out="${OUT_PREFIX}${idx}.sh"
  local chunk=("${configs[@]:start:per_chunk}")
  if (( ${#chunk[@]} == 0 )); then
    return
  fi

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "UV_BIN=\${UV_BIN:-${UV_BIN}}"
    echo "SCRIPT_DIR=\$(cd -- \"\$(dirname -- \"\${BASH_SOURCE[0]}\")\" && pwd)"
    echo "WORKDIR=\${WORKDIR:-\$(cd -- \"\${SCRIPT_DIR}/../../..\" && pwd)}"
    echo "CONFIG_DIR=\${CONFIG_DIR:-\${SCRIPT_DIR}}"
    echo "MAIN_PATH=\${MAIN_PATH:-\"${MAIN_PATH}\"}"
    echo
    echo "cd \"\${WORKDIR}\""
    for cfg in "${chunk[@]}"; do
      rel_cfg=${cfg#"${CONFIG_ROOT_ABS}/"}
      echo "\"\${UV_BIN}\" run \"\${MAIN_PATH}\" --config \"\${CONFIG_DIR}/${rel_cfg}\""
    done
  } > "${out}"
  chmod +x "${out}"
  echo "Wrote ${out} (${#chunk[@]} jobs)"
}

for i in {0..3}; do
  start=$((i * per_chunk))
  write_chunk $((i + 1)) "${start}"
done

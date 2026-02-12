#!/usr/bin/env bash
set -euo pipefail

# Split a config directory into four runnable scripts.
CONFIG_ROOT="${CONFIG_ROOT:-paper/Adaptive Batching on ViTs/configs-uai/sgd/food101}"
OUT_PREFIX="${OUT_PREFIX:-${CONFIG_ROOT}/run-part}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PATH="${MAIN_PATH:-paper/Adaptive Batching on ViTs/main.py}"

configs=()
while IFS= read -r line; do
  configs+=("$line")
done < <(find "${CONFIG_ROOT}" -type f -name '*.json' | sort)
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
    echo "PYTHON_BIN=\${PYTHON_BIN:-${PYTHON_BIN}}"
    echo "MAIN_PATH=\${MAIN_PATH:-\"${MAIN_PATH}\"}"
    echo
    for cfg in "${chunk[@]}"; do
      echo "\"\${PYTHON_BIN}\" \"\${MAIN_PATH}\" --config \"${cfg}\""
    done
  } > "${out}"
  chmod +x "${out}"
  echo "Wrote ${out} (${#chunk[@]} jobs)"
}

for i in {0..3}; do
  start=$((i * per_chunk))
  write_chunk $((i + 1)) "${start}"
done

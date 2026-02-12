#!/usr/bin/env bash
set -euo pipefail
UV_BIN=${UV_BIN:-uv}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=${WORKDIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
CONFIG_DIR=${CONFIG_DIR:-${SCRIPT_DIR}}
MAIN_PATH=${MAIN_PATH:-"main.py"}

cd "${WORKDIR}"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs512-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs512-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs512-seed3.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs64-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs64-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs64-seed3.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs8-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs8-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs8-seed3.json"

#!/usr/bin/env bash
set -euo pipefail
UV_BIN=${UV_BIN:-uv}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=${WORKDIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
CONFIG_DIR=${CONFIG_DIR:-${SCRIPT_DIR}}
MAIN_PATH=${MAIN_PATH:-"main.py"}

cd "${WORKDIR}"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs1024-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs1024-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs1024-seed3.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs16-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs16-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs16-seed3.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs256-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs256-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "${CONFIG_DIR}/aboba/bs256-seed3.json"

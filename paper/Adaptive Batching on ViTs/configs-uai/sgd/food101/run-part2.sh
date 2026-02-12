#!/usr/bin/env bash
set -euo pipefail
UV_BIN=${UV_BIN:-uv}
MAIN_PATH=${MAIN_PATH:-"paper/Adaptive Batching on ViTs/main.py"}

"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs512-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs512-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs512-seed3.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs64-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs64-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs64-seed3.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs8-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs8-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs8-seed3.json"

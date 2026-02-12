#!/usr/bin/env bash
set -euo pipefail
UV_BIN=${UV_BIN:-uv}
MAIN_PATH=${MAIN_PATH:-"paper/Adaptive Batching on ViTs/main.py"}

"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs1024-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs1024-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs1024-seed3.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs16-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs16-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs16-seed3.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs256-seed1.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs256-seed2.json"
"${UV_BIN}" run "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs256-seed3.json"

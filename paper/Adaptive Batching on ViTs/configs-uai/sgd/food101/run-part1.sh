#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
MAIN_PATH=${MAIN_PATH:-"paper/Adaptive Batching on ViTs/main.py"}

"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs1024-seed1.json"
"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs1024-seed2.json"
"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs1024-seed3.json"
"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs16-seed1.json"
"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs16-seed2.json"
"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs16-seed3.json"
"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs256-seed1.json"
"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs256-seed2.json"
"${PYTHON_BIN}" "${MAIN_PATH}" --config "paper/Adaptive Batching on ViTs/configs-uai/sgd/food101/aboba/bs256-seed3.json"

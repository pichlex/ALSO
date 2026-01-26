#!/usr/bin/env bash

uv run main.py --config configs/sgd/food101/aboba/config8.json
uv run main.py --config configs/sgd/food101/aboba/config16.json
uv run main.py --config configs/sgd/food101/aboba/config32.json
uv run main.py --config configs/sgd/food101/aboba/config48.json

uv run main.py --config configs/sgd/food101/basic/config64.json
uv run main.py --config configs/sgd/food101/basic/config128.json
uv run main.py --config configs/sgd/food101/basic/config256.json
uv run main.py --config configs/sgd/food101/basic/config512.json

uv run main.py --config configs/sgd/food101/seesaw/config.json
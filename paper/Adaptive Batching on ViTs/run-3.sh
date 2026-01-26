#!/usr/bin/env bash

uv run main.py --config configs/sgd/tiny/aboba/config8.json
uv run main.py --config configs/sgd/tiny/aboba/config16.json
uv run main.py --config configs/sgd/tiny/aboba/config32.json
uv run main.py --config configs/sgd/tiny/aboba/config48.json

uv run main.py --config configs/sgd/tiny/basic/config64.json
uv run main.py --config configs/sgd/tiny/basic/config128.json
uv run main.py --config configs/sgd/tiny/basic/config256.json
uv run main.py --config configs/sgd/tiny/basic/config512.json

uv run main.py --config configs/sgd/tiny/seesaw/config.json

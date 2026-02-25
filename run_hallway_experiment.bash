#!/bin/bash
screen -dmS maic_hallway bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=0'
screen -dmS maic_hallway bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=2289'
screen -dmS maic_hallway bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=3608'

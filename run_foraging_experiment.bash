#!/bin/bash
screen -dmS maic_foraging bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=0'
screen -dmS maic_foraging bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=2289'
screen -dmS maic_foraging bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=3608'

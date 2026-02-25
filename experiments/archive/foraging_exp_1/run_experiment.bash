#!/bin/bash
# single runs
screen -dmS maic_foraging_1 bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=0'
screen -dmS maic_foraging_2 bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=2289'
screen -dmS maic_foraging_3 bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=3608'

# parallel runs
# 20
screen -dmS maic_foraging_1_parallel bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=0 runner=parallel batch_size_run=20'
screen -dmS maic_foraging_2_parallel bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=2289 runner=parallel batch_size_run=20'
screen -dmS maic_foraging_3_parallel bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=foraging with seed=3608 runner=parallel batch_size_run=20'

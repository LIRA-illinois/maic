#!/bin/bash
# single runs
screen -dmS maic_hallway_1 bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=0'
screen -dmS maic_hallway_2 bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=2289'
screen -dmS maic_hallway_3 bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=3608'

# parallel runs
screen -dmS maic_hallway_1_parallel bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=0 runner=parallel batch_size_run=20'
screen -dmS maic_hallway_2_parallel bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=2289 runner=parallel batch_size_run=20'
screen -dmS maic_hallway_3_parallel bash -c 'source .venv/bin/activate; python3 src/main.py --config=maic --env-config=join1 with seed=3608 runner=parallel batch_size_run=20'


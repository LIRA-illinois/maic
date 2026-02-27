venv_name=.venv
project_name=maic

####################
# Experiment management
####################
# shows GPU status, useful for checking how much VRAM is in use
nvidia:
	watch -n 0.2 nvidia-smi

tb:
	screen -dmS tensorboard_${project_name} bash -c 'source .venv/bin/activate; tensorboard --bind_all --port=6009 --logdir "results/tb_logs/"'

list_screen_experiments:
	screen -ls | grep "exp" | awk "{print $1}" | cut -d"	" -f 2

# find screen sesions with "exp" in them, use cut to grab the session names, adds a prefix and suffix to quit the session, puts commands in a txt file, and opens the txt file. Does NOT stop the experiments, user must choose which sessions to quit and copy + paste the commands into the terminal.
list_screen_experiments_quit:
	screen -ls | grep "exp" | awk "{print $1}" | cut -d"	" -f 2 | sed 's/^/screen -X -S /; s/$$/ quit/' > screen_cmds.txt
	code screen_cmds.txt
	sleep 0.1
	rm -r screen_cmds.txt

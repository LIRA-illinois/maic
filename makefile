####################
# Experiment management
####################
# shows GPU status, useful for checking how much VRAM is in use
nvidia:
	watch -n 0.2 nvidia-smi

tb:
	tensorboard --bind_all --port=6008 --logdir "results/tb_logs/"

#!/bin/bash
export MASTER_ADDR=$(hostname) # For DDP

export MPICH_GPU_SUPPORT_ENABLED=0

config_path=./config.yaml

cmd="python train_vqvae.py --config $config_path"

# srun to get multiple GPUs; source DDP vars to use pytorch DDP
srun -l bash -c "source export_DDP_vars.sh && $cmd"



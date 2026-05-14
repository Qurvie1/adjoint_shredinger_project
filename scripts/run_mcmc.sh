#!/bin/bash
# scripts/run_mcmc.sh

export CUDA_VISIBLE_DEVICES=0

python -m mcmc.metropolis \
    --system ala2 \
    --n_steps 1000000 \
    --save_interval 1000 \
    --proposal_scale 0.1 \
    --kbt 0.6 \
    --output_dir outputs/mcmc_ala2
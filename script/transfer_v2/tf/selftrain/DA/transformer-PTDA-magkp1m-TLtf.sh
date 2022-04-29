#!/usr/bin/env bash

# Run the job
export CONFIG_PATH="script/transfer_v2/tf/selftrain/DA/transformer-PTDA-magkp1m-TLtf-round10.yml"
CUDA_VISIBLE_DEVICES=2 nohup python train.py -config $CONFIG_PATH > slurm_output/transformer-PTDA-magkp1m-TLtf-round10.nohup.out &

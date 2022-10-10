#!/usr/bin/env bash

# Run the job
export CONFIG_PATH="script/transfer_v2/tf/selftrain/FT/transformer-PTDAFT-magkp1m-round7-fewshot10k.yml"
CUDA_VISIBLE_DEVICES=1 nohup python train.py -config $CONFIG_PATH > slurm_output/transformer-PTDAFT-magkp1m-round7-fewshot10k.nohup.out &

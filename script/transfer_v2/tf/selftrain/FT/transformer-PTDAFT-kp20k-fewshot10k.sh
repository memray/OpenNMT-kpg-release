#!/usr/bin/env bash

# Run the job
export CONFIG_PATH="script/transfer_v2/tf/selftrain/FT/transformer-PTDAFT-kp20k-round10-fewshot10k.yml"
CUDA_VISIBLE_DEVICES=0 nohup python train.py -config $CONFIG_PATH > slurm_output/transformer-PTDAFT-kp20k-round10-fewshot10k.nohup.out &

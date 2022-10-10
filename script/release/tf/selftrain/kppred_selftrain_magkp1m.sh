#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 nohup /ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks pred -data_dir /zfs1/hdaqing/rum20/kp/data/kp/json/ -exp_root_dir /zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_PTDA_selftrain/transformer-PTDA_TLtf_magkp1m_step20k-tlrs_55-round9/ckpts -gpu 0 -batch_size 64 -beam_size 1 -max_length 60 -testsets magkpcs_1m -splits train --wait_time 0 --step_base 1 --data_format jsonl > slurm_output/kppred_selftrain_magkpcs1m_round9.nohup.out &

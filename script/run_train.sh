#!/usr/bin/env bash
# train
CUDA_VISIBLE_DEVICES=0,1 nohup python train.py -config config/config-transformer-keyphrase.yml > output/nohup.kp20k.one2one.transformer.log &
CUDA_VISIBLE_DEVICES=2,3 nohup python train.py -config config/config-transformer-keyphrase-magkp.yml > output/nohup.mag.one2one.transformer.log &
CUDA_VISIBLE_DEVICES=4,5 nohup python train.py -config config/config-rnn-keyphrase.yml > output/nohup.kp20k.one2one.rnn.log &
CUDA_VISIBLE_DEVICES=6,7 nohup python train.py -config config/config-rnn-keyphrase-magkp.yml > output/nohup.mag.one2one.rnn.log &

CUDA_VISIBLE_DEVICES=4 nohup python train.py -config config/train/config-rnn-keyphrase.drop00.yml > output/keyphrase/meng17/nohup.kp20k.one2one.birnn.Dropout00.log &
CUDA_VISIBLE_DEVICES=5 nohup python train.py -config config/train/config-rnn-keyphrase.drop05.yml > output/keyphrase/meng17/nohup.kp20k.one2one.birnn.Dropout05.log &
CUDA_VISIBLE_DEVICES=6 nohup python train.py -config config/train/config-rnn-keyphrase.drop05.coverage.yml > output/keyphrase/meng17/nohup.kp20k.one2one.birnn.Dropout05.CovATT.log &
CUDA_VISIBLE_DEVICES=7 nohup python train.py -config config/train/config-rnn-keyphrase.drop05.coverage.noreuse.yml > output/keyphrase/meng17/nohup.kp20k.one2one.birnn.Dropout05.CovATT.NoReuse.log &


# Opal
CUDA_VISIBLE_DEVICES=0 nohup sh run_script/kp-transformer-1gpu-bs4096-train300k.sh > output/keyphrase/meng17/nohup_magkp-meng17-on2one-transformer-Layer4-Heads8-Dim512-Emb512-Dropout0.1-Copytrue-Covtrue-Contextboth.log &
CUDA_VISIBLE_DEVICES=1 nohup sh run_script/kp-transformer-1gpu-bs4096-train300k.sh > output/keyphrase/meng17/nohup_magkp-meng17-on2one-transformer-Layer4-Heads8-Dim128-Emb128-Dropout0.1-Copytrue-Covtrue-Contextboth.log &
CUDA_VISIBLE_DEVICES=2 nohup sh run_script/kp-transformer-1gpu-bs4096-train300k.sh > output/keyphrase/meng17/nohup_magkp-meng17-on2one-transformer-Layer2-Heads4-Dim128-Emb128-Dropout0.1-Copytrue-Covtrue-Contextboth.log &

# test (use CPU is safer)
# kp20k kp20k_valid2k duc inspec krapivin nus semeval
# nus & semeval
mkdir -p "output/keyphrase/meng17-bs32/"
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets nus semeval -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-nus_semeval.log &
# DUC
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets duc -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-duc.log &
# inspec
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets inspec -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-inspec.log &
#krapivin
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets krapivin -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-krapivin.log &
# kp20k_valid2k
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets kp20k_valid2k -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-kp20k_valid2k.log &

# kp20k
nohup python kp_run_eval.py -config config/test/config-rnn-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/selected/ -output_dir output/keyphrase/meng17/ -testsets kp20k -batch_size 64 > output/keyphrase/meng17/kp_run_eval-kp20k.log &


# CRC
sbatch script/srun_kp-transformer-1gpu-bs4096-train300k.sh

sbatch script/kpeval_duc.sh
sbatch script/kpeval_nus.sh
sbatch script/kpeval_inspec.sh
sbatch script/kpeval_krapivin.sh
sbatch script/kpeval_kp20k_valid2k.sh

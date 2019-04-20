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
4695 4696 4697 4698

# test (use CPU is safer)
# kp20k kp20k_valid2k duc inspec krapivin nus semeval
# nus & semeval
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets nus semeval -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-nus_semeval.log &
# DUC
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets duc -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-duc.log &
# inspec
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets inspec -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-inspec.log &
#krapivin
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets krapivin -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-krapivin.log &
# kp20k_valid2k
nohup python kp_run_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-bs32/ -testsets kp20k_valid2k -batch_size 64 > output/keyphrase/meng17-bs32/kp_run_eval-kp20k_valid2k.log &
5183 5523 5185 5186 5250
31613 31712 31811 31910 32009

# kp20k
nohup python kp_run_eval.py -config config/test/config-rnn-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/selected/ -output_dir output/keyphrase/meng17/ -testsets kp20k -batch_size 64 > output/keyphrase/meng17/kp_run_eval-kp20k.log &
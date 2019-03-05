#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 nohup python train.py -config config/config-transformer-keyphrase.yml > output/nohup.kp20k.one2one.transformer.log &
CUDA_VISIBLE_DEVICES=2,3 nohup python train.py -config config/config-transformer-keyphrase-magkp.yml > output/nohup.mag.one2one.transformer.log &
CUDA_VISIBLE_DEVICES=4,5 nohup python train.py -config config/config-rnn-keyphrase.yml > output/nohup.kp20k.one2one.rnn.log &
CUDA_VISIBLE_DEVICES=6,7 nohup python train.py -config config/config-rnn-keyphrase-magkp.yml > output/nohup.mag.one2one.rnn.log &
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 nohup python train.py -config config/config-transformer-keyphrase.yml > output/nohup_kp20k.one2one.transformer.400k.log &
CUDA_VISIBLE_DEVICES=5,6 nohup python train.py -config config/config-rnn-keyphrase.yml > output/nohup_kp20k.one2one.rnn.200k.log &
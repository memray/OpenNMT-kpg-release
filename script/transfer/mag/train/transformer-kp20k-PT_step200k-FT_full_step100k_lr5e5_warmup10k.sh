#!/usr/bin/env bash

#sleep 9000
source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found

cd /zfs1/hdaqing/rum20/kp/OpenNMT-kpg-transfer/

mkdir /zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_transformer_fewshot_v2/transformer-kp20k-PT_step200k-FT_full_step100k_lr5e5_warmup10k/

PYTHONUNBUFFERED=1;TOKENIZERS_PARALLELISM=false;CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 train.py -config script/transfer/mag/train/transformer-kp20k-PT_step200k-FT_full_step100k_lr5e5_warmup10k.yml --report_every 100 > /zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_transformer_fewshot_v2/transformer-kp20k-PT_step200k-FT_full_step100k_lr5e5_warmup10k/train.nohup.out &

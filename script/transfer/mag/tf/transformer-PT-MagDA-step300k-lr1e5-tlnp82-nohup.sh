#!/usr/bin/env bash

source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found

cd /zfs1/hdaqing/rum20/kp/OpenNMT-kpg-transfer/

PYTHONUNBUFFERED=1;TOKENIZERS_PARALLELISM=false;CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 train.py -config script/transfer/mag/train/transformer-PT-MagDA-step300k-lr1e5-tlnp82.yml --report_every 100 > /zfs1/hdaqing/rum20/kp/transfer_exps/kp_mag/transformer-PT-DA_MagTL-step300k-lr1e5-tlnp82/train.nohup.out &

#!/usr/bin/env bash

source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found

cd /zfs1/hdaqing/rum20/kp/OpenNMT-kpg-transfer/

PYTHONUNBUFFERED=1;TOKENIZERS_PARALLELISM=false;CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 train.py -config script/transfer_v2/mag/transformer-PTDA-magcs12m-tlrs55-step200k.yml --report_every 100 > /zfs1/hdaqing/rum20/kp/transfer_exps_v2/mag_DA/transformer-PTDA_magcs12m_tlrs55-lr1e5-step200k/train.nohup.out &

#!/usr/bin/env bash

source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found

cd /zfs1/hdaqing/rum20/kp/fairseq-kpg

#sleep 14400
#kill -9 192000
#kill -9 192001
#kill -9 192002
#kill -9 192003

PYTHONUNBUFFERED=1;TOKENIZERS_PARALLELISM=false;CUDA_VISIBLE_DEVICES=1,2,3 nohup /ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 train.py /zfs1/hdaqing/rum20/kp/data/wiki/phrase --disable-validation --save-dir /zfs1/hdaqing/rum20/kp/transfer_exps_v2/kp_PT/bart-wiki-step200k/ckpts --task keyphrasification_pretrain --dataset-type wiki --max-source-length 512 --max-target-length 256 --max-phrase-len 6 --max-target-phrases 16 --phrase-corr-rate 0.1 --random-span-rate 0.05 --arch bart_large --restore-file /zfs1/hdaqing/rum20/kp/data/kp/cache/bart.large/model.pt --bpe hf_pretrained_bpe --bpe-vocab /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.1 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --lr 1e-5 --update-freq 6 --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --seed 7 --fixed-validation-seed 7 --max-tokens 1024 --clip-norm 1.0 --save-interval-updates 5000 --warmup-updates 10000 --total-num-update 200000 --num-workers 12 --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project transfer_kp_wiki > /zfs1/hdaqing/rum20/kp/transfer_exps_v2/kp_PT/bart-wiki-step200k/train.nohup.out &


#!/usr/bin/env bash

#sleep 14400
source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
cd /zfs1/hdaqing/rum20/kp/fairseq-kpg

exp_dir=/zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_mag/bart-DA_MagTL100k-lr1e5-step20k
mkdir -p $exp_dir

PYTHONUNBUFFERED=1;TOKENIZERS_PARALLELISM=false;CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 train.py /zfs1/hdaqing/rum20/kp/data/kp/oag_v1_cs_nokp_100k/ --dataset-type scipaper --label-data /zfs1/hdaqing/rum20/kp/data/kp/oag_v1_cs_nokp_wikiTL_100k/ --label-sample-ratio [1.0] --save-dir ${exp_dir}/ckpts --disable-validation --task keyphrasification --max-source-length 512 --max-target-length 64 --max-target-phrases 16 --kp-concat-type pres_abs --arch bart_large --restore-file /zfs1/hdaqing/rum20/kp/data/kp/cache/bart.large/model.pt --bpe hf_pretrained_bpe --bpe-vocab /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.1 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-06 --weight-decay 0.01 --lr 1e-5 --lr-scheduler polynomial_decay --clip-norm 1.0 --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --log-format simple --log-interval 10 --fixed-validation-seed 7 --max-tokens 640 --update-freq 5 --save-interval-updates 5000 --warmup-updates 2000 --max-update 20000 --total-num-update 20000 --num-workers 20 --find-unused-parameters --memory-efficient-fp16 --ddp-backend=no_c10d --wandb-project transfer_kp_mag  > ${exp_dir}/train.nohup.out &

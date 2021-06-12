#!/usr/bin/env bash

#/home/ubuntu/anaconda3/bin/conda config --set env_prompt '({name})'
#/home/ubuntu/anaconda3/bin/conda activate /home/ubuntu/efs/.conda/kp

export TOKENIZERS_PARALLELISM=false
export WANDB_NAME=bart_kppretrain_wiki_1e5_controlled-DA_kp20k-NP_TL-step100k
export WANDB_API_KEY=72618587b1afa7c116440deb53224bd999919d0f
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/ubuntu/efs/rum20/fairseq-kpg/fairseq_cli

UPDATE_FREQ=9

/home/ubuntu/efs/.conda/kp/bin/python3.7 train.py /home/ubuntu/efs/rum20/data/kp/json/kp20k_train100k/train.json --dataset-type scipaper --label-data /home/ubuntu/efs/rum20/data/kp/json/kp20k_train100k/train.noun_phrase.json:/home/ubuntu/efs/rum20/exps/bart_kppretrain_wiki_1e5_controlled/outputs/beamsearch-width_1-maxlen_40/pred/checkpoint_step_100000-data_kp20k_train100k_train.pred --label-sample-ratio '(0.5,0.5)' --save-dir /home/ubuntu/efs/rum20/exps/bart_kppretrain_wiki_1e5_controlled-DA_kp20k-NP_TL/ckpts --disable-validation --task keyphrasification --max-source-length 512 --max-target-length 128 --kp-concat-type pres_abs --add-control-prefix-prob 0.8 --max-target-phrases 30 --max-phrase-len 8 --arch bart_large --restore-file /home/ubuntu/efs/rum20/exps/bart_kppretrain_wiki_1e5_controlled/ckpts/checkpoint_step_100000.pt --bpe hf_pretrained_bpe --bpe-vocab /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path /home/ubuntu/efs/rum20/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.0 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-08 --clip-norm 0.1 --lr 1e-6 --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --fixed-validation-seed 7 --max-tokens 512 --update-freq $UPDATE_FREQ --save-interval-updates 2000 --warmup-updates 2000 --total-num-update 20000 --num-workers 8 --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project transfer_kp_wiki

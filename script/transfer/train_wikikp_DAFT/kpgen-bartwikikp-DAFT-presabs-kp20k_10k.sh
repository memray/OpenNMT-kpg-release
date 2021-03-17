#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --partition=gtx1080
#SBATCH --partition=v100
#SBATCH --account=hdaqing

#SBATCH --job-name=train-bartwikikp-DAFT-kp20k-fewshot10k
#SBATCH --output=slurm_output/train-bartwikikp-DAFT-kp20k-fewshot10k.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Load modules
#module restore
#module load cuda/10.0.130
#module load gcc/6.3.0
#module load python/anaconda3.6-5.2.0
#source activate py36
#module unload python/anaconda3.6-5.2.0

# GPU usage:
#   max_token=2048, update_freq=4, OOM
#   max_token=1536, update_freq=6, 27xxxMiB / 32480MiB, OOM...
#   max_token=1024, update_freq=8,  / 32480MiB
cd /zfs1/hdaqing/rum20/kp/fairseq-kpg/fairseq_cli/
export WANDB_NAME=bartwikikp-DAFT-kp20k-fewshot10k
export TOKENIZERS_PARALLELISM=false
cmd="python train.py /zfs1/hdaqing/rum20/kp/data/kp/json/kp20k_train10k/train.json --dataset-type scipaper --save-dir /zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp_fewshot10k/bartwikikp_DAFT_kp20k_fewshot10k/ckpts --disable-validation --task keyphrasification --max-source-length 512 --max-target-length 128 --kp-concat-type pres_abs --arch bart_large --restore-file /zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp_wiki/wiki-DA_kp20k-NP_TL/ckpts/checkpoint_step_20000.pt --bpe hf_pretrained_bpe --bpe-vocab /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.0 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas (0.9,0.999) --adam-eps 1e-08 --clip-norm 0.1 --lr 1e-5 --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --fixed-validation-seed 7 --max-tokens 2752 --update-freq 6 --save-interval-updates 500 --warmup-updates 1000 --total-num-update 10000 --num-workers 8 --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project transfer_kp_fewshot"

echo $CONFIG_PATH
echo $cmd

$cmd

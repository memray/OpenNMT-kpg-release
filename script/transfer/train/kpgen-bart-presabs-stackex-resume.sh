#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:2
#SBATCH --partition=titanx
#SBATCH --partition=gtx1080s
#SBATCH --partition=v100
#SBATCH --account=hdaqing

#SBATCH --job-name=train-bart-stackex
#SBATCH --output=slurm_output/train-bart-stackex-rerun.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=3-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Load modules
#module restore
#module load cuda/10.0.130
#module load gcc/6.3.0
#module load python/anaconda3.6-5.2.0
#source activate py36
#module unload python/anaconda3.6-5.2.0

# GPU usage: bsz=~90: 31k+MiB / 32480MiB
cd /zfs1/hdaqing/rum20/kp/fairseq-kpg/fairseq_cli/
export WANDB_NAME=bartFT_presabs_stackex_100kstep
export TOKENIZERS_PARALLELISM=false
cmd="python train.py /zfs1/hdaqing/rum20/kp/data/kp/json/stackex/ --save-dir /zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp/bartFT_presabs_stackex_100k_rerun/ckpts --disable-validation --task keyphrasification --max-source-length 512 --max-target-length 128 --kp-concat-type pres_abs --arch bart_large --restore-file checkpoint_last.pt --bpe hf_pretrained_bpe --bpe-vocab /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.0 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --optimizer adam --adam-betas (0.9,0.999) --adam-eps 1e-08 --clip-norm 0.1 --lr 1e-5 --lr-scheduler polynomial_decay --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --fixed-validation-seed 7 --max-tokens 512 --update-freq 16 --save-interval-updates 5000 --warmup-updates 10000 --total-num-update 100000 --max-update 100000 --num-workers 16 --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project transfer_kp"

echo $CONFIG_PATH
echo $cmd

$cmd

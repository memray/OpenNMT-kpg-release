#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=hdaqing

#SBATCH --partition=gtx1080
#SBATCH --partition=titanx
#SBATCH --partition=v100
#SBATCH --job-name=v100_eval
#SBATCH --output=slurm_output/v100_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

cmd="python kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks pred -data_dir /zfs1/hdaqing/rum20/kp/data/kp/json/ -exp_root_dir /zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp/ -testsets kp20k openkp kptimes jptimes stackex -splits test -batch_size 8 -beam_size 50 -max_length 40 -beam_terminate full --step_base 10000 --data_format jsonl -gpu 0"

echo $cmd
$cmd

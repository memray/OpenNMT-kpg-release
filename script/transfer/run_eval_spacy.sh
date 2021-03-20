#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --cluster=htc
#SBATCH --partition=scavenger

#SBATCH --account=hdaqing

#SBATCH --job-name=kp_spacyeval
#SBATCH --output=slurm_output/kp_spacyeval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=3-0:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

exp_root_dir="/zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp_o2o/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp_wiki/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp_fewshot10k/"

dataset_args="kp20k kp20k_valid2k kptimes kptimes_valid2k jptimes duc openkp openkp_valid2k stackex stackex_valid2k"

cmd="/ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks eval -data_dir /zfs1/hdaqing/rum20/kp/data/kp/json/ -exp_root_dir $exp_root_dir -testsets $dataset_args -splits test -batch_size 8 -beam_size 1 -max_length 40 -beam_terminate full --step_base 1 --data_format jsonl"

echo $cmd
echo $PWD
$cmd

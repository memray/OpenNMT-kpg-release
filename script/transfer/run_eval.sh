#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --cluster=htc
#SBATCH --partition=scavenger

#SBATCH --account=hdaqing

#SBATCH --job-name=kp_eval
#SBATCH --output=slurm_output/kp_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1-0:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found

exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_fewshot/"
exp_root_dir="/zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_o2o/"
exp_root_dir="/zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_fewshot10k_devbest/"

exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_bart_DA/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_fewshot-v2/"
exp_root_dir="/zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_transformer_DA"
exp_root_dir="/zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_transformer_fewshot"

dataset_args="kp20k kp20k_valid2k kptimes kptimes_valid2k jptimes duc openkp openkp_valid2k stackex stackex_valid2k"

cmd="/ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks eval -data_dir /zfs1/hdaqing/rum20/kp/data/kp/json/ -exp_root_dir $exp_root_dir -testsets $dataset_args -splits test -batch_size 8 -beam_size 50 -max_length 40 --step_base 1 --data_format jsonl"

echo $cmd
echo $PWD
$cmd

#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=hdaqing

#SBATCH --partition=v100
#SBATCH --job-name=v100_pred-3d
#SBATCH --output=slurm_output/v100_pred.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=3-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long


source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
cmd="/ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks pred -data_dir /zfs1/hdaqing/rum20/kp/data/kp/json/ -exp_root_dir /zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_fewshot-v1_DA1e6_FT1e5_devbest/ -gpu 0 -batch_size 1 -beam_size 50 -max_length 40 -testsets kp20k openkp kptimes jptimes stackex kp20k_valid2k openkp_valid2k kptimes_valid2k jptimes_valid2k stackex_valid2k duc -splits test --wait_time 360000 -sleep_time 360000 --step_base 1 --data_format jsonl -gpu 0"
#cmd="python kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks pred -data_dir /zfs1/hdaqing/rum20/kp/data/kp/json/ -exp_root_dir /zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp_fewshot10k/ -gpu 0 -batch_size 1 -beam_size 50 -max_length 40 -testsets kp20k openkp kptimes jptimes stackex -splits test --wait_time 0 --step_base 1 --data_format jsonl --pred_trained_only -gpu 0"

echo $cmd
$cmd

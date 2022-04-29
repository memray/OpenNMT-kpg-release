#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=hdaqing

#SBATCH --partition=titanx
#SBATCH --job-name=titanx_pred-magkp1m-3d
#SBATCH --output=slurm_output/titanx_pred-magkp1m-3d.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=3-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long


source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
cmd="/ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 kp_gen_eval_transfer.py -config config/transfer_kp/infer/keyphrase-one2seq.yml -tasks pred -data_dir /zfs1/hdaqing/rum20/kp/data/kp/json/ -exp_root_dir /zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_PTDA_selftrain/transformer-PTDA_TLtf_magkp1m_step20k-tlrs_55-round5/ckpts -gpu 0 -batch_size 32 -beam_size 1 -max_length 60 -testsets magkpcs_1m -splits train --wait_time 0 --step_base 1 --data_format jsonl"

echo $cmd
$cmd


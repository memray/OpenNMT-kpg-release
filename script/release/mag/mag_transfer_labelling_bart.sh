#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=hdaqing

#SBATCH --partition=gtx1080 # titanx gtx1080 v100
#SBATCH --time=3-00:00:00 # 6 days walltime in dd-hh:mm format

#SBATCH --job-name=mag_transfer_labelling_bart-gtx10803d
#SBATCH --output=slurm_output/mag_transfer_labelling-gtx10803d.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --qos=long

source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
cmd="/ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 kp_gen_magkp_transfer_labelling.py -config config/transfer_kp/infer/keyphrase-one2seq-controlled.yml -tasks pred -exp_root_dir /zfs1/hdaqing/rum20/kp/transfer_exps_v2/kp_PT/bart-wiki-step200k -data_dir /zfs1/hdaqing/rum20/kp/data/kp/magkp_cs/oag_v1_cs_nokp_13m/ -output_dir /zfs1/hdaqing/rum20/kp/data/kp/magkp_cs/oag_v1_cs_nokp_TLbart_13m_v2/ -gpu 0 -batch_size 32 -beam_size 1 -max_length 60"

echo $cmd
$cmd


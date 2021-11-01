#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=hdaqing

#SBATCH --partition=gtx1080
#SBATCH --job-name=one2seq-preabs-magkp+kp20kFT
#SBATCH --output=slurm_output/one2seq-preabs-magkp+kp20kFT.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Load modules
#module restore
#module load cuda/10.0.130
#module load gcc/6.3.0
#module load python/anaconda3.6-5.2.0
#source activate py36
#module unload python/anaconda3.6-5.2.0

# Run the job
export CONFIG_PATH="config/train/pt/3rd/config-kpgen-one2seq-verbatim_append-transformer-magkp+kp20kFT.yml"

cmd="python train.py -config $CONFIG_PATH"

echo $CONFIG_PATH
echo $cmd

$cmd

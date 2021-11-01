#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=v100
#SBATCH --partition=gtx1080
#SBATCH --partition=titanx
#SBATCH --account=hdaqing

#SBATCH --job-name=rnn-presabs-kp20k-E2D2
#SBATCH --output=slurm_output/rnn-presabs-kp20k-E2D2.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
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

# Run the job
export CONFIG_PATH="script/empirical_study/diverse/rnn/rnn-presabs-kp20k-E2D2.yml"
cmd="python train.py -config $CONFIG_PATH"

echo $CONFIG_PATH
echo $cmd

$cmd

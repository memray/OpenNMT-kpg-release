#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=v100
#SBATCH --partition=gtx1080
#SBATCH --partition=titanx
#SBATCH --account=hdaqing

#SBATCH --job-name=transformer-DA-kp20k-step40k-tlnp_28
#SBATCH --output=slurm_output/transformer-DA-kp20k-step40k-tlnp_28.out
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
export CONFIG_PATH="script/transfer_v2/tf/DAcompare/transformer-DA-kp20k-step40k-tlnp_28.yml"
cmd="python train.py -config $CONFIG_PATH"

echo $CONFIG_PATH
echo $cmd

$cmd

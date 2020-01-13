#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=preprocess_magkp
#SBATCH --output=slurm_output/preprocess_magkp.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long


cmd="srun python -m preprocess -config config/preprocess/config-preprocess-keyphrase-magkp.yml"

echo $cmd
$cmd

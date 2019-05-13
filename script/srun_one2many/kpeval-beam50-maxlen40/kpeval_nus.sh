#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=eval_kp_one2many_nus
#SBATCH --output=slurm_output/eval_kp_one2many_nus.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Run the job
python kp_run_eval.py -config script/srun_one2many/kpeval-beam50-maxlen40/config-test-keyphrase-one2many.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17-one2many-beam50-maxlen40/ -gpu -1 -testsets nus

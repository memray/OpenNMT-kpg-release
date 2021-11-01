#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=eval_kp_one2one_inspec
#SBATCH --output=slurm_output/eval_kp_one2one_inspec.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Run the job
python kp_gen_eval.py -config config/test/config-test-keyphrase.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17/ -output_dir output/keyphrase/meng17/ -testsets inspec -gpu -1

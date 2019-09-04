#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=eval_one2one_kp20k_valid2k-evalonly
#SBATCH --output=slurm_output/eval_one2one_kp20k_valid2k-evalonly.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Run the job
python kp_gen_eval.py -config config/test/config-test-keyphrase-one2one.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17-one2one/ -output_dir output/keyphrase/meng17-one2one/ -testsets kp20k_valid2k -gpu -1 -tasks eval



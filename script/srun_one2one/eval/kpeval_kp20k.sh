#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --partition=high-mem
#SBATCH --job-name=eval_kp_one2one_kp20k
#SBATCH --output=slurm_output/eval_kp_one2one_kp20k.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
# do not change, 32gb will OOM for one2one
#SBATCH --mem=64GB
#SBATCH --time=4-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Run the job
srun python kp_gen_eval.py -config config/test/config-test-keyphrase-one2one.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/ -output_dir output/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/ -testsets kp20k -gpu -1 -tasks pred


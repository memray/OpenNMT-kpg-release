#!/usr/bin/env bash
#SBATCH --cluster=htc
#SBATCH --partition=scavenger
#SBATCH --partition=htc
#SBATCH --account=hdaqing

#SBATCH --job-name={job_name}
#SBATCH --output={slurm_output_dir}/{job_name}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=3-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

cmd="python kp_gen_eval.py -config config/test/config-test-keyphrase-one2one.yml -tasks {task_args} -data_dir data/keyphrase/meng17/ -ckpt_dir {ckpt_dir} -output_dir {output_dir} -gpu -1 -testsets {dataset_args} -batch_size {batch_size} -beam_size {beam_size} -max_length {max_length} -beam_terminate full"

echo $cmd
echo $PWD
$cmd


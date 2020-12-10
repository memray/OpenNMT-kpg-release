#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=hdaqing

#SBATCH --partition=titanx
#SBATCH --partition=gtx1080
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_output_dir}/{job_name}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

cmd="python kp_gen_eval.py -config config/test/config-test-keyphrase-one2seq-transfer.yml -tasks {task_args} -data_dir /zfs1/hdaqing/rum20/kp/data/kp/json/ -exp_root_dir {exp_root_dir} -testsets {dataset_args} -splits test -batch_size {batch_size} -beam_size {beam_size} -max_length {max_length} -beam_terminate full --step_base {step_base} --data_format jsonl -gpu 0"
echo $cmd
echo $PWD
$cmd


#!/usr/bin/env bash
#SBATCH --cluster=smp
#SBATCH --cluster=htc
#SBATCH --partition=scavenger
#SBATCH --account=hdaqing

#SBATCH --job-name={job_name}
#SBATCH --output=slurm_output/mag_np.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found

cmd="/ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python onmt/keyphrase/extract_np.py -input_path {input_path} -output_dir {output_dir}"

echo $cmd
$cmd

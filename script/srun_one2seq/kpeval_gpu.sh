#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --partition=gtx1080
#SBATCH --constraint=ti
#SBATCH --job-name=eval_kp_one2seq_gpu
#SBATCH --output=slurm_output/eval_kp_one2seq_gpu.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=3-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Run the job
if (($# == 0)); then
  echo -e "Please pass argumensts -a <task1> <task2> ... -c <ckpt_dir> -o <output_dir> -g <gpu_id> -b <batch_size> -s <beam_size> -l <max_length> -t full/topbeam -e -p -d <dataset1> <dataset2>..."
  exit 2
fi
while getopts ":a:c:o:g:b:s:l:t:e:p:d:" opt; do
  case $opt in
    a)
      echo "-a (tasks) was triggered, Parameter: $OPTARG" >&2
      tasks+=("$OPTARG")
      task_args=()
      for TASK in "${tasks[@]}"
      do
        task_args=$task_args" $TASK"
      done
      ;;
    c)
      echo "-c (ckpt_dir) was triggered, Parameter: $OPTARG" >&2
      ckpt_dir=$OPTARG
      ;;
    o)
      echo "-o (output_dir) was triggered, Parameter: $OPTARG" >&2
      output_dir=$OPTARG
      ;;
    g)
      echo "-g (gpu_id, -1 means using cpu) was triggered, Parameter: $OPTARG" >&2
      gpu_id=$OPTARG
      ;;
    b)
      echo "-b (batch_size) was triggered, Parameter: $OPTARG" >&2
      batch_size=$OPTARG
      ;;
    s)
      echo "-s (beam_size) was triggered, Parameter: $OPTARG" >&2
      beam_size=$OPTARG
      ;;
    l)
      echo "-l (max_search_length) was triggered, Parameter: $OPTARG" >&2
      max_length=$OPTARG
      ;;
    t)
      echo "-t (beam_terminate=full/topbeam) was triggered, Parameter: $OPTARG" >&2
      beam_terminate=$OPTARG
      ;;
    e)
      echo "-e (eval_topbeam) was triggered, Parameter: $OPTARG" >&2
      eval_topbeam=true
      ;;
    p)
      echo "-p (onepass) was triggered, Parameter: $OPTARG" >&2
      onepass=true
      ;;
    i)
      echo "-i (ignore_existing) was triggered, Parameter: $OPTARG" >&2
      ignore_existing=true
      ;;
    d)
      echo "-d (datasets) was triggered, Parameter: $OPTARG" >&2
      datasets+=("$OPTARG")

      dataset_args=()
      for DATASET in "${datasets[@]}"
      do
        dataset_args=$dataset_args" $DATASET"
      done
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

cmd="python kp_gen_eval.py -config config/test/config-test-keyphrase-one2seq.yml -tasks $task_args -data_dir data/keyphrase/meng17/ -ckpt_dir $ckpt_dir -output_dir $output_dir -gpu 0 -batch_size $batch_size -beam_size $beam_size -max_length $max_length -beam_terminate $beam_terminate -testsets $dataset_args"

if [ "$eval_topbeam" = true ]; then
  cmd="${cmd} --eval_topbeam"
fi
if [ "$onepass" = true ]; then
  cmd="${cmd} --onepass"
fi
if [ "$ignore_existing" = true ]; then
  cmd="${cmd} --ignore_existing"
fi
if [ -z "$beam_terminate" ]; then
  echo "-t beam_terminate must be given, full/topbeam, exiting"
fi

echo $cmd
echo $PWD
$cmd

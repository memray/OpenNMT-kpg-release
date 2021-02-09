#!/usr/bin/env bash
PROJECT_DIR="/zfs1/hdaqing/rum20/kp/OpenNMT-kpg-transfer"

CURDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEMPLATE_PATH="$CURDIR/kpeval_cpu_template.sh"

echo $0
echo $PROJECT_DIR
slurm_output_dir="$PROJECT_DIR/slurm_output"

task_args="eval" # pred or eval
batch_size=1
# beam_size=200
# max_length=6
beam_size=128
max_length=8
step_base=1

# evaluate with all predictions
datasets=(kp20k kp20k_valid2k kptimes kptimes_valid2k jptimes jptimes_valid2k openkp openkp_valid2k stackex stackex_valid2k duc)

for dataset in "${datasets[@]}"
do
    exp_root_dir="/zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp_o2o/"
    EXP_NAME="$task_args-o2o-$dataset-bs$beam_size"
    DUMP_SCRIPT_PATH="$CURDIR/$EXP_NAME.sh"
    replaces="s/{job_name}/$EXP_NAME/;";
    replaces="$replaces s|{task_args}|$task_args|g;";
    replaces="$replaces s|{dataset_args}|$dataset|g;";
    replaces="$replaces s|{exp_root_dir}|$exp_root_dir|g;";
    replaces="$replaces s|{slurm_output_dir}|$slurm_output_dir|g;";
    replaces="$replaces s|{batch_size}|$batch_size|g;";
    replaces="$replaces s|{max_length}|$max_length|g;";
    replaces="$replaces s|{beam_size}|$beam_size|g;";
    replaces="$replaces s|{step_base}|$step_base|g;";
    cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}

    echo $EXP_NAME
    echo $DUMP_SCRIPT_PATH
    sbatch $DUMP_SCRIPT_PATH

done


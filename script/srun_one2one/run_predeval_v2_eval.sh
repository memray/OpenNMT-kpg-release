#!/usr/bin/env bash

task_args="eval" # pred or eval
batch_size="4"
max_length="6"

PROJECT_DIR="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/"
CURDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEMPLATE_PATH="$CURDIR/eval_cpu_template.sh"

echo $0
echo $PROJECT_DIR
echo $CURDIR

# evaluate with all predictions
datasets=(duc inspec semeval kp20k kp20k_valid2k krapivin nus)
datasets=(kp20k)

beam_size="32" # 8, 16, 32, 64, 200
slurm_output_dir="$PROJECT_DIR/slurm_output"

for dataset in "${datasets[@]}"
do
    # V1 models ( -i true means ignore_existing pred or eval, -p true means onepass)
#    ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2one/meng17-one2one-kp20k/"
#    output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k/meng17-one2one-fullbeam/meng17-one2one-beam$beam_size-maxlen6/"
#    EXP_NAME="one2one-beamsearch-$task_args-$dataset-beam$beam_size"
#    DUMP_SCRIPT_PATH="$CURDIR/$EXP_NAME.sh"
#    replaces="s/{job_name}/$EXP_NAME/;";
#    replaces="$replaces s|{task_args}|$task_args|g;";
#    replaces="$replaces s|{dataset_args}|$dataset|g;";
#    replaces="$replaces s|{ckpt_dir}|$ckpt_dir|g;";
#    replaces="$replaces s|{output_dir}|$output_dir|g;";
#    replaces="$replaces s|{slurm_output_dir}|$slurm_output_dir|g;";
#    replaces="$replaces s|{batch_size}|$batch_size|g;";
#    replaces="$replaces s|{max_length}|$max_length|g;";
#    replaces="$replaces s|{beam_size}|$beam_size|g;";
#    cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}
#
#    echo $EXP_NAME
#    echo $DUMP_SCRIPT_PATH
#    sbatch $DUMP_SCRIPT_PATH

    # selected V1 models
#    ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/"
#    output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/"
#    EXP_NAME="one2one-v1top-$TASKS-$dataset"
#    DUMP_SCRIPT_PATH="$CURDIR/$EXP_NAME.sh"
#    replaces="s/{job_name}/$EXP_NAME/;";
#    replaces="$replaces s|{task_args}|$task_args|g;";
#    replaces="$replaces s|{dataset_args}|$dataset|g;";
#    replaces="$replaces s|{ckpt_dir}|$ckpt_dir|g;";
#    replaces="$replaces s|{output_dir}|$output_dir|g;";
#    replaces="$replaces s|{slurm_output_dir}|$PROJECT_DIR/slurm_output|g;";
#    cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}
#    sbatch ${DUMP_SCRIPT_PATH}


    # V2/V3 models
    ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2one/meng17-one2one-kp20k-v3/"
    output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-v3/meng17-one2one-fullbeam/meng17-one2one-beam$beam_size-maxlen6/"
    EXP_NAME="one2one-beamsearch-$task_args-$dataset-beam$beam_size"
    DUMP_SCRIPT_PATH="$CURDIR/$EXP_NAME.sh"
    replaces="s/{job_name}/$EXP_NAME-v3/;";
    replaces="$replaces s|{task_args}|$task_args|g;";
    replaces="$replaces s|{dataset_args}|$dataset|g;";
    replaces="$replaces s|{ckpt_dir}|$ckpt_dir|g;";
    replaces="$replaces s|{output_dir}|$output_dir|g;";
    replaces="$replaces s|{slurm_output_dir}|$slurm_output_dir|g;";
    replaces="$replaces s|{batch_size}|$batch_size|g;";
    replaces="$replaces s|{max_length}|$max_length|g;";
    replaces="$replaces s|{beam_size}|$beam_size|g;";
    cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}

    echo $EXP_NAME
    echo $DUMP_SCRIPT_PATH
    sbatch $DUMP_SCRIPT_PATH

done

